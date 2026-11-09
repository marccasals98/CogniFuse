"""Learned local frame pooling: weighting, masks, gradients, and checkpoint I/O."""

import copy
import json
from pathlib import Path
import sys
import unittest

import torch

import test_frame_inputs as frame_fixtures
from scripts.frame_inputs import AttentionAcousticWordPooling, MeanAcousticWordPooling
from scripts.evaluate import evaluate, load_model, parse_args

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from model import Classifier


class AcousticWordAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.frames = torch.randn(2, 7, 3)
        self.intervals = torch.tensor([[[0, 3], [2, 5], [-1, -1]],
                                       [[1, 4], [-1, -1], [-1, -1]]])
        self.audio_mask = torch.tensor([[True, True, False], [True, False, False]])
        self.frame_mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
        self.pool = AttentionAcousticWordPooling(3, 4)

    def call(self, frames=None, diagnostics=False):
        return self.pool(self.frames if frames is None else frames, self.intervals,
                         self.audio_mask, self.frame_mask, return_attention=diagnostics)

    def test_normalization_locality_padding_and_detached_diagnostics(self):
        output, rows = self.call(diagnostics=True)
        self.assertEqual(tuple(output.shape), (2, 3, 3))
        self.assertTrue((output[~self.audio_mask] == 0).all())
        self.assertEqual([(r['batch_index'], r['word_index']) for r in rows], [(0, 0), (0, 1), (1, 0)])
        for row in rows:
            weights = row['weights']
            self.assertEqual(len(weights), row['end_frame_exclusive'] - row['start_frame'])
            torch.testing.assert_close(weights.sum(), torch.tensor(1.0), rtol=0, atol=1e-7)
            self.assertTrue((weights > 0).all())
            self.assertFalse(weights.requires_grad)
        changed = self.frames.clone()
        changed[0, 5:] = 10000  # Valid frames outside every window.
        changed[1, 4:] = float('nan')  # Padded frames must never reach the scorer.
        torch.testing.assert_close(output, self.call(changed), rtol=0, atol=0)
        # Changing a frame belonging only to the first word cannot affect the second.
        changed = self.frames.clone()
        changed[0, 0] = 10000
        other = self.call(changed)
        torch.testing.assert_close(output[0, 1], other[0, 1], rtol=0, atol=0)
        self.assertFalse(torch.equal(output[0, 0], other[0, 0]))
        single = self.pool(self.frames[:1], self.intervals[:1], self.audio_mask[:1], self.frame_mask[:1])
        torch.testing.assert_close(output[:1], single, rtol=0, atol=0)

    def test_uniform_scores_recover_mean_and_single_frame_is_identity(self):
        with torch.no_grad():
            self.pool.scorer[-1].weight.zero_()
        output = self.call()
        means = MeanAcousticWordPooling()(self.frames, self.intervals, self.audio_mask, self.frame_mask)
        torch.testing.assert_close(output, means, rtol=1e-6, atol=1e-7)
        intervals = torch.tensor([[[2, 3]]])
        output, rows = self.pool(self.frames[:1], intervals, torch.ones(1, 1, dtype=torch.bool),
                                 self.frame_mask[:1], return_attention=True)
        torch.testing.assert_close(output[0, 0], self.frames[0, 2], rtol=0, atol=0)
        self.assertEqual(rows[0]['weights'].tolist(), [1.0])

    def test_gradients_update_scorer_and_exclude_unused_frames(self):
        frames = self.frames.clone().requires_grad_(True)
        before = {name: value.detach().clone() for name, value in self.pool.named_parameters()}
        optimizer = torch.optim.SGD(self.pool.parameters(), lr=0.1)
        output = self.call(frames)
        output.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            and p.grad.abs().sum() > 0 for p in self.pool.parameters()))
        self.assertTrue((frames.grad[0, 5:] == 0).all())
        self.assertTrue((frames.grad[1, 0] == 0).all())
        self.assertTrue((frames.grad[1, 4:] == 0).all())
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in self.pool.named_parameters()))

    def test_empty_rows_and_bad_configuration_or_intervals(self):
        output, rows = self.pool(self.frames, self.intervals, torch.zeros_like(self.audio_mask),
                                 self.frame_mask, return_attention=True)
        self.assertTrue((output == 0).all())
        self.assertEqual(rows, [])
        for width in (0, -1, True):
            with self.assertRaisesRegex(ValueError, 'positive integers'):
                AttentionAcousticWordPooling(3, width)
        bad_mask = self.frame_mask.clone()
        bad_mask[0, 1] = False
        with self.assertRaisesRegex(ValueError, 'includes invalid'):
            self.pool(self.frames, self.intervals, self.audio_mask, bad_mask)
        with self.assertRaisesRegex(ValueError, 'width differs'):
            self.pool(self.frames[..., :2], self.intervals, self.audio_mask, self.frame_mask)

    def test_classification_gradient_reaches_attention_with_frozen_frames(self):
        fixture = frame_fixtures.FrameInputTests('test_real_training_entrypoint_one_epoch')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        params = copy.deepcopy(fixture.params)
        params.acoustic_word_pooling = 'attention'
        params.acoustic_word_attention_hidden_dim = 4
        # Three acoustic channels avoid the nearly invariant two-channel LayerNorm
        # of the tiny packaging fixture; this checks actual classifier gradients.
        params.speech_feature_extractor_output_vectors_dimension = 3
        model = Classifier(params, 'cpu').eval()
        batch = {'frames': self.frames.detach(), 'word_frame_intervals': self.intervals,
                 'audio_word_mask': self.audio_mask, 'frame_mask': self.frame_mask,
                 'text_words': torch.randn(2, 3, 3), 'text_word_mask': self.audio_mask,
                 'paired_word_mask': self.audio_mask,
                 'schema_sha256': params.frame_package_schema_sha256}
        logits = model(frame_batch=batch)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            and p.grad.abs().sum() > 0 for p in model.acoustic_word_pooling.parameters()))
        self.assertIsNone(batch['frames'].grad)
        changed = dict(batch)
        changed['frames'] = self.frames.clone()
        changed['frames'][1, 4:] = 10000
        torch.testing.assert_close(logits, model(frame_batch=changed), rtol=0, atol=0)
        wrong = copy.deepcopy(params)
        wrong.precomputed_input_type = 'words'
        with self.assertRaisesRegex(ValueError, 'requires precomputed_input_type=frames'):
            Classifier(wrong, 'cpu')

    def test_attention_training_checkpoint_reload_and_standalone_evaluation(self):
        fixture = frame_fixtures.FrameInputTests('test_real_training_entrypoint_one_epoch')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.params.acoustic_word_pooling = 'attention'
        fixture.params.acoustic_word_attention_hidden_dim = 4
        checkpoint_path, checkpoint = fixture.run_training_epoch()
        settings = checkpoint['settings']
        self.assertEqual(settings.acoustic_word_pooling, 'attention')
        self.assertEqual(settings.acoustic_word_attention_hidden_dim, 4)
        self.assertTrue(any(k.startswith('acoustic_word_pooling.scorer.') for k in checkpoint['model']))
        loaded = load_model(settings, checkpoint['model'], 'cpu')
        self.assertIsInstance(loaded.acoustic_word_pooling, AttentionAcousticWordPooling)
        arguments = ['--checkpoint', str(checkpoint_path), '--dataset_to_evaluate', str(fixture.features / 'a1'),
                     '--device', 'cpu', '--threads', '1']
        predictions = []
        for suffix in ('first', 'reload'):
            output = fixture.root / suffix
            evaluate(parse_args([*arguments, '--output_dir', str(output)]))
            predictions.append((output / 'predictions.csv').read_text())
        self.assertEqual(predictions[0], predictions[1])
        manifest = json.loads((fixture.root / 'reload/manifest.json').read_text())
        self.assertEqual(manifest['recording_ids'], ['a1'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
