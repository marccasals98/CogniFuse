"""Word-branch shapes, regularization, cross-attention masks, and real trainer I/O."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

import test_frame_inputs as frame_fixtures
from scripts.frame_inputs import collate_frame_items, frame_batch_to_device, frame_item
from scripts.word_fusion import DecomposedWordFusion, LOSS_WEIGHTS, validate_fusion_configuration
from scripts.evaluate import evaluate, load_model, parse_args

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from model import Classifier
from train import Trainer


class WordFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.audio = torch.randn(2, 6, 6)
        self.text = torch.randn(2, 6, 6)
        self.mask = torch.tensor([[True, True, False, True, True, False],
                                  [True, False, False, False, False, False]])
        self.module = DecomposedWordFusion(6, 8, 2, 1)

    def call(self, audio=None, text=None, **kwargs):
        return self.module(self.audio if audio is None else audio,
                           self.text if text is None else text, self.mask, **kwargs)

    def fixture(self):
        fixture = frame_fixtures.FrameInputTests('test_real_training_entrypoint_one_epoch')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.params.fusion_method = 'decomposed_fixed'
        fixture.params.decomposition_dim = 8
        fixture.params.decomposition_heads = 2
        fixture.params.decomposition_cross_radius = 1
        return fixture

    def test_four_branches_fixed_weights_and_n_word_axis(self):
        result = self.call(return_auxiliary=True)
        for name in ('audio_private', 'text_private', 'shared', 'cross_modal', 'fused'):
            self.assertEqual(tuple(result[name].shape), (2, 6, 8))
            self.assertTrue((result[name][~self.mask] == 0).all())
        expected = sum(result[k] for k in ('audio_private', 'text_private', 'shared', 'cross_modal')) / 4
        torch.testing.assert_close(result['fused'], expected, rtol=0, atol=0)
        self.assertTrue((result['fusion_weights'][self.mask] == 0.25).all())
        self.assertTrue((result['fusion_weights'][~self.mask] == 0).all())
        self.assertEqual(set(result['auxiliary_losses']), set(LOSS_WEIGHTS))
        self.assertTrue(all(v.ndim == 0 and torch.isfinite(v) and v >= 0
                            for v in result['auxiliary_losses'].values()))
        self.assertEqual(self.call()['auxiliary_losses'], {})

    def test_masked_nans_do_not_affect_outputs_losses_or_backward(self):
        expected = self.call(return_auxiliary=True)
        audio, text = self.audio.clone(), self.text.clone()
        audio[~self.mask], text[~self.mask] = float('nan'), float('inf')
        audio.requires_grad_(True)
        text.requires_grad_(True)
        result = self.call(audio, text, return_auxiliary=True)
        torch.testing.assert_close(result['fused'], expected['fused'], rtol=0, atol=0)
        for key, loss in expected['auxiliary_losses'].items():
            torch.testing.assert_close(result['auxiliary_losses'][key], loss, rtol=0, atol=0)
        (result['fused'].square().sum() + sum(result['auxiliary_losses'].values())).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.module.parameters()))
        self.assertTrue((audio.grad[~self.mask] == 0).all())
        self.assertTrue((text.grad[~self.mask] == 0).all())

    def test_local_bidirectional_attention_and_global_option(self):
        original = self.call(return_attention=True)
        positions = torch.arange(6)
        local = (positions[:, None] - positions[None, :]).abs() <= 1
        for weights in original['cross_attention'].values():
            self.assertEqual(tuple(weights.shape), (2, 2, 6, 6))
            self.assertFalse(weights.requires_grad)
            allowed = (local[None, None] & self.mask[:, None, :, None]
                       & self.mask[:, None, None, :]).expand_as(weights)
            self.assertTrue((weights[~allowed] == 0).all())
            sums = weights.sum(dim=-1)
            torch.testing.assert_close(sums, self.mask[:, None].expand_as(sums).float(),
                                       rtol=1e-6, atol=1e-6)
        changed = self.text.clone()
        changed[0, 4] += torch.arange(6) * 10
        local_changed = self.call(text=changed)
        torch.testing.assert_close(original['cross_modal'][0, 0], local_changed['cross_modal'][0, 0],
                                   rtol=1e-6, atol=1e-6)
        global_module = DecomposedWordFusion(6, 8, 2, -1)
        global_module.load_state_dict(self.module.state_dict(), strict=True)
        global_before = global_module(self.audio, self.text, self.mask)
        global_after = global_module(self.audio, changed, self.mask)
        self.assertFalse(torch.allclose(global_before['cross_modal'][0, 0], global_after['cross_modal'][0, 0]))

    def test_shared_agreement_and_variance_floor_have_defined_behavior(self):
        same = self.call(text=self.audio, return_auxiliary=True)
        self.assertEqual(same['auxiliary_losses']['shared_agreement'].item(), 0.0)
        constant = self.audio[:, :1].expand(-1, 6, -1)
        repeated = self.module(constant, constant, torch.ones_like(self.mask), return_auxiliary=True)
        torch.testing.assert_close(repeated['auxiliary_losses']['shared_variance'], torch.tensor(0.99),
                                   rtol=0, atol=1e-6)
        single = self.module(self.audio[:, :1], self.text[:, :1], self.mask[:, :1], return_auxiliary=True)
        self.assertEqual(single['auxiliary_losses']['shared_variance'].item(), 0.0)

    def test_auxiliary_losses_average_recordings_and_ignore_padding(self):
        batch = self.call(return_auxiliary=True)
        individual = [self.module(self.audio[b:b+1], self.text[b:b+1], self.mask[b:b+1],
                                  return_auxiliary=True) for b in range(2)]
        for name, value in batch['auxiliary_losses'].items():
            expected = sum(r['auxiliary_losses'][name] for r in individual) / 2
            torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6)
        padded_audio = F.pad(self.audio, (0, 0, 0, 4), value=float('nan'))
        padded_text = F.pad(self.text, (0, 0, 0, 4), value=float('nan'))
        padded_mask = F.pad(self.mask, (0, 4), value=False)
        padded = self.module(padded_audio, padded_text, padded_mask, return_auxiliary=True)
        for name, value in batch['auxiliary_losses'].items():
            torch.testing.assert_close(value, padded['auxiliary_losses'][name], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(batch['fused'], padded['fused'][:, :6], rtol=1e-5, atol=1e-6)

    def test_reconstruction_targets_are_detached(self):
        with torch.no_grad():
            for decoder in (self.module.audio_decoder, self.module.text_decoder):
                decoder.weight.zero_()
                decoder.bias.zero_()
        audio, text = self.audio.clone().requires_grad_(True), self.text.clone().requires_grad_(True)
        result = self.call(audio, text, return_auxiliary=True)
        result['auxiliary_losses']['reconstruction'].backward()
        self.assertTrue((audio.grad == 0).all())
        self.assertTrue((text.grad == 0).all())
        self.assertGreater(float(self.module.audio_decoder.weight.grad.abs().sum()), 0)

    def test_loss_composition_and_zero_coefficients_keep_parameters_connected(self):
        trainer = Trainer.__new__(Trainer)
        trainer.params = SimpleNamespace()
        trainer.loss_function = torch.nn.CrossEntropyLoss()
        result = self.call(return_auxiliary=True)
        logits = result['fused'].sum(dim=1)[:, :2]
        prediction = {'logits': logits, 'auxiliary_losses': result['auxiliary_losses']}
        labels = torch.tensor([0, 1])
        total, metrics = trainer.compute_training_loss(prediction, labels)
        expected = trainer.loss_function(logits, labels) + sum(
            default * result['auxiliary_losses'][name] for name, (_, default) in LOSS_WEIGHTS.items())
        torch.testing.assert_close(total, expected, rtol=0, atol=1e-6)
        self.assertEqual(set(metrics), {'classification', 'total', *LOSS_WEIGHTS})
        for setting, _ in LOSS_WEIGHTS.values():
            setattr(trainer.params, setting, 0.0)
        total, _ = trainer.compute_training_loss(prediction, labels)
        torch.testing.assert_close(total, trainer.loss_function(logits, labels), rtol=0, atol=0)
        total.backward()
        self.assertTrue(all(p.grad is not None for p in self.module.parameters()))

    def test_model_branches_receive_classification_gradients_and_reload(self):
        fixture = self.fixture()
        params = copy.deepcopy(fixture.params)
        params.speech_feature_extractor_output_vectors_dimension = 6
        params.acoustic_word_pooling = 'attention'
        params.acoustic_word_attention_hidden_dim = 4
        model = Classifier(params, 'cpu').eval()
        frames = torch.randn(2, 12, 6)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        intervals = torch.tensor([[[0, 3], [3, 6], [6, 9], [9, 12]],
                                  [[0, 3], [3, 6], [-1, -1], [-1, -1]]])
        batch = {'frames': frames, 'frame_mask': torch.arange(12)[None] < torch.tensor([[12], [6]]),
                 'word_frame_intervals': intervals, 'audio_word_mask': mask, 'text_word_mask': mask,
                 'paired_word_mask': mask, 'text_words': torch.randn(2, 4, 3),
                 'schema_sha256': params.frame_package_schema_sha256}
        output = model(frame_batch=batch, return_auxiliary=True, return_representations=True)
        branches = output['representations']
        self.assertEqual(tuple(branches['fused'].shape), (2, 4, 8))
        for name in ('audio_private', 'text_private', 'shared', 'cross_modal'):
            branches[name].retain_grad()
        F.cross_entropy(output['logits'], torch.tensor([0, 1])).backward(retain_graph=True)
        for name in ('audio_private', 'text_private', 'shared', 'cross_modal'):
            self.assertGreater(float(branches[name].grad.abs().sum()), 0)
        model.zero_grad()
        trainer = Trainer.__new__(Trainer)
        trainer.params, trainer.loss_function = params, torch.nn.CrossEntropyLoss()
        total, _ = trainer.compute_training_loss(output, torch.tensor([0, 1]))
        total.backward()
        for group in (model.word_fusion, model.acoustic_word_pooling):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in group.parameters()), 0)
        self.assertIsNone(frames.grad)
        checkpoint = fixture.root / 'decomposition.chkpt'
        torch.save({'settings': params, 'model': model.state_dict()}, checkpoint)
        saved = torch.load(checkpoint, weights_only=False)
        loaded = load_model(saved['settings'], saved['model'], 'cpu')
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(model(frame_batch=batch), loaded(frame_batch=batch), rtol=1e-6, atol=1e-6)
        changed = dict(batch)
        changed['frames'] = frames.clone()
        changed['frames'][~batch['frame_mask']] = float('nan')
        changed['text_words'] = batch['text_words'].clone()
        changed['text_words'][~mask] = float('nan')
        with torch.no_grad():
            torch.testing.assert_close(model(frame_batch=batch), model(frame_batch=changed), rtol=0, atol=0)

    def test_cli_training_and_standalone_evaluator(self):
        fixture = self.fixture()
        fixture.params.acoustic_word_pooling = 'attention'
        fixture.params.acoustic_word_attention_hidden_dim = 4
        path, checkpoint = fixture.run_training_epoch()
        self.assertEqual(checkpoint['settings'].fusion_method, 'decomposed_fixed')
        self.assertEqual(checkpoint['settings'].decomposition_dim, 8)
        self.assertTrue(any(k.startswith('word_fusion.shared_encoder.') for k in checkpoint['model']))
        self.assertFalse(any(k.startswith('seq_to_seq_layer.') for k in checkpoint['model']))
        logs = '\n'.join(p.read_text() for p in (fixture.root / 'logs').rglob('*.log'))
        self.assertIn('shared_agreement', logs)
        output = fixture.root / 'decomposed-evaluation'
        evaluate(parse_args(['--checkpoint', str(path), '--dataset_to_evaluate', str(fixture.features / 'a1'),
                             '--output_dir', str(output), '--device', 'cpu', '--threads', '1']))
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['checkpoint_settings']['fusion_method'], 'decomposed_fixed')
        self.assertEqual(manifest['recording_ids'], ['a1'])

    def test_configuration_and_empty_recordings_are_rejected(self):
        fixture = self.fixture()
        for setting, value in [('decomposition_dim', 2), ('decomposition_heads', 3),
                               ('decomposition_cross_radius', 0), ('decomposition_cross_radius', -2),
                               ('shared_agreement_weight', -1), ('shared_variance_weight', float('nan'))]:
            params = copy.deepcopy(fixture.params)
            setattr(params, setting, value)
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                validate_fusion_configuration(params)
        wrong = copy.deepcopy(fixture.params)
        wrong.precomputed_input_type = 'words'
        with self.assertRaisesRegex(ValueError, 'requires precomputed frame'):
            Classifier(wrong, 'cpu')
        mask = self.mask.clone()
        mask[1] = False
        with self.assertRaisesRegex(ValueError, 'at least one valid word pair'):
            self.module(self.audio, self.text, mask)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
