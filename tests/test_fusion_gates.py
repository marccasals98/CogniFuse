"""Learned gate behavior, gradient paths, masking, and checkpoint integration."""

import copy
import json
import math
import unittest

import torch
from torch.nn import functional as F

import test_frame_inputs as frame_fixtures
import test_fusion_quality as quality_fixtures
from scripts.frame_inputs import collate_frame_items, frame_item
from scripts.reliability import fit_quality_normalization
from scripts.word_fusion import (BRANCH_NAMES, DecomposedWordFusion,
                                 summarize_fusion_weights, validate_fusion_configuration)
from scripts.evaluate import evaluate, load_model, parse_args
from model import Classifier
from train import Trainer


class FusionGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.audio, self.text = torch.randn(2, 5, 6), torch.randn(2, 5, 6)
        self.mask = torch.tensor([[True, True, False, True, False],
                                  [True, False, False, False, False]])
        self.quality = torch.randn(2, 5, 14)

    def module(self, method):
        return DecomposedWordFusion(6, 8, 2, 1, fusion_method=method,
                                    gate_hidden_dim=11,
                                    quality_dim=14 if method == 'reliability_gated' else 0)

    def fixture(self, ready=False):
        if ready:
            parent = quality_fixtures.FusionQualityTests(
                'test_training_checkpoint_and_evaluator_restore_normalization')
            parent.setUp()
            self.addCleanup(parent.doCleanups)
            dataset = parent.ready_dataset()
            fixture = parent.fixture
            fixture.params.frame_quality_normalization = fit_quality_normalization(dataset)
        else:
            fixture = frame_fixtures.FrameInputTests('test_real_training_entrypoint_one_epoch')
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
        fixture.params.decomposition_dim = 8
        fixture.params.decomposition_heads = 2
        fixture.params.fusion_gate_hidden_dim = 11
        fixture.params.acoustic_word_pooling = 'attention'
        fixture.params.acoustic_word_attention_hidden_dim = 4
        return fixture

    def test_gate_weights_form_masked_convex_combination(self):
        for method in ('decomposed_gated', 'reliability_gated'):
            with self.subTest(method=method):
                module = self.module(method)
                result = module(self.audio, self.text, self.mask, reliability_features=self.quality)
                weights = result['fusion_weights']
                self.assertEqual(tuple(weights.shape), (2, 5, 4))
                torch.testing.assert_close(weights.sum(-1), self.mask.float())
                self.assertTrue((weights[self.mask] > 0).all())
                self.assertTrue((weights[~self.mask] == 0).all())
                expected = sum(weights[..., j:j+1] * result[name]
                               for j, name in enumerate(BRANCH_NAMES))
                torch.testing.assert_close(result['fused'], expected, rtol=0, atol=0)
                self.assertTrue((result['fused'][~self.mask] == 0).all())
                self.assertFalse(torch.allclose(weights[self.mask], torch.full_like(weights[self.mask], .25)))

    def test_quality_only_changes_reliability_gate_with_branches_fixed(self):
        changed = self.quality.clone()
        changed[0, 0, 0] += 3
        for method in ('decomposed_gated', 'reliability_gated'):
            module = self.module(method)
            if method == 'reliability_gated':
                # Controlled synthetic routing tests wiring, not learned reliability behavior.
                with torch.no_grad():
                    for parameter in module.gate.parameters():
                        parameter.zero_()
                    module.gate[0].weight[0, 4 * 8] = 1
                    module.gate[2].weight[0, 0] = 1
            before = module(self.audio, self.text, self.mask, reliability_features=self.quality)
            after = module(self.audio, self.text, self.mask, reliability_features=changed)
            for name in BRANCH_NAMES:
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
            if method == 'decomposed_gated':
                torch.testing.assert_close(before['fusion_weights'], after['fusion_weights'], rtol=0, atol=0)
                torch.testing.assert_close(before['fused'], after['fused'], rtol=0, atol=0)
            else:
                self.assertFalse(torch.allclose(before['fusion_weights'][0, 0], after['fusion_weights'][0, 0]))
                self.assertFalse(torch.allclose(before['fused'][0, 0], after['fused'][0, 0]))
                torch.testing.assert_close(before['fusion_weights'][0, 1:], after['fusion_weights'][0, 1:],
                                           rtol=0, atol=0)

    def test_classification_gradients_reach_gates_branches_and_quality(self):
        for method in ('decomposed_gated', 'reliability_gated'):
            module = self.module(method)
            q = self.quality.clone().requires_grad_(True)
            result = module(self.audio, self.text, self.mask, reliability_features=q)
            for name in BRANCH_NAMES:
                result[name].retain_grad()
            head = torch.nn.Linear(8, 2)
            pooled = result['fused'].sum(1) / self.mask.sum(1)[:, None]
            F.cross_entropy(head(pooled), torch.tensor([0, 1])).backward()
            for name in BRANCH_NAMES:
                self.assertGreater(float(result[name].grad[self.mask].abs().sum()), 0)
            for name, parameter in module.named_parameters():
                if 'decoder' in name:
                    continue
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for parameter in module.gate.parameters():
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
            if method == 'reliability_gated':
                self.assertGreater(float(q.grad[self.mask].abs().sum()), 0)
                self.assertTrue((q.grad[~self.mask] == 0).all())
            else:
                self.assertIsNone(q.grad)

    def test_masked_nonfinite_values_do_not_change_outputs_or_gradients(self):
        for method in ('decomposed_gated', 'reliability_gated'):
            module = self.module(method)
            expected = module(self.audio, self.text, self.mask, return_auxiliary=True,
                              reliability_features=self.quality)
            audio, text, quality = self.audio.clone(), self.text.clone(), self.quality.clone()
            audio[~self.mask], text[~self.mask], quality[~self.mask] = float('nan'), float('inf'), float('nan')
            audio.requires_grad_()
            quality.requires_grad_()
            result = module(audio, text, self.mask, return_auxiliary=True, reliability_features=quality)
            torch.testing.assert_close(result['fusion_weights'], expected['fusion_weights'], rtol=0, atol=0)
            torch.testing.assert_close(result['fused'], expected['fused'], rtol=0, atol=0)
            (result['fused'].square().sum() + sum(result['auxiliary_losses'].values())).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in module.parameters()))
            self.assertTrue((audio.grad[~self.mask] == 0).all())
            if quality.grad is not None:
                self.assertTrue((quality.grad[~self.mask] == 0).all())

    def test_padding_and_individual_recording_invariance(self):
        for method in ('decomposed_gated', 'reliability_gated'):
            module = self.module(method).eval()
            result = module(self.audio, self.text, self.mask, reliability_features=self.quality)
            padded = module(F.pad(self.audio, (0, 0, 0, 3), value=float('nan')),
                            F.pad(self.text, (0, 0, 0, 3), value=float('nan')),
                            F.pad(self.mask, (0, 3), value=False),
                            reliability_features=F.pad(self.quality, (0, 0, 0, 3), value=float('nan')))
            torch.testing.assert_close(result['fused'], padded['fused'][:, :5], rtol=1e-5, atol=1e-6)
            for b in range(2):
                single = module(self.audio[b:b+1], self.text[b:b+1], self.mask[b:b+1],
                                reliability_features=self.quality[b:b+1])
                torch.testing.assert_close(result['fusion_weights'][b:b+1], single['fusion_weights'],
                                           rtol=1e-5, atol=1e-6)

    def test_summary_averages_valid_words_within_each_recording(self):
        weights = self.mask[..., None].float().expand(-1, -1, 4) / 4
        weights[0, 0] = torch.tensor([1., 0., 0., 0.])
        summary = summarize_fusion_weights(weights.requires_grad_(), self.mask)
        self.assertEqual(summary['branch_order'], BRANCH_NAMES)
        self.assertEqual(summary['valid_words'].tolist(), [3, 1])
        torch.testing.assert_close(summary['mean_weights'][0], torch.tensor([.5, 1/6, 1/6, 1/6]))
        torch.testing.assert_close(summary['mean_entropy'], torch.tensor([2 * math.log(4) / 3, math.log(4)]))
        self.assertFalse(summary['mean_weights'].requires_grad)

    def test_reliability_features_and_configuration_are_validated(self):
        module = self.module('reliability_gated')
        for quality in (None, self.quality[..., :13], self.quality.double(),
                        self.quality.masked_fill(self.mask[..., None], float('nan'))):
            with self.subTest(quality_type=type(quality)), self.assertRaises(ValueError):
                module(self.audio, self.text, self.mask, reliability_features=quality)
        fixture = self.fixture()
        fixture.params.fusion_method = 'reliability_gated'
        with self.assertRaisesRegex(ValueError, 'ready quality packages'):
            Classifier(fixture.params, 'cpu')
        for hidden in (0, -1, 1.2, True):
            fixture.params.fusion_gate_hidden_dim = hidden
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                validate_fusion_configuration(fixture.params)

    def test_model_inspection_reload_and_normalization_identity(self):
        fixture = self.fixture(ready=True)
        fixture.params.fusion_method = 'reliability_gated'
        model = Classifier(fixture.params, 'cpu').eval()
        batch = collate_frame_items([frame_item(fixture.features / uid, label,
                        quality_state=fixture.params.frame_quality_normalization)
                        for uid, label in [('a1', 1), ('c1', 0)]])
        output = model(frame_batch=batch, return_fusion_weights=True, return_auxiliary=True)
        self.assertNotIn('representations', output)
        self.assertFalse(output['fusion_weights'].requires_grad)
        self.assertTrue(torch.equal(output['fusion_mask'], batch['paired_word_mask']))
        torch.testing.assert_close(output['fusion_weights'].sum(-1), batch['paired_word_mask'].float())
        trainer = Trainer.__new__(Trainer)
        trainer.params, trainer.loss_function = fixture.params, torch.nn.CrossEntropyLoss()
        loss, terms = trainer.compute_training_loss(output, batch['labels'])
        self.assertIn('shared_agreement', terms)
        loss.backward()
        for parameter in model.word_fusion.gate.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        saved = fixture.root / 'gate.chkpt'
        torch.save({'settings': fixture.params, 'model': model.state_dict()}, saved)
        checkpoint = torch.load(saved, weights_only=False)
        loaded = load_model(checkpoint['settings'], checkpoint['model'], 'cpu')
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, loaded.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            restored = loaded(frame_batch=batch, return_fusion_weights=True)
        # Grad-enabled and inference MHA kernels can differ by float32 rounding.
        torch.testing.assert_close(output['logits'], restored['logits'], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(output['fusion_weights'], restored['fusion_weights'], rtol=1e-6, atol=1e-6)
        bad = dict(batch, quality_normalization_sha256='different-training-fold')
        with self.assertRaisesRegex(ValueError, 'normalization state'):
            loaded(frame_batch=bad)
        bad = {k: v for k, v in batch.items() if k != 'reliability_features'}
        with self.assertRaisesRegex(ValueError, 'matching.*features'):
            loaded(frame_batch=bad)
        wrong = copy.deepcopy(fixture.params)
        wrong.fusion_gate_hidden_dim += 1
        with self.assertRaises(RuntimeError):
            load_model(wrong, checkpoint['model'], 'cpu')

    def test_cli_conventional_training_and_standalone_evaluation(self):
        self.check_training_and_evaluation('decomposed_gated', ready=False)

    def test_cli_reliability_training_and_standalone_evaluation(self):
        self.check_training_and_evaluation('reliability_gated', ready=True)

    def check_training_and_evaluation(self, method, ready):
        fixture = self.fixture(ready=ready)
        fixture.params.fusion_method = method
        path, checkpoint = fixture.run_training_epoch()
        settings = checkpoint['settings']
        self.assertEqual(settings.fusion_method, method)
        self.assertEqual(settings.fusion_gate_hidden_dim, 11)
        self.assertIn(method, settings.model_name)
        self.assertIn('word_fusion.gate.0.weight', checkpoint['model'])
        self.assertIn('word_fusion.gate.2.weight', checkpoint['model'])
        logs = '\n'.join(p.read_text() for p in (fixture.root / 'logs').rglob('*.log'))
        self.assertIn('shared_agreement', logs)
        if ready:
            self.assertEqual(settings.frame_quality_normalization['fit_uids'], ['a2', 'c2'])
        output = fixture.root / 'gated-evaluation'
        evaluate(parse_args(['--checkpoint', str(path), '--dataset_to_evaluate', str(fixture.features / 'a1'),
                             '--output_dir', str(output), '--device', 'cpu', '--threads', '1']))
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['checkpoint_settings']['fusion_method'], method)
        self.assertEqual(manifest['recording_ids'], ['a1'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
