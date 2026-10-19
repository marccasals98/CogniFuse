"""Check paired/bootstrap statistics and real classifier checkpoint inference."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import f1_score
import torch

from scripts.evaluate import (EvaluationDataset, bootstrap_macro_f1, evaluate,
                              file_identity, load_model, parse_args, read_labels)
from scripts.settings import TRAIN_DEFAULT_SETTINGS


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir='/tmp', prefix='cognifuse-evaluate-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.features = self.root / 'features'
        self.features.mkdir()
        self.parameters = SimpleNamespace(**{**TRAIN_DEFAULT_SETTINGS,
            'dataset': 'adresso', 'number_classes': 2, 'model_name': 'fixture',
            'precomputed_features_dir': str(self.features),
            'precomputed_audio_suffix': '_audio.pt', 'precomputed_text_suffix': '_text.pt',
            'speech_feature_extractor_output_vectors_dimension': 4,
            'text_feature_extractor_output_vectors_dimension': 4,
            'classifier_hidden_layers': 0, 'classifier_hidden_layers_width': 8,
            'classifier_layer_drop_out': .2, 'fold': 1})
        for i, uid in enumerate(('r1', 'r2', 'r3', 'r4')):
            for suffix in ('_audio', '_text'):
                torch.save(torch.ones(i + 2, 4), self.features / (uid + suffix + '.pt'))
                mask = torch.ones(i + 2, dtype=torch.bool)
                mask[-1] = False
                torch.save(mask, self.features / (uid + suffix + '_mask.pt'))
        model = load_model(self.parameters, self.model_weights(), 'cpu')
        self.checkpoint = self.root / 'fixture.chkpt'
        torch.save({'model': model.state_dict(), 'settings': self.parameters,
                    'training_variables': {'epoch': 7}}, self.checkpoint)
        self.labels = self.root / 'labels.csv'
        self.labels.write_text('ID,Dx,patient\nr4,ProbableAD,p2\nr2,Control,p1\nr1,Control,p1\nr3,ProbableAD,p2\n')

    def model_weights(self):
        scripts_dir = str(Path(__file__).resolve().parents[1] / 'scripts')
        import sys
        sys.path.insert(0, scripts_dir)
        try:
            from model import Classifier
            model = Classifier(self.parameters, 'cpu')
        finally:
            sys.path.remove(scripts_dir)
        output = model.classifier_layer.fully_connected_layer.get_submodule('classifier_layer_output_layer.0')
        with torch.no_grad():
            output.weight.zero_()
            output.bias.copy_(torch.tensor([0., 1.]))
        return model.state_dict()

    def args(self, name='scored', labels=True):
        argv = ['--checkpoint', str(self.checkpoint), '--dataset_to_evaluate', str(self.features),
                '--output_dir', str(self.root / name), '--device', 'cpu', '--threads', '1',
                '--bootstrap_samples', '300']
        if labels:
            argv += ['--labels_path', str(self.labels)]
        return parse_args(argv)

    def test_recording_bootstrap_matches_paired_sklearn_reference(self):
        truth, prediction = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1])
        result, values = bootstrap_macro_f1(truth, prediction, 3, samples=300, seed=42)
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(truth), size=(300, len(truth)))
        expected = [f1_score(truth[i], prediction[i], labels=[0, 1, 2], average='macro', zero_division=0) for i in indices]
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose([result['lower'], result['upper']], np.quantile(expected, [.025, .975]))
        _, repeated = bootstrap_macro_f1(truth, prediction, 3, samples=300, seed=42)
        np.testing.assert_array_equal(values, repeated)

    def test_group_bootstrap_keeps_participant_recordings_together(self):
        truth, prediction = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
        result, values = bootstrap_macro_f1(truth, prediction, 2, samples=100, seed=2,
                                           groups=['p1', 'p1', 'p2', 'p2'])
        rng = np.random.default_rng(2)
        members = [np.array([0, 1]), np.array([2, 3])]
        expected = []
        for _ in range(100):
            selected = rng.integers(0, 2, size=2)
            indices = np.concatenate([members[i] for i in selected])
            expected.append(f1_score(truth[indices], prediction[indices], labels=[0, 1], average='macro', zero_division=0))
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-15)
        self.assertEqual(result['resampling_units'], 2)

    def test_labels_match_ids_instead_of_csv_order(self):
        labels, groups = read_labels(self.labels, ['r1', 'r2', 'r3', 'r4'], ['cn', 'ad'], 'adresso', group_column='patient')
        self.assertEqual(labels, [0, 0, 1, 1])
        self.assertEqual(groups, ['p1', 'p1', 'p2', 'p2'])

    def test_duplicate_missing_extra_and_unknown_labels_fail(self):
        examples = ['ID,Dx\nr1,Control\nr1,ProbableAD\n', 'ID,Dx\nr1,Control\n',
                    'ID,Dx\nr1,Control\nextra,ProbableAD\n', 'ID,Dx\nr1,unknown\n']
        for text in examples:
            self.labels.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_labels(self.labels, ['r1', 'r2'], ['cn', 'ad'], 'adresso')

    def test_missing_masks_pairs_duplicate_ids_and_dimensions_fail(self):
        mask = self.features / 'r1_audio_mask.pt'
        mask.unlink()
        with self.assertRaises(FileNotFoundError):
            EvaluationDataset(self.features, self.parameters)
        torch.save(torch.ones(2, dtype=torch.bool), mask)
        audio = self.features / 'r1_audio.pt'
        audio.unlink()
        with self.assertRaisesRegex(ValueError, 'complete audio/text'):
            EvaluationDataset(self.features, self.parameters)
        torch.save(torch.ones(2, 5), audio)
        dataset = EvaluationDataset(self.features, self.parameters)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            dataset[0]
        nested = self.features / 'other'
        nested.mkdir()
        torch.save(torch.ones(2, 4), nested / 'r1_audio.pt')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            EvaluationDataset(self.features, self.parameters)

    def test_frozen_checkpoint_inference_macro_f1_and_roundtrip_outputs(self):
        before = file_identity(self.checkpoint)
        loaded_models = []
        original_loader = load_model
        def loader(*args):
            model = original_loader(*args)
            model.forward = unittest.mock.Mock(wraps=model.forward)
            loaded_models.append(model)
            return model
        with patch('scripts.evaluate.load_model', side_effect=loader):
            metrics = evaluate(self.args())
        self.assertEqual(metrics['status'], 'evaluated')
        self.assertAlmostEqual(metrics['macro_f1'], 1 / 3)
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(loaded_models[0].forward.call_count, 4)
        self.assertFalse(loaded_models[0].training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in loaded_models[0].parameters()))
        self.assertEqual(file_identity(self.checkpoint), before)
        output = self.root / 'scored'
        with (output / 'predictions.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r['uid'] for r in rows], ['r1', 'r2', 'r3', 'r4'])
        self.assertEqual([r['prediction'] for r in rows], ['1'] * 4)
        self.assertTrue(all(abs(float(r['probability_0']) + float(r['probability_1']) - 1) < 1e-6 for r in rows))
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['checkpoint']['sha256'], before['sha256'])
        self.assertEqual(manifest['checkpoint_epoch'], 7)
        self.assertEqual(len(manifest['input_files']), 16)
        self.assertEqual(len((output / 'bootstrap_scores.csv').read_text().splitlines()), 301)

    def test_cv_checkpoint_selection_metadata_survives_evaluation(self):
        checkpoint = torch.load(self.checkpoint, weights_only=False)
        checkpoint['settings'].fold_results_path = 'fold_0.json'
        checkpoint['selection_protocol'] = 'best_validation_checkpoint'
        checkpoint['training_variables'].update(best_model_epoch=3, best_model_step=100)
        torch.save(checkpoint, self.checkpoint)
        args = self.args('selected-checkpoint', labels=False)
        evaluate(args)
        manifest = json.loads((args.output_dir / 'manifest.json').read_text())
        self.assertEqual(manifest['selection_protocol'], 'best_validation_checkpoint')
        self.assertEqual(manifest['selected_epoch'], 3)
        self.assertEqual(manifest['selected_step'], 100)

    def test_inference_without_labels_does_not_invent_scores(self):
        metrics = evaluate(self.args('unlabeled', labels=False))
        self.assertEqual(metrics['status'], 'inference_only')
        self.assertNotIn('macro_f1', metrics)
        self.assertFalse((self.root / 'unlabeled/bootstrap_scores.csv').exists())

    def test_output_no_overwrite_and_changed_inputs_detected(self):
        evaluate(self.args())
        before = {p.name: p.read_bytes() for p in (self.root / 'scored').iterdir()}
        with self.assertRaises(FileExistsError):
            evaluate(self.args())
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / 'scored').iterdir()})
        dataset = EvaluationDataset(self.features, self.parameters)
        dataset[0]
        torch.save(torch.ones(2, 4) * 2, self.features / 'r1_audio.pt')
        with self.assertRaisesRegex(ValueError, 'changed'):
            dataset.verify_inputs()

    def test_seed_and_invalid_bootstrap_settings(self):
        for truth, pred, count, samples, confidence in [([], [], 2, 2, .95),
                 ([0], [2], 2, 2, .95), ([0], [0], 2, 1, .95), ([0], [0], 2, 2, 1.)]:
            with self.assertRaises(ValueError):
                bootstrap_macro_f1(truth, pred, count, samples=samples, confidence_level=confidence)


if __name__ == '__main__':
    unittest.main()
