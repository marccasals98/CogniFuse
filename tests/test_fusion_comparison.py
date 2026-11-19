"""Development cohort readiness and frozen, matched three-mode CV audits."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import torch

import test_fusion_quality as quality_fixtures
from scripts.word_fusion import DECOMPOSED_METHODS, DecomposedWordFusion
from utils.package_fusion_frames import digest
from utils.prepare_fusion_cohort import cohort_plan, prepare_cohort, read_json, write_json
from utils.compare_fusion_modes import (prepare_experiment, run_variant, summarize,
                                       verify_inputs, submit, variant_command)


class FusionComparisonTests(unittest.TestCase):
    def setUp(self):
        self.parent = quality_fixtures.FusionQualityTests('test_training_checkpoint_and_evaluator_restore_normalization')
        self.parent.setUp()
        self.addCleanup(self.parent.doCleanups)
        self.parent.ready_dataset()
        self.fixture = self.parent.fixture
        self.root = self.fixture.root
        self.source = self.root / 'legacy_source'
        self.source.mkdir()
        self.uids = ['a1', 'a2', 'c1', 'c2']
        self.split_hashes = {str(p): digest(p) for p in (self.fixture.dataset_root / 'train/splits').glob('*.npy')}
        write_json(self.source / 'config.json', {'dataset_root': str(self.fixture.dataset_root),
            'split_sources': [{'path': p, 'sha256': h} for p, h in self.split_hashes.items()]})
        write_json(self.source / 'summary.json', {'records': {
            'train/' + uid: {'status': 'complete'} for uid in self.uids}})
        self.cohort = self.root / 'cohort'
        (self.cohort / 'quality/train').mkdir(parents=True)
        receipts = {}
        for uid in self.uids:
            shutil.copytree(self.fixture.features / uid, self.cohort / 'quality/train' / uid)
            receipts[uid] = {'metadata_sha256': digest(self.cohort / 'quality/train' / uid / 'metadata.json')}
        config = cohort_plan(self.source, self.fixture.dataset_root, 2)
        write_json(self.cohort / 'config.json', config)
        metadata = read_json(self.cohort / 'quality/train/a1/metadata.json')
        write_json(self.cohort / 'summary.json', {'status': 'complete', 'training_ready': True,
            'completed': 4, 'records': receipts, 'schema_sha256': metadata['schema_sha256']})
        self.experiment = self.root / 'experiment'

    def prepare(self):
        return prepare_experiment(self.cohort, self.experiment, epochs=1, num_folds=2)

    def test_plan_is_development_only_and_requires_exact_source_folds(self):
        plan = cohort_plan(self.source, self.fixture.dataset_root, 2)
        self.assertEqual(plan['development_uids'], self.uids)
        self.assertFalse(plan['official_test_used'])
        self.assertNotIn('t1', plan['development_uids'])
        config = read_json(self.source / 'config.json')
        config['split_sources'][0]['sha256'] = 'changed'
        write_json(self.source / 'config.json', config)
        with self.assertRaisesRegex(ValueError, 'folds differ'):
            cohort_plan(self.source, self.fixture.dataset_root, 2)

    def test_gate_construction_preserves_common_initialization_and_rng(self):
        models, random_values = [], []
        for method in DECOMPOSED_METHODS:
            torch.manual_seed(31)
            module = DecomposedWordFusion(6, 8, 2, fusion_method=method,
                quality_dim=14 if method == 'reliability_gated' else 0)
            models.append({n: v for n, v in module.state_dict().items() if not n.startswith('gate.')})
            random_values.append(torch.rand(10))
        for state in models[1:]:
            for key, value in models[0].items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        for value in random_values[1:]:
            torch.testing.assert_close(value, random_values[0], rtol=0, atol=0)

    def test_prepare_freezes_inputs_modes_and_identical_common_model_weights(self):
        manifest = self.prepare()
        self.assertEqual(manifest['variants'], list(DECOMPOSED_METHODS))
        self.assertEqual(len(manifest['folds']), 2)
        self.assertEqual(len({r['common_state_sha256'] for r in manifest['initialization'].values()}), 1)
        for method in DECOMPOSED_METHODS:
            command = variant_command(self.experiment, manifest, method)
            self.assertIn('--no-use_weights_and_biases', command)
            self.assertIn('--cross_validate', command)
            self.assertEqual(command[command.index('--fusion_method') + 1], method)
        self.assertFalse(summarize(self.experiment)['complete'])
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            summarize(self.experiment, require_complete=True)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_changes_to_frozen_code_packages_or_splits_are_rejected(self):
        manifest = self.prepare()
        paths = [self.experiment / next(iter(manifest['source_sha256'])),
                 Path(next(iter(manifest['split_sha256']))),
                 self.cohort / 'quality/train/a1/metadata.json']
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            try:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    verify_inputs(self.experiment, manifest)
            finally:
                path.write_bytes(original)

    def test_incomplete_cohort_cannot_start_comparison(self):
        path = self.cohort / 'summary.json'
        state = read_json(path)
        state['training_ready'] = False
        write_json(path, state)
        with self.assertRaisesRegex(ValueError, 'verified development'):
            self.prepare()
        self.assertFalse(self.experiment.exists())

    def test_submission_limits_concurrency_and_prevents_duplicate_launches(self):
        self.prepare()
        with patch('utils.compare_fusion_modes.subprocess.run', side_effect=[
                subprocess.CompletedProcess([], 0, '12345\n', ''),
                subprocess.CompletedProcess([], 0, '12346\n', '')]) as scheduler:
            result = submit(self.experiment)
        self.assertEqual(result['array_job_id'], '12345')
        self.assertEqual(result['summary_job_id'], '12346')
        commands = [call.args[0] for call in scheduler.call_args_list]
        self.assertIn('--array=0-2%2', commands[0])
        self.assertIn('--dependency=afterok:12345', commands[1])
        with self.assertRaises(FileExistsError):
            submit(self.experiment)

    def test_cohort_packaging_resumes_without_overwriting_and_checks_receipts(self):
        plan = cohort_plan(self.source, self.fixture.dataset_root, 2)
        destination = self.root / 'resumable'

        def fake_export(feature_root, uid, target):
            original = self.parent.root / 'frame_features' / uid
            shutil.copytree(original, target)
            path = target / 'metadata.json'
            metadata = read_json(path)
            metadata['provenance'].update(feature_root=plan['feature_root'], recording_key='train/' + uid)
            metadata['provenance']['verified_sha256'].update(plan['source_sha256'])
            write_json(path, metadata)

        with patch('utils.prepare_fusion_cohort.package_fusion_frames', side_effect=fake_export) as exporter:
            result = prepare_cohort(self.source, self.fixture.dataset_root, destination, 2)
            self.assertTrue(result['training_ready'])
            self.assertEqual(exporter.call_count, 4)
        with patch('utils.prepare_fusion_cohort.package_fusion_frames', side_effect=AssertionError('No overwrite')):
            self.assertTrue(prepare_cohort(self.source, self.fixture.dataset_root, destination, 2)['training_ready'])
        path = destination / 'quality/train/a1/metadata.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Completed quality package changed'):
            prepare_cohort(self.source, self.fixture.dataset_root, destination, 2)

    def test_real_three_mode_cv_and_audits_reject_mismatched_normalization(self):
        self.prepare()
        for method in DECOMPOSED_METHODS:
            run_variant(self.experiment, method)
        result = summarize(self.experiment, require_complete=True)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['paired_fold_differences']), 3)
        self.assertTrue((self.experiment / 'comparison.csv').exists())
        folds = self.experiment / 'reliability_gated/results'
        fold = read_json(folds / 'fold_0.json')
        checkpoint_path = Path(fold['checkpoint'])
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint['settings'].frame_quality_normalization['fit_uids'] = ['a1', 'c1']
        torch.save(checkpoint, checkpoint_path)
        with self.assertRaisesRegex(ValueError, 'normalization has different training'):
            summarize(self.experiment, require_complete=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
