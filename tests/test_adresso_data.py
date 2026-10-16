"""ADReSSo labels, saved partitions, and compatibility with the existing loader."""

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.adresso import ADReSSoIndex, UNLABELED
from scripts.data import PrecomputedADDataset, PrecomputedADReSSoDataset, make_precomputed_dataset
from scripts.cross_validation import plan_folds
from scripts.settings import TRAIN_DEFAULT_SETTINGS


class ADReSSoDataTests(unittest.TestCase):
    def setUp(self):
        # The home TMPDIR is on NFS, where retained .nfs entries can make
        # cleanup fail with ENOTEMPTY. Keep these fixtures on local storage.
        temporary = tempfile.TemporaryDirectory(prefix='cognifuse-adresso-', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.diagnosis = self.root / 'diagnosis'
        self.labels = {'c1': 'cn', 'c2': 'cn', 'a1': 'ad', 'a2': 'ad'}
        for uid, diagnosis in self.labels.items():
            path = self.diagnosis / 'train/audio' / diagnosis / (uid + '.wav')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            self.features(self.diagnosis / 'train/text' / diagnosis, uid)
        for uid in ('t1', 't2'):
            path = self.diagnosis / 'test-dist/audio' / (uid + '.wav')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            self.features(self.diagnosis / 'test-dist/text', uid)
        self.splits = self.diagnosis / 'train/splits'
        self.splits.mkdir(parents=True)
        self.save_split('train', 0, ['a2', 'c2'])
        self.save_split('val', 0, ['c1', 'a1'])
        self.save_split('train', 1, ['c1', 'a1'])
        self.save_split('val', 1, ['a2', 'c2'])
        self.params = SimpleNamespace(**{
            **TRAIN_DEFAULT_SETTINGS, 'dataset': 'adresso', 'dataset_root': str(self.root),
            'num_folds': 2, 'number_classes': 2, 'fold': 0,
            'precomputed_features_dir': str(self.diagnosis / 'train/text'),
            'precomputed_audio_suffix': '_audio.pt', 'precomputed_text_suffix': '_text.pt',
        })

    @staticmethod
    def features(directory, uid):
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in ('_audio', '_text'):
            torch.save(torch.arange(12, dtype=torch.float32).reshape(3, 4), directory / (uid + suffix + '.pt'))
            torch.save(torch.tensor([True, True, False]), directory / (uid + suffix + '_mask.pt'))

    def save_split(self, split, fold, uids):
        np.save(self.splits / f'{split}_uids{fold}.npy', np.array(uids))

    def test_root_resolution_and_explicit_saved_order(self):
        index = ADReSSoIndex(self.root)
        self.assertEqual(index.root, ADReSSoIndex(self.diagnosis).root)
        self.assertEqual([r['uid'] for r in index.select('train', 0, 2)], ['a2', 'c2'])
        self.assertEqual([r['uid'] for r in index.select('validation', 0, 2)], ['c1', 'a1'])
        self.assertEqual(index.summary(2)['training_classes'], {'ad': 2, 'cn': 2})
        with self.assertRaisesRegex(ValueError, 'explicit fold'):
            index.select('train', None, 2)

    def test_binary_labels_nested_features_masks_and_class_weights(self):
        train = make_precomputed_dataset(self.params, split='train', fold=0)
        validation = make_precomputed_dataset(self.params, split='val', fold=0)
        self.assertIsInstance(train, PrecomputedADReSSoDataset)
        self.assertIsInstance(train, PrecomputedADDataset)
        self.assertEqual(train.label_map, {'cn': 0, 'ad': 1})
        self.assertEqual([r['label'] for r in train.segments], [1, 0])
        self.assertEqual([r['label'] for r in validation.segments], [0, 1])
        self.assertEqual(train.get_classes_weights(), [2.0, 2.0])
        speech, targets, text, masks = train.collate_fn([train[0], train[1]])
        self.assertEqual(tuple(speech.shape), (2, 3, 4))
        self.assertEqual(tuple(text.shape), (2, 3, 4))
        self.assertEqual(targets.tolist(), [1, 0])
        self.assertEqual(tuple(masks.shape), (2, 2, 3))
        self.assertFalse(masks[:, :, -1].any())

    def test_flat_repaired_features_are_supported_without_moving_sources(self):
        flat = self.root / 'repaired'
        for uid in self.labels:
            self.features(flat, uid)
        dataset = make_precomputed_dataset(self.params, embeddings_dir=str(flat), split='train', fold=1)
        self.assertEqual([r['uid'] for r in dataset.segments], ['c1', 'a1'])
        self.assertTrue(all(Path(r['speech_path']).parent == flat for r in dataset.segments))

    def test_official_test_is_unlabeled_unfolded_and_uses_test_features(self):
        # Predictions are not ground truth, even if the template is populated.
        (self.diagnosis / 'test-dist/test_results_task1.csv').write_text('ID,Prediction\nt1,1\nt2,0\n')
        dataset = make_precomputed_dataset(self.params, split='test', fold=999)
        self.assertEqual([r['uid'] for r in dataset.segments], ['t1', 't2'])
        self.assertEqual([dataset[i][1].item() for i in range(len(dataset))], [UNLABELED, UNLABELED])
        self.assertTrue(all('/test-dist/text/' in r['speech_path'] for r in dataset.segments))
        self.assertEqual(dataset.class_counts, {0: 0, 1: 0})
        with self.assertRaisesRegex(ValueError, 'Unlabeled'):
            dataset.get_classes_weights()

    def test_cv_factory_reuses_saved_folds_and_keeps_test_out(self):
        plans = plan_folds(self.params)
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0]['training_uids'], ['a2', 'c2'])
        self.assertEqual(plans[0]['validation_uids'], ['a1', 'c1'])
        self.assertEqual(plans[0]['group_id_source'], 'recording_uid')
        self.assertEqual(set().union(*(set(p['validation_uids']) for p in plans)), set(self.labels))

    def test_duplicate_overlap_unknown_and_test_contamination_are_rejected(self):
        cases = [(['a2', 'a2'], 'Duplicate'), (['a2', 'c1'], 'overlap'),
                 (['a2', 'unknown'], 'complete training cohort'),
                 (['a2', 't1'], 'Official test UID')]
        for uids, message in cases:
            with self.subTest(uids=uids):
                self.save_split('train', 0, uids)
                with self.assertRaisesRegex(ValueError, message):
                    ADReSSoIndex(self.root).development_folds(2)

    def test_missing_or_inconsistent_fold_coverage_is_rejected(self):
        self.save_split('train', 1, ['a2', 'c2'])
        self.save_split('val', 1, ['c1', 'a1'])
        with self.assertRaisesRegex(ValueError, 'multiple saved folds'):
            ADReSSoIndex(self.root).development_folds(2)
        with self.assertRaisesRegex(ValueError, 'complete saved'):
            ADReSSoIndex(self.root).development_folds(3)

    def test_duplicate_recording_uids_across_classes_or_partitions_are_rejected(self):
        duplicate = self.diagnosis / 'train/audio/ad/c1.wav'
        duplicate.touch()
        with self.assertRaisesRegex(ValueError, 'Duplicate ADReSSo'):
            ADReSSoIndex(self.root)
        duplicate.unlink()
        (self.diagnosis / 'test-dist/audio/c1.wav').touch()
        with self.assertRaisesRegex(ValueError, 'train/test recording UIDs overlap'):
            ADReSSoIndex(self.root)

    def test_missing_inputs_fail_without_silently_shrinking_the_cohort(self):
        mask = self.diagnosis / 'train/text/ad/a2_audio_mask.pt'
        mask.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing precomputed mask'):
            make_precomputed_dataset(self.params, split='train', fold=0)
        torch.save(torch.ones(3, dtype=torch.bool), mask)
        (self.diagnosis / 'train/text/ad/a2_audio.pt').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing precomputed feature'):
            make_precomputed_dataset(self.params, split='train', fold=0)

    def test_ambiguous_feature_paths_are_rejected(self):
        self.features(self.diagnosis / 'train/text', 'a2')
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            make_precomputed_dataset(self.params, split='train', fold=0)

    def test_layer_tensors_still_require_explicit_aggregation(self):
        torch.save(torch.zeros(3, 12, 4), self.diagnosis / 'train/text/ad/a2_audio.pt')
        with self.assertRaisesRegex(ValueError, 'tokens, channels'):
            make_precomputed_dataset(self.params, split='train', fold=0)

    def test_ppa_default_factory_preserves_existing_label_map(self):
        labels = self.root / 'ppa.csv'
        labels.write_text('filename,patient_id,diagnosis\np.wav,p,svPPA\n')
        flat = self.root / 'ppa_features'
        self.features(flat, 'p')
        params = SimpleNamespace(**{**TRAIN_DEFAULT_SETTINGS, 'train_labels_path': str(labels),
            'validation_labels_path': str(labels), 'precomputed_features_dir': str(flat),
            'precomputed_audio_suffix': '_audio.pt', 'precomputed_text_suffix': '_text.pt'})
        dataset = make_precomputed_dataset(params, split='train', fold=1)
        self.assertIs(type(dataset), PrecomputedADDataset)
        self.assertEqual(dataset.label_map, {'lvPPA': 0, 'nfPPA': 1, 'svPPA': 2})
        self.assertEqual(dataset[0][1].item(), 2)

    def test_cli_defaults_and_invalid_adresso_configuration(self):
        # train.py uses the historical top-level scripts/utils.py import.
        # Exercise its entry-point environment in a child interpreter so it
        # does not shadow the repository's utils package in the test suite.
        code = textwrap.dedent('''
            from contextlib import redirect_stderr
            import io
            import sys
            sys.path.insert(0, 'scripts')
            from train import ArgsParser, Trainer
            root = sys.argv[1]
            parser = ArgsParser()
            parser.main([])
            assert (parser.arguments.dataset, parser.arguments.number_classes) == ('ppa', 3)
            parser = ArgsParser()
            parser.main(['--dataset', 'adresso', '--dataset_root', root])
            assert parser.arguments.number_classes == 2
            try:
                Trainer(parser.arguments)
            except ValueError as error:
                assert 'precomputed_features_dir' in str(error)
            else:
                raise AssertionError('ADReSSo raw training should fail before initialization')
            for argv in (['--dataset', 'adresso'],
                         ['--dataset', 'adresso', '--dataset_root', root, '--number_classes', '3'],
                         ['--dataset_dry_run']):
                with redirect_stderr(io.StringIO()):
                    try:
                        ArgsParser().main(argv)
                    except SystemExit as error:
                        assert error.code == 2
                    else:
                        raise AssertionError('Invalid CLI configuration accepted')
        ''')
        result = subprocess.run([sys.executable, '-c', code, str(self.root)],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
