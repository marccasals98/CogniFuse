"""Dataset, model, trainer, and evaluation parity for frozen frame packages."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import test_package_fusion_frames as package_fixtures
from scripts.data import make_precomputed_dataset
from scripts.frame_inputs import collate_frame_items, frame_batch_to_device, frame_item
from scripts.settings import TRAIN_DEFAULT_SETTINGS
from utils.package_fusion_frames import schema_fingerprint

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from model import Classifier
from train import Trainer
from scripts.evaluate import FrameEvaluationDataset, evaluate, parse_args
from scripts.cross_validation import plan_folds


class FrameInputTests(unittest.TestCase):
    def setUp(self):
        self.fixture = package_fixtures.FusionFramePackageTests(
            'test_cross_grid_mapping_invalid_words_and_exact_reconstruction')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.export()
        self.root = self.fixture.root
        self.features = self.root / 'frame_features'
        self.dataset_root = self.root / 'diagnosis'
        for uid, label in [('a1', 'ad'), ('a2', 'ad'), ('c1', 'cn'), ('c2', 'cn'), ('t1', None)]:
            path = self.dataset_root / ('train/audio/' + label if label else 'test-dist/audio') / (uid + '.wav')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            directory = self.features / uid
            shutil.copytree(self.fixture.output, directory)
            metadata = json.loads((directory / 'metadata.json').read_text())
            metadata.update(uid=uid, partition='train' if label else 'test')
            (directory / 'metadata.json').write_text(json.dumps(metadata))
        splits = self.dataset_root / 'train/splits'
        splits.mkdir()
        for fold, (train, val) in enumerate([(['a2', 'c2'], ['a1', 'c1']), (['a1', 'c1'], ['a2', 'c2'])]):
            for split, uids in [('train', train), ('val', val)]:
                np.save(splits / f'{split}_uids{fold}.npy', np.array(uids))
        schema = metadata['schema']
        self.params = SimpleNamespace(**{**TRAIN_DEFAULT_SETTINGS,
            'dataset': 'adresso', 'dataset_root': str(self.dataset_root), 'number_classes': 2,
            'num_folds': 2, 'fold': 0, 'load_checkpoint': False,
            'precomputed_features_dir': str(self.features), 'precomputed_input_type': 'frames',
            'frame_package_schema': schema, 'frame_package_schema_sha256': schema_fingerprint(schema),
            'speech_feature_extractor_output_vectors_dimension': 2,
            'text_feature_extractor_output_vectors_dimension': 3,
            'speech_adapter': 'LinearAdapter', 'speech_adapter_output_vectors_dimension': 4,
            'text_adapter': 'LinearAdapter', 'text_adapter_output_vectors_dimension': 4,
            'seq_to_seq_method': 'MultiHeadAttention', 'seq_to_seq_heads_number': 2,
            'seq_to_one_method': 'AttentionPooling', 'seq_to_seq_input_dropout': 0,
            'seq_to_one_input_dropout': 0, 'classifier_layer_drop_out': 0,
            'classifier_hidden_layers': 0, 'classifier_hidden_layers_width': 8})

    def test_saved_folds_labels_and_missing_packages(self):
        dataset = make_precomputed_dataset(self.params, split='train', fold=0)
        self.assertEqual([r['uid'] for r in dataset.segments], ['a2', 'c2'])
        self.assertEqual(dataset.feature_shapes, (torch.Size([10, 2]), torch.Size([3, 3])))
        self.assertEqual(dataset.class_counts, {0: 1, 1: 1})
        self.assertEqual(dataset[0]['label'].item(), 1)
        self.assertEqual(plan_folds(self.params)[0]['validation_uids'], ['a1', 'c1'])
        with self.assertRaisesRegex(ValueError, 'explicit package directory'):
            make_precomputed_dataset(self.params, split='test')
        test_params = copy.deepcopy(self.params)
        test_params.precomputed_test_features_dir = str(self.features)
        test = make_precomputed_dataset(test_params, split='test')
        self.assertEqual(test[0]['label'].item(), -1)
        shutil.rmtree(self.features / 'a2')
        with self.assertRaises(FileNotFoundError):
            make_precomputed_dataset(self.params, split='train', fold=0)

    def test_padded_batch_logits_parity_gradients_and_reload(self):
        item = frame_item(self.features / 'a1', 1)
        short = copy.deepcopy(item)
        short['frames'], short['frame_mask'] = short['frames'][:5], short['frame_mask'][:5]
        for key in ('word_ids', 'word_frame_intervals', 'audio_word_mask', 'text_words',
                    'text_word_mask', 'paired_word_mask', 'word_quality', 'word_quality_observed'):
            short[key] = short[key][:1]
        batch = collate_frame_items([item, short])
        self.assertEqual(tuple(batch['frames'].shape), (2, 10, 2))
        self.assertEqual(batch['word_ids'][1].tolist(), [0, -1, -1])
        self.assertEqual(batch['word_frame_intervals'][1, 1:].tolist(), [[-1, -1], [-1, -1]])
        self.assertEqual(tuple(batch['word_quality'].shape), (2, 3, 0))
        model = Classifier(self.params, 'cpu').eval()
        old_params = copy.deepcopy(self.params)
        old_params.precomputed_input_type = 'words'
        old = Classifier(old_params, 'cpu').eval()
        old.load_state_dict(model.state_dict(), strict=True)
        pooled = model.acoustic_word_pooling(batch['frames'], batch['word_frame_intervals'],
                                            batch['audio_word_mask'], batch['frame_mask'])
        masks = torch.stack([batch['paired_word_mask']] * 2, dim=1)
        logits = model(frame_batch=frame_batch_to_device(batch, 'cpu'))
        torch.testing.assert_close(logits, old(pooled, batch['text_words'], masks), rtol=0, atol=0)
        single = collate_frame_items([item])
        torch.testing.assert_close(logits[:1], model(frame_batch=frame_batch_to_device(single, 'cpu')),
                                   rtol=1e-5, atol=1e-6)
        changed = copy.deepcopy(batch)
        changed['frames'][~changed['frame_mask']] = 10000
        changed['text_words'][~changed['paired_word_mask']] = -10000
        torch.testing.assert_close(logits, model(frame_batch=frame_batch_to_device(changed, 'cpu')), rtol=0, atol=0)
        trainer = Trainer.__new__(Trainer)
        trainer.params, trainer.net, trainer.device = self.params, model, 'cpu'
        forwarded, labels = trainer.forward_batch(batch)
        torch.testing.assert_close(forwarded, logits, rtol=0, atol=0)
        torch.nn.functional.cross_entropy(forwarded, labels).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        self.assertIsNone(batch['frames'].grad)
        self.assertFalse(any('acoustic_word_pooling' in k for k in model.state_dict()))
        checkpoint = self.root / 'frame.chkpt'
        torch.save({'settings': self.params, 'model': model.state_dict()}, checkpoint)
        saved = torch.load(checkpoint, weights_only=False)
        loaded = Classifier(saved['settings'], 'cpu').eval()
        loaded.load_state_dict(saved['model'], strict=True)
        torch.testing.assert_close(logits, loaded(frame_batch=frame_batch_to_device(batch, 'cpu')), rtol=0, atol=0)

    def test_ppa_reuses_patient_disjoint_folds(self):
        rows = ['filename,patient_id,diagnosis']
        for label in ('lvPPA', 'nfPPA', 'svPPA'):
            for patient in range(2):
                for visit in range(2):
                    uid = f'{label}_{patient}_{visit}'
                    directory = self.features / uid
                    shutil.copytree(self.fixture.output, directory)
                    metadata = json.loads((directory / 'metadata.json').read_text())
                    metadata.update(uid=uid, partition='train')
                    (directory / 'metadata.json').write_text(json.dumps(metadata))
                    rows.append(f'{uid}.wav,{label}_{patient},{label}')
        labels = self.root / 'labels.csv'
        labels.write_text('\n'.join(rows) + '\n')
        params = copy.deepcopy(self.params)
        params.dataset, params.number_classes = 'ppa', 3
        params.train_labels_path = params.validation_labels_path = str(labels)
        train = make_precomputed_dataset(params, split='train', fold=0)
        val = make_precomputed_dataset(params, split='val', fold=0)
        self.assertFalse(set(train.df.patient_id) & set(val.df.patient_id))
        self.assertEqual(len(train) + len(val), 12)
        self.assertEqual(set(train.label_map), {'lvPPA', 'nfPPA', 'svPPA'})
        self.assertEqual(train[0]['schema_sha256'], params.frame_package_schema_sha256)

    def test_schema_partition_and_invalid_frame_ranges_are_rejected(self):
        item = frame_item(self.features / 'a1', 1)
        wrong = copy.deepcopy(item)
        wrong['schema_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'common package schema'):
            collate_frame_items([item, wrong])
        model = Classifier(self.params, 'cpu')
        batch = collate_frame_items([item])
        batch['schema_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'checkpoint package schema'):
            model(frame_batch=batch)
        batch['schema_sha256'] = item['schema_sha256']
        batch['frame_mask'][0, 1] = False
        with self.assertRaisesRegex(ValueError, 'includes invalid'):
            model(frame_batch=batch)
        path = self.features / 'a2/metadata.json'
        metadata = json.loads(path.read_text())
        metadata['partition'] = 'test'
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'partition differs'):
            make_precomputed_dataset(self.params, split='train', fold=0)
        metadata['partition'] = 'train'
        metadata['schema']['text_encoding']['tokenizer_sha256'] = 'another-tokenizer'
        metadata['schema_sha256'] = schema_fingerprint(metadata['schema'])
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'checkpoint schema'):
            make_precomputed_dataset(self.params, split='train', fold=0)

    def test_standalone_evaluation_uses_frame_checkpoint_schema(self):
        model = Classifier(self.params, 'cpu').eval()
        checkpoint = self.root / 'frame.chkpt'
        torch.save({'settings': self.params, 'model': model.state_dict()}, checkpoint)
        args = parse_args(['--checkpoint', str(checkpoint), '--dataset_to_evaluate', str(self.features / 'a1'),
                           '--output_dir', str(self.root / 'evaluation'), '--device', 'cpu', '--threads', '1',
                           '--batch_size', '2'])
        evaluate(args)
        manifest = json.loads((self.root / 'evaluation/manifest.json').read_text())
        self.assertEqual(manifest['recording_ids'], ['a1'])
        self.assertEqual(len(manifest['input_files']), 13)
        wrong = copy.deepcopy(self.params)
        wrong.frame_package_schema_sha256 = 'wrong'
        with self.assertRaisesRegex(ValueError, 'checkpoint schema'):
            FrameEvaluationDataset(self.features / 'a1', wrong)
        incomplete = self.root / 'incomplete-evaluation'
        (incomplete / 'unfinished').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'Incomplete frame package'):
            FrameEvaluationDataset(incomplete, self.params)

    def run_training_epoch(self):
        command = [sys.executable, str(SCRIPTS / 'train.py'), '--dataset', 'adresso',
                   '--dataset_root', str(self.dataset_root), '--number_classes', '2', '--num_folds', '2',
                   '--fold', '0', '--precomputed_features_dir', str(self.features),
                   '--precomputed_input_type', 'frames', '--max_epochs', '1',
                   '--acoustic_word_pooling', getattr(self.params, 'acoustic_word_pooling', 'mean'),
                   '--acoustic_word_attention_hidden_dim', str(getattr(
                       self.params, 'acoustic_word_attention_hidden_dim', 64)),
                   '--training_batch_size', '2', '--evaluation_batch_size', '2', '--num_workers', '0',
                   '--speech_adapter', 'LinearAdapter', '--speech_adapter_output_vectors_dimension', '4',
                   '--text_adapter', 'LinearAdapter', '--text_adapter_output_vectors_dimension', '4',
                   '--seq_to_seq_method', 'NoneSeqToSeq', '--classifier_hidden_layers', '0',
                   '--classifier_hidden_layers_width', '8', '--no-use_weights_and_biases',
                   '--early_stopping', '0', '--update_optimizer_every', '0',
                   '--eval_and_save_best_model_every', '0',
                   '--fold_results_path', str(self.root / 'fold.json'),
                   '--model_output_folder', str(self.root / 'models'),
                   '--log_file_folder', str(self.root / 'logs')]
        result = subprocess.run(command, cwd=SCRIPTS.parent, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])
        paths = list((self.root / 'models').rglob('*.chkpt'))
        self.assertEqual(len(paths), 1)
        saved = torch.load(paths[0], map_location='cpu', weights_only=False)
        self.assertEqual(saved['settings'].precomputed_input_type, 'frames')
        self.assertEqual(saved['settings'].frame_package_schema_sha256, self.params.frame_package_schema_sha256)
        self.assertEqual(saved['settings'].acoustic_word_pooling,
                         getattr(self.params, 'acoustic_word_pooling', 'mean'))
        return paths[0], saved

    def test_real_training_entrypoint_one_epoch(self):
        self.run_training_epoch()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
