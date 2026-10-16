"""Full-cohort completeness, resumability, and immutable batch exports."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_adresso_data as fixture_module
from utils.batch_adresso_embeddings import run_batch, english_vocabulary, file_names
from utils.prepro_adresso_recording import identity, TEXT_MODEL, AUDIO_MODEL


class BatchADReSSoTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ADReSSoDataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.args = SimpleNamespace(dataset_root=self.fixture.root,
            output_root=self.fixture.root / 'batch', pauses=True, chunk_size=8,
            repair_timestamps=False, num_folds=2, local_files_only=True,
            device='cpu', threads=1, reuse_recording_dir=[])

    def prepare(self, root, uid, split, pauses, index=None):
        records = index.training if split == 'train' else index.test
        record = next(r for r in records if r['uid'] == uid)
        return {'record': record, 'sources': {'audio': identity(record['audio_path'])}}

    def export(self, prepared, directory, *args):
        directory.mkdir(parents=True)
        r = prepared['record']
        names = file_names(r['uid'], True)
        values = [torch.ones(3, 768), torch.tensor([True, True, True]),
                  torch.cat((torch.ones(2, 768), torch.zeros(1, 768))), torch.tensor([True, True, False])]
        for name, value in zip(names, values):
            torch.save(value, directory / name)
        metadata = {'uid': r['uid'], 'partition': r['partition'], 'label': r['label'],
            'text_model': TEXT_MODEL, 'audio_model': AUDIO_MODEL,
            'chunk_size': 8, 'text_model_revision': 'fixture', 'audio_model_revision': 'fixture',
            'transcript_column': 'transcription_pause', 'truncated_tokens': 0,
            'sources': prepared['sources'], 'audio_shape': [3, 768], 'text_shape': [3, 768],
            'audio_valid_positions': 2, 'text_valid_positions': 3,
            'timestamp_repair': prepared.get('timestamp_repair')}
        (directory / names[-1]).write_text(json.dumps(metadata))
        return metadata

    def run_with_models(self, prepare=None):
        with patch('utils.batch_adresso_embeddings.AutoConfig.from_pretrained', return_value=SimpleNamespace(_commit_hash='fixture')), \
             patch('utils.batch_adresso_embeddings.AutoTokenizer.from_pretrained', return_value=SimpleNamespace(model_max_length=16, num_special_tokens_to_add=lambda pair=False: 2)), \
             patch('utils.batch_adresso_embeddings.AutoModel.from_pretrained') as model, \
             patch('utils.batch_adresso_embeddings.build_audio_encoder') as audio, \
             patch('utils.batch_adresso_embeddings.prepare_recording', side_effect=prepare or self.prepare), \
             patch('utils.batch_adresso_embeddings.export_recording', side_effect=self.export):
            state = run_batch(self.args)
            return state, model.call_count, audio.call_count

    def test_complete_cohort_is_verified_by_real_loader_and_resume_skips_encoders(self):
        state, models, audio = self.run_with_models()
        self.assertTrue(state['training_ready'])
        self.assertEqual(state['completed'], 6)
        self.assertEqual(state['readiness']['loader_checked_recordings'], 6)
        self.assertEqual((models, audio), (1, 1))
        before = {str(p): p.read_bytes() for p in self.args.output_root.glob('*/embeddings/*')}
        resumed, models, audio = self.run_with_models()
        self.assertTrue(resumed['training_ready'])
        self.assertEqual((models, audio), (0, 0))
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.args.output_root.glob('*/embeddings/*')})

    def test_failed_record_cannot_mark_partial_cohort_ready(self):
        def prepare(*args, **kwargs):
            if args[1] == 'a1':
                raise ValueError('Transcript mismatch')
            return self.prepare(*args, **kwargs)
        state, _, _ = self.run_with_models(prepare)
        self.assertFalse(state['training_ready'])
        self.assertEqual(state['completed'], 5)
        self.assertEqual(state['failed'], 1)
        self.assertNotIn('readiness', state)

    def test_changed_completed_export_is_rejected_without_overwrite(self):
        self.run_with_models()
        file = self.args.output_root / 'train/embeddings/a1distil_pauses.pt'
        torch.save(torch.zeros(3, 768), file)
        modified = file.read_bytes()
        state, _, _ = self.run_with_models()
        self.assertFalse(state['training_ready'])
        self.assertIn('hash changed', state['records']['train/a1']['error'])
        self.assertEqual(file.read_bytes(), modified)

    def test_interrupted_publication_reuses_validated_attempt(self):
        self.run_with_models()
        state_path = self.args.output_root / 'summary.json'
        state = json.loads(state_path.read_text())
        state['records']['train/a1'] = {'status': 'failed', 'error': 'interrupted publication'}
        state_path.write_text(json.dumps(state))
        missing = self.args.output_root / 'train/embeddings/a1distil_pauses_audio_mask.pt'
        missing.unlink()
        resumed, models, audio = self.run_with_models()
        self.assertTrue(resumed['training_ready'])
        self.assertTrue(missing.is_file())
        self.assertEqual((models, audio), (0, 0))

    def test_vocabulary_alias_preserves_trained_ids_and_rejects_collisions(self):
        self.assertEqual(english_vocabulary({'A': 7, 'B': 9, '|': 1, '<pad>': 0}),
                         {'a': 7, 'b': 9, '|': 1, '<pad>': 0})
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            english_vocabulary({'A': 7, 'a': 8})


if __name__ == '__main__':
    unittest.main()
