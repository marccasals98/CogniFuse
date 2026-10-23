"""Batch orchestration tests: cohort integrity, resumability and immutable outputs."""

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils.batch_word_embeddings import (
    audit_record, cohort, completed_stage, digest, identity, publish_file, read_json,
    run_batch, validate_package, validated_audio_package, verify_training_cohort, write_new_json,
)
from utils.package_word_embeddings import package_word_embeddings


class BatchWordTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def packaged_fixture(self):
        """An old tokenizer masked one acoustically valid word in training."""
        audio, text = self.root / 'audio_words', self.root / 'old_text'
        audio.mkdir()
        text.mkdir()
        recording, transcript = self.root / 'a.mp3', self.root / 'a.csv'
        recording.write_bytes(b'recording')
        transcript.write_text('word,start,end\nhola,0,1\nhola,1,2\n...,2,2\nfin,2,3\n')
        words = ['hola', 'hola', '...', 'fin']
        am, tm = torch.tensor([True, True, False, True]), torch.tensor([True, False, True, True])
        windows = self.root / 'windows.json'
        write_new_json(windows, {'words': words, 'word_alignment_mask': am.tolist()})
        meta = {'format': 'audio_word_embeddings_v1', 'words': words,
                'audio_source': identity(recording), 'word_source': identity(transcript),
                'feature_shape': [4, 8], 'valid_word_count': 3,
                'audio_model': 'audio-model', 'ctc_model': 'ctc-model', 'context_frames': 0,
                'word_windows': [{'word_id': i, 'word': w, 'status': 'aligned' if am[i] else 'no_ctc_units'}
                                 for i, w in enumerate(words)],
                'inputs': {'word_windows': {'path': str(windows), 'sha256': digest(windows)}}}
        write_new_json(audio / 'metadata.json', meta)
        vectors = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        vectors[~am] = 0
        torch.save(vectors, audio / 'audio_word_embeddings.pt')
        torch.save(am, audio / 'word_mask.pt')
        vectors = torch.ones(4, 8)
        vectors[~tm] = 0
        torch.save(vectors, text / 'text_word_embeddings.pt')
        torch.save(tm, text / 'text_word_mask.pt')
        torch.save(am & tm, text / 'paired_word_mask.pt')
        write_new_json(text / 'metadata.json', {
            'format': 'text_word_embeddings_v1', 'words': words, 'chunk_size': 32,
            'word_mappings': [{'word_id': i, 'word': w, 'status': 'encoded' if tm[i] else 'no_text_tokens'}
                              for i, w in enumerate(words)],
            'text_model': 'old-model', 'model_revision': 'old-revision',
            'audio_source': meta['audio_source'], 'word_source': meta['word_source'],
            'feature_shape': [4, 8], 'audio_feature_shape': [4, 8],
            'valid_audio_words': 3, 'valid_text_words': 3, 'valid_paired_words': 2,
            'inputs': {p.name: {'path': str(p), 'sha256': digest(p)} for p in audio.iterdir()},
        })
        source = self.root / 'old_packages'
        package_word_embeddings(audio, text, source)
        record = {'uid': 'a', 'audio': str(recording), 'words': str(transcript)}
        config = {'text_model': 'new-model', 'text_revision': 'new-revision', 'audio_model': 'audio-model',
                  'ctc_model': 'ctc-model', 'context_frames': 0, 'chunk_size': 32}
        return source, record, config

    def test_audio_reuse_allows_new_text_encoder_but_rejects_incompatible_or_changed_sources(self):
        source, record, config = self.packaged_fixture()
        with self.assertRaisesRegex(ValueError, 'text model differs'):
            validate_package(source, record, config)
        audio, _, _ = validated_audio_package(source, record, config)
        self.assertEqual(audio, self.root / 'audio_words')
        for key, value in [('context_frames', 1), ('audio_model', 'different'), ('chunk_size', 16)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validated_audio_package(source, record, {**config, key: value})
        torch.save(torch.zeros(4, 8), audio / 'audio_word_embeddings.pt')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            validated_audio_package(source, record, config)

    def test_reencode_runs_only_text_and_package_preserves_audio_and_resumes(self):
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from utils.batch_word_embeddings import StageWorker
        from utils.prepro_word_text import export_text_words
        source, record, config = self.packaged_fixture()
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        vocab = self.root / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhola\n.\nfin\n')
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), model_max_length=32)
        model = BertModel(BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
                                    num_attention_heads=2, intermediate_size=16, max_position_embeddings=32))
        model.config._commit_hash = config['text_revision']
        labels = self.root / 'labels.csv'
        labels.write_text('filename,patient_id,diagnosis\na.mp3,p,svPPA\n')
        args = SimpleNamespace(output_root=self.root / 'new_batch', labels_csv=labels, reuse_packaged_dir=[],
                               reuse_audio_packaged_dir=source, device='cpu', local_files_only=True)
        stages = []
        original_run = StageWorker.run
        def run(worker, stage, record, previous, output):
            stages.append(stage)
            if stage == 'text_words':
                export_text_words(previous['audio_words'], output, tokenizer, model,
                                  config['text_model'], chunk_size=config['chunk_size'])
            elif stage == 'package':
                original_run(worker, stage, record, previous, output)
            else:
                self.fail(f'Acoustic inference must be skipped: {stage}')
        with patch.object(StageWorker, 'run', run), \
             patch('utils.compare_word_alignment.inspect_word_alignment', return_value={'summary': {}}), \
             patch('utils.batch_word_embeddings.verify_training_cohort', return_value=[]):
            self.assertTrue(run_batch(args, [record], config, {})['training_ready'])
            self.assertEqual(stages, ['text_words', 'package'])
            stages.clear()
            self.assertTrue(run_batch(args, [record], config, {})['training_ready'])
            self.assertEqual(stages, [])
        target = args.output_root / 'embeddings'
        result, _ = validate_package(target, record, config)
        self.assertEqual(result['words'], ['hola', 'hola', '...', 'fin'])
        self.assertEqual(result['paired_mask'], [True, True, False, True])
        old = torch.load(source / 'a_word_audio.pt', weights_only=True)
        new = torch.load(target / 'a_word_audio.pt', weights_only=True)
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))

    def test_missing_audio_reuse_fails_without_running_encoders(self):
        labels = self.root / 'labels.csv'
        labels.write_text('filename,patient_id,diagnosis\na.mp3,p,svPPA\n')
        args = SimpleNamespace(output_root=self.root / 'batch', labels_csv=labels, reuse_packaged_dir=[],
                               reuse_audio_packaged_dir=self.root / 'missing', device='cpu', local_files_only=True)
        with patch('utils.batch_word_embeddings.StageWorker') as worker:
            result = run_batch(args, [{'uid': 'a', 'audio': 'a.mp3', 'words': 'a.csv'}], {}, {})
            worker.return_value.run.assert_not_called()
        self.assertFalse(result['training_ready'])
        self.assertEqual(result['failures']['a']['stage'], 'reuse_audio_words')

    def test_cohort_class_filter_dedup_and_patient_conflicts(self):
        labels = self.root / 'labels.csv'
        labels.write_text(',NHC ID HSP,DX_Pilar\na.mp3,p1,svPPA\na.mp3,p1,svPPA\nb.mp3,p2,exclude\nc.mp3,p3,nfPPA\n')
        rows = cohort(labels, self.root, self.root)
        self.assertEqual([r['uid'] for r in rows], ['a', 'c'])
        with labels.open('a') as stream:
            stream.write('d.mp3,p1,lvPPA\n')
        with self.assertRaisesRegex(ValueError, 'Conflicting diagnoses'):
            cohort(labels, self.root, self.root)

    def test_uid_collision_rejected(self):
        labels = self.root / 'labels.csv'
        labels.write_text('filename,patient_id,diagnosis\na.mp3,p1,svPPA\na.wav,p2,svPPA\n')
        with self.assertRaisesRegex(ValueError, 'collision'):
            cohort(labels, self.root, self.root)

    def test_input_audit_reports_each_bad_word_without_modifying(self):
        audio, words = self.root / 'a.mp3', self.root / 'a.csv'
        audio.write_bytes(b'test')
        words.write_text('word,start,end\nhola,0,1\n12,1,2\n여기서,2,3\n')
        before = words.read_bytes()
        record = {'audio': str(audio), 'words': str(words)}
        issues = audit_record(record, {'<pad>': 0, '|': 1, 'h': 2, 'o': 3, 'l': 4, 'a': 5}, 0)
        self.assertEqual([p['word_id'] for p in issues], [1, 2])
        self.assertEqual(words.read_bytes(), before)

    def test_interrupted_attempt_preserved_and_success_reused(self):
        def fail(output):
            output.mkdir()
            (output / 'partial.pt').write_bytes(b'partial')
            raise RuntimeError('interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            completed_stage(self.root, 'frames', {'input': 'first'}, fail)
        partial = list(self.root.glob('frames/*/partial.pt'))[0]
        def succeed(output):
            output.mkdir()
            (output / 'frames.pt').write_bytes(b'complete')
        first = completed_stage(self.root, 'frames', {'input': 'first'}, succeed)
        again = completed_stage(self.root, 'frames', {'input': 'first'}, lambda _: self.fail('Must reuse'))
        self.assertEqual(first, again)
        self.assertEqual(partial.read_bytes(), b'partial')
        changed = completed_stage(self.root, 'frames', {'input': 'changed'}, succeed)
        self.assertNotEqual(first, changed)
        self.assertTrue(first.exists())

    def test_modified_completed_stage_fails_instead_of_reusing(self):
        def action(output):
            output.mkdir()
            (output / 'file').write_text('original')
        result = completed_stage(self.root, 'ctc', {}, action)
        (result / 'file').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'modified'):
            completed_stage(self.root, 'ctc', {}, action)

    def test_publish_resume_and_collision_preserve_existing_files(self):
        source, target = self.root / 'source', self.root / 'collected/file'
        source.write_bytes(b'original')
        publish_file(source, target)
        publish_file(source, target)
        self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
        source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            publish_file(source, target)
        self.assertEqual(target.read_bytes(), b'original')

    def test_failed_record_is_reported_and_never_marks_cohort_ready(self):
        labels = self.root / 'labels.csv'
        labels.write_text('filename,patient_id,diagnosis\na.mp3,p,svPPA\n')
        args = SimpleNamespace(output_root=self.root / 'batch', labels_csv=labels,
                               reuse_packaged_dir=[], device='cpu', local_files_only=True)
        records = [{'uid': 'a', 'audio': 'a.mp3', 'words': 'a.csv'}]
        issues = {'a': [{'word_id': 0, 'word': '12', 'error': 'unsupported'}]}
        with patch('utils.batch_word_embeddings.StageWorker') as worker, \
             patch('utils.batch_word_embeddings.verify_training_cohort') as verify:
            result = run_batch(args, records, {}, issues)
            worker.return_value.run.assert_not_called()
            verify.assert_not_called()
        self.assertFalse(result['training_ready'])
        self.assertEqual(result['total_recordings'], 1)
        self.assertEqual(result['packaged_recordings'], 0)
        self.assertIn('a', result['failures'])
        report = list((args.output_root / 'reports').glob('*_input_issues.csv'))[0]
        with report.open() as stream:
            self.assertEqual(list(csv.DictReader(stream))[0]['word'], '12')

    def test_real_loader_readiness_checks_all_folds_and_missing_files(self):
        labels, embeddings = self.root / 'labels.csv', self.root / 'embeddings'
        embeddings.mkdir()
        records = []
        with labels.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['filename', 'patient_id', 'diagnosis'])
            for label in ('lvPPA', 'nfPPA', 'svPPA'):
                for index in range(5):
                    uid = f'{label}_{index}'
                    records.append({'uid': uid})
                    writer.writerow([uid + '.mp3', uid, label])
                    for suffix in ('_word_audio', '_word_text'):
                        torch.save(torch.ones(index + 1, 8), embeddings / (uid + suffix + '.pt'))
                        torch.save(torch.ones(index + 1, dtype=torch.bool), embeddings / (uid + suffix + '_mask.pt'))
        folds = verify_training_cohort(records, labels, embeddings)
        self.assertEqual(len(folds), 5)
        self.assertTrue(all(f['train_recordings'] == 12 and f['validation_recordings'] == 3 for f in folds))
        with self.assertRaisesRegex(ValueError, 'cohort differs'):
            verify_training_cohort(records[:-1], labels, embeddings)
        (embeddings / 'lvPPA_0_word_text_mask.pt').unlink()
        with self.assertRaises(FileNotFoundError):
            verify_training_cohort(records, labels, embeddings)


if __name__ == '__main__':
    unittest.main()
