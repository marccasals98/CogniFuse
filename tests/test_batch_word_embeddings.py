"""Batch orchestration tests: cohort integrity, resumability and immutable outputs."""

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils.batch_word_embeddings import (
    audit_record, cohort, completed_stage, digest, publish_file, read_json,
    run_batch, verify_training_cohort, write_new_json,
)


class BatchWordTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

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
