"""Boundary-comparison checks with repeated words and known time differences."""

import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from utils.audio_frame_timing import frame_timing
from utils.compare_word_alignment import compare_word_windows, format_report, inspect_word_alignment
from utils.ctc_word_windows import export_word_windows
from utils.transcript_normalization import normalize_ctc_words


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv_path = self.root / 'whisper.csv'
        self.csv_path.write_text('word,start,end,probability\na,0,0.2,0.9\n...,0.2,0.4,\na,0.8,1.0,0.8\n')
        stat = self.csv_path.stat()
        self.whisper = list(csv.DictReader(self.csv_path.read_text().splitlines()))
        vocabulary = {'<pad>': 0, '|': 1, 'a': 2}
        transcript = normalize_ctc_words(['a', '...', 'a'], vocabulary, 0)
        timing = frame_timing(SimpleNamespace(conv_kernel=[1], conv_stride=[1]), 20, 10, 20)
        units = [
            {'target_index': 0, 'token_id': 2, 'token': 'a', 'word_id': 0,
             'start_frame': 1, 'end_frame_exclusive': 3, 'start_time': 0.1, 'end_time': 0.3},
            {'target_index': 1, 'token_id': 1, 'token': '|', 'word_id': -1,
             'start_frame': 3, 'end_frame_exclusive': 4, 'start_time': 0.3, 'end_time': 0.4},
            {'target_index': 2, 'token_id': 2, 'token': 'a', 'word_id': 2,
             'start_frame': 5, 'end_frame_exclusive': 7, 'start_time': 0.5, 'end_time': 0.7},
        ]
        alignment = {
            'format': 'ctc_unit_alignment_v1', 'ctc_model': 'test-model', 'blank_id': 0,
            'words': transcript['words'], 'word_mappings': transcript['word_mappings'], 'units': units,
            'aligned_frames': 20, 'timing': timing,
            'frame_span_convention': 'start_inclusive_end_exclusive',
            'audio_source': {'path': str(self.root / 'recording.wav')},
            'word_source': {'path': str(self.csv_path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
        }
        self.alignment_dir = self.root / 'step4'
        self.alignment_dir.mkdir()
        (self.alignment_dir / 'alignment.json').write_text(json.dumps(alignment))
        self.windows_dir = self.root / 'step5'
        self.windows = export_word_windows(self.alignment_dir, self.windows_dir, context_frames=2)

    def test_differences_flags_and_repeated_words(self):
        report = compare_word_windows(self.windows, self.whisper)
        first, punctuation, last = report['rows']
        self.assertEqual([row['word_id'] for row in report['rows']], [0, 1, 2])
        self.assertAlmostEqual(first['delta_start'], 0.1)
        self.assertAlmostEqual(first['delta_end'], 0.1)
        self.assertAlmostEqual(last['delta_start'], 0.3)
        self.assertAlmostEqual(last['signed_start_difference'], -0.3)
        self.assertFalse(first['review_flag'])
        self.assertTrue(last['review_flag'])
        self.assertIsNone(punctuation['ctc_start'])
        self.assertIsNone(punctuation['delta_start'])
        self.assertIsNone(punctuation['review_flag'])
        self.assertEqual(report['summary']['review_word_ids'], [2])
        self.assertAlmostEqual(report['summary']['mean_start_disagreement_seconds'], 0.2)
        self.assertEqual(report['summary']['compared_words'], 2)
        self.assertEqual(report['summary']['words_without_ctc'], 1)

    def test_context_is_not_used_for_comparison(self):
        self.assertEqual(self.windows['word_windows'][0]['expanded_start_time'], 0)
        report = compare_word_windows(self.windows, self.whisper)
        self.assertEqual(report['rows'][0]['ctc_start'], 0.1)
        self.assertEqual(report['ctc_boundaries'], 'original_unexpanded')
        self.assertEqual(report['context_frames_in_source'], 2)

    def test_threshold_and_float_tolerance(self):
        self.assertEqual(compare_word_windows(self.windows, self.whisper, 0.3)['summary']['review_flag_count'], 0)
        self.assertEqual(compare_word_windows(self.windows, self.whisper, 0)['summary']['review_flag_count'], 2)
        for threshold in (-0.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                compare_word_windows(self.windows, self.whisper, threshold)

    def test_wrong_words_counts_and_invalid_values(self):
        with self.assertRaises(ValueError):
            compare_word_windows(self.windows, self.whisper[:-1])
        for field, value in [('word', 'different'), ('start', 'nan'), ('end', '-1'), ('probability', '2')]:
            bad = copy.deepcopy(self.whisper)
            bad[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_word_windows(self.windows, bad)

    def test_all_unaligned_statistics_are_null(self):
        windows = copy.deepcopy(self.windows)
        windows['words'] = ['...']
        windows['word_windows'] = [windows['word_windows'][1]]
        windows['word_windows'][0]['word_id'] = 0
        windows['word_alignment_mask'] = [False]
        report = compare_word_windows(windows, [self.whisper[1]])
        self.assertEqual(report['summary']['compared_words'], 0)
        self.assertIsNone(report['summary']['mean_start_disagreement_seconds'])
        self.assertIn('no CTC units', format_report(report))

    def test_export_read_only_mode_and_no_overwrite(self):
        before = self.csv_path.read_bytes()
        original_windows = (self.windows_dir / 'word_windows.json').read_bytes()
        report = inspect_word_alignment(self.windows_dir)
        self.assertIn('[1,3)', format_report(report))
        output = self.root / 'step6'
        saved = inspect_word_alignment(self.windows_dir, output)
        self.assertEqual(json.loads((output / 'comparison.json').read_text()), saved)
        self.assertEqual((output / 'comparison.txt').read_text(), format_report(saved))
        with (output / 'comparison.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]['delta_start'], '')
        snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            inspect_word_alignment(self.windows_dir, output)
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in output.iterdir()})
        self.assertEqual(self.csv_path.read_bytes(), before)
        self.assertEqual((self.windows_dir / 'word_windows.json').read_bytes(), original_windows)

    def test_changed_whisper_source_rejected(self):
        self.csv_path.write_text(self.csv_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Whisper CSV changed'):
            inspect_word_alignment(self.windows_dir)

    def test_changed_step4_source_rejected(self):
        path = self.alignment_dir / 'alignment.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Step 4 alignment changed'):
            inspect_word_alignment(self.windows_dir)

    def test_modified_step5_windows_rejected(self):
        path = self.windows_dir / 'word_windows.json'
        value = json.loads(path.read_text())
        value['word_windows'][0]['start_time'] = 0.15
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Step 5 windows disagree'):
            inspect_word_alignment(self.windows_dir)


if __name__ == '__main__':
    unittest.main()
