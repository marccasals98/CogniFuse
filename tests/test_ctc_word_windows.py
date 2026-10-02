"""Word-window merging checks using small known character alignments."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from utils.audio_frame_timing import frame_interval_seconds, frame_timing
from utils.ctc_word_windows import INTERVAL_FIELDS, export_word_windows, merge_word_windows
from utils.transcript_normalization import normalize_ctc_words


def fixture(words=None):
    words = ['¡AA!', '...', 'ñ'] if words is None else words
    vocabulary = {'<pad>': 0, '|': 1, 'a': 2, 'ñ': 3}
    transcript = normalize_ctc_words(words, vocabulary, 0)
    count = len(transcript['target_ids']) * 2 + 1
    config = SimpleNamespace(conv_kernel=[10, 3, 3, 3, 3, 2, 2], conv_stride=[5, 2, 2, 2, 2, 2, 2])
    timing = frame_timing(config, (count - 1) * 320 + 400, 16000, count)
    units = []
    for index, (token_id, token, owner) in enumerate(zip(
            transcript['target_ids'], transcript['tokens'], transcript['token_word_ids'])):
        start, end = 2 * index + 1, 2 * index + 2
        units.append({'target_index': index, 'token_id': token_id, 'token': token,
                      'word_id': owner, 'start_frame': start, 'end_frame_exclusive': end,
                      'start_time': frame_interval_seconds(start, timing)[0],
                      'end_time': frame_interval_seconds(end - 1, timing)[1]})
    return {
        'format': 'ctc_unit_alignment_v1', 'ctc_model': 'synthetic',
        'audio_source': {}, 'word_source': {}, 'blank_id': 0,
        'words': words, 'word_mappings': transcript['word_mappings'], 'units': units,
        'aligned_frames': count, 'timing': timing,
        'frame_span_convention': 'start_inclusive_end_exclusive',
    }


class WordWindowTests(unittest.TestCase):
    def test_merge_spans_preserves_words_and_internal_blanks(self):
        source = fixture()
        original = copy.deepcopy(source)
        result = merge_word_windows(source)
        first, punctuation, last = result['word_windows']
        self.assertEqual(result['words'], ['¡AA!', '...', 'ñ'])
        self.assertEqual([row['word_id'] for row in result['word_windows']], [0, 1, 2])
        self.assertEqual((first['start_frame'], first['end_frame_exclusive']), (1, 4))
        self.assertEqual(first['ctc_unit_count'], 2)
        self.assertEqual((last['start_frame'], last['end_frame_exclusive']), (7, 8))
        self.assertEqual(result['word_alignment_mask'], [True, False, True])
        self.assertEqual(result['aligned_word_count'], 2)
        self.assertTrue(all(punctuation[field] is None for field in INTERVAL_FIELDS))
        for row in (first, last):
            for field in ('start_frame', 'end_frame_exclusive', 'start_time', 'end_time'):
                self.assertEqual(row[field], row['expanded_' + field])
        self.assertAlmostEqual(first['start_time'], 0.02)
        self.assertAlmostEqual(first['end_time'], 0.085)
        self.assertEqual(source, original)

    def test_context_clips_to_valid_frames_and_can_overlap(self):
        for context in (2, 4, 8, 100):
            with self.subTest(context=context):
                result = merge_word_windows(fixture(), context)
                rows = [row for row in result['word_windows'] if row['status'] == 'aligned']
                self.assertEqual(rows[0]['start_frame'], 1)
                self.assertEqual(rows[0]['expanded_start_frame'], 0)
                self.assertEqual(rows[-1]['expanded_end_frame_exclusive'], 9)
                self.assertGreater(rows[0]['expanded_end_frame_exclusive'], rows[-1]['expanded_start_frame'])
                self.assertLessEqual(rows[0]['expanded_start_frame'], rows[-1]['expanded_start_frame'])
                self.assertLessEqual(rows[0]['expanded_end_frame_exclusive'], rows[-1]['expanded_end_frame_exclusive'])
                for row in rows:
                    self.assertTrue(0 <= row['expanded_start_frame'] < row['expanded_end_frame_exclusive'] <= 9)
                    self.assertLessEqual(row['expanded_end_time'], result['timing']['audio_duration_seconds'])

    def test_repeated_words_remain_distinct(self):
        result = merge_word_windows(fixture(['a', 'a']))
        self.assertEqual([row['word'] for row in result['word_windows']], ['a', 'a'])
        self.assertEqual([row['start_frame'] for row in result['word_windows']], [1, 5])
        self.assertEqual([row['word_id'] for row in result['word_windows']], [0, 1])

    def test_punctuation_at_edges_and_single_word(self):
        result = merge_word_windows(fixture(['...', 'ñ', '!']))
        self.assertEqual(result['word_alignment_mask'], [False, True, False])
        self.assertEqual(result['word_windows'][1]['ctc_form'], 'ñ')
        self.assertEqual(len(result['word_windows']), 3)
        self.assertEqual(merge_word_windows(fixture(['ñ']))['aligned_word_count'], 1)

    def test_invalid_context_and_corrupted_spans_rejected(self):
        for context in (-1, 0.5, True):
            with self.subTest(context=context), self.assertRaises(ValueError):
                merge_word_windows(fixture(), context)
        for issue in ('ownership', 'order', 'padding', 'missing_character', 'time', 'mapping', 'timing'):
            with self.subTest(issue=issue):
                source = fixture()
                if issue == 'ownership':
                    source['units'][0]['word_id'] = 2
                elif issue == 'order':
                    source['units'][1]['start_frame'] = 0
                elif issue == 'padding':
                    source['units'][-1]['end_frame_exclusive'] = 10
                elif issue == 'missing_character':
                    source['units'].pop()
                elif issue == 'time':
                    source['units'][0]['end_time'] = 123.0
                elif issue == 'mapping':
                    source['word_mappings'][0]['word'] = 'changed'
                else:
                    source['timing']['valid_frame_count'] = 8
                with self.assertRaises(ValueError):
                    merge_word_windows(source)

    def test_export_preserves_source_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'step4', root / 'step5'
            source.mkdir()
            path = source / 'alignment.json'
            path.write_text(json.dumps(fixture()), encoding='utf-8')
            before = path.read_bytes()
            result = export_word_windows(source, output)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(result['alignment_source']['sha256'], hashlib.sha256(before).hexdigest())
            self.assertEqual(json.loads((output / 'word_windows.json').read_text()), result)
            with (output / 'word_windows.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['word'] for row in rows], ['¡AA!', '...', 'ñ'])
            self.assertEqual(rows[1]['start_frame'], '')
            saved = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                export_word_windows(source, output)
            self.assertEqual(saved, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == '__main__':
    unittest.main()
