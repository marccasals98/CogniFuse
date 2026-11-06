"""Verify strict ADReSSo correspondence and a preserved-source export."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from scripts.data import PrecomputedADDataset
from utils.prepro_adresso_recording import export_recording, identity, match_words


class ADReSSoPreprocessingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir='/tmp', prefix='adresso-prepro-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.text = 'Hello , world.'
        self.rows = [{'word': 'hello', 'start': '0.0', 'end': '0.1'},
                     {'word': 'world', 'start': '0.2', 'end': '0.4'}]
        self.words, audit = match_words(self.text, self.rows, 0.5)
        source = self.root / 'source.csv'
        source.write_text('Original source bytes must remain unchanged\n')
        self.prepared = {'record': {'uid': 'r1', 'partition': 'train', 'label': 1,
                                    'audio_path': str(self.root / 'audio.wav')},
                         'transcription': self.text, 'transcript_column': 'transcription',
                         'words': self.words, 'word_audit': audit,
                         'audio_info': {'duration_seconds': 0.5}, 'sources': {'words': identity(source)}}
        (self.root / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n,\n.\n')
        self.tokenizer = BertTokenizerFast(vocab=str(self.root / 'vocab.txt'), model_max_length=16)
        self.model = BertModel(BertConfig(vocab_size=9, hidden_size=4, num_hidden_layers=1,
                                        num_attention_heads=1, intermediate_size=8,
                                        max_position_embeddings=16)).eval()
        self.frames = torch.arange(100, dtype=torch.float32).reshape(25, 4)
        self.encoder = (None, SimpleNamespace(config=SimpleNamespace(_commit_hash='fixture')))

    def export(self, prepared=None):
        with patch('utils.prepro_adresso_recording.extract_audio_features', return_value=(self.frames, 50)):
            return export_recording(prepared or self.prepared, self.root / 'export', self.tokenizer,
                                    self.model, self.encoder, chunk_size=8)

    def test_case_only_matching_preserves_words_times_and_punctuation_gaps(self):
        self.assertEqual(self.words, [('Hello', 0.0, 0.1), ('world', 0.2, 0.4)])
        self.assertEqual(self.rows[0]['word'], 'hello')
        result = self.export()
        self.assertEqual(result['case_only_matches'], 1)
        comma = next(r for r in result['token_alignment'] if r['token'] == ',')
        self.assertEqual(comma['audio_interval'], (0.1, 0.2))
        self.assertEqual(result['loader_validation'], 'passed')

    def test_missing_timestamp_words_and_lexical_changes_are_rejected(self):
        for text in ('Hello yes world.', 'Goodbye world.', 'Hello world extra.'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                match_words(text, self.rows, 0.5)

    def test_malformed_and_nonmonotonic_timestamps_are_rejected(self):
        for start, end in (('nan', '.4'), ('-.1', '.4'), ('.4', '.2'), ('.2', '.9')):
            rows = [dict(self.rows[0]), {**self.rows[1], 'start': start, 'end': end}]
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                match_words(self.text, rows, .5)
        rows = [{**self.rows[0], 'start': '.1'}, {**self.rows[1], 'start': '.05'}]
        with self.assertRaisesRegex(ValueError, 'Nonmonotonic'):
            match_words(self.text, rows, .5)

    def test_export_masks_roundtrip_batching_and_no_overwrite(self):
        before = (self.root / 'source.csv').read_bytes()
        result = self.export()
        output = self.root / 'export'
        text = PrecomputedADDataset._load_features(output / 'r1distil.pt')
        audio = PrecomputedADDataset._load_features(output / 'r1distil_audio.pt')
        tm = PrecomputedADDataset._load_mask(output / 'r1distil_mask.pt', len(text))
        am = PrecomputedADDataset._load_mask(output / 'r1distil_audio_mask.pt', len(audio))
        self.assertTrue(tm.all())
        self.assertTrue(am[0])
        self.assertFalse(am[-1])
        speech, labels, text_batch, masks = PrecomputedADDataset.collate_fn([
            (audio, torch.tensor(1), text, am, tm)])
        self.assertEqual(tuple(masks.shape), (1, 2, len(text)))
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(result['truncated_tokens'], 0)
        saved = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(saved, {p.name: p.read_bytes() for p in output.iterdir()})
        self.assertEqual(before, (self.root / 'source.csv').read_bytes())

    def test_changed_source_fails_before_creating_output(self):
        (self.root / 'source.csv').write_text('Changed source')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.export()
        self.assertFalse((self.root / 'export').exists())

    def test_long_transcripts_keep_all_positions(self):
        text = ' '.join(['hello'] * 20)
        rows = [{'word': 'hello', 'start': str(i / 100), 'end': str((i + 1) / 100)} for i in range(20)]
        words, audit = match_words(text, rows, .5)
        prepared = {**self.prepared, 'transcription': text, 'words': words, 'word_audit': audit}
        result = self.export(prepared)
        self.assertEqual(result['text_shape'], [22, 4])
        self.assertGreater(result['chunks'], 1)
        self.assertEqual(result['content_tokens'], 20)


if __name__ == '__main__':
    unittest.main()
