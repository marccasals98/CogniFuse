"""Unsupported-word opt-in preserves IDs and reaches the existing training masks."""

import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from utils.audio_frame_timing import frame_timing
from utils.batch_word_embeddings import audit_record
from utils.compare_word_alignment import inspect_word_alignment
from utils.ctc_forced_alignment import align_ctc_units, export_alignment
from utils.ctc_skipped_words import normalize_skipping_unsupported, skip_metadata
from utils.ctc_word_windows import export_word_windows, merge_word_windows
from utils.package_word_embeddings import AUDIO_SUFFIX, TEXT_SUFFIX, package_word_embeddings
from utils.prepro_ctc import source_identity
from utils.prepro_word_text import export_text_words
from utils.transcript_normalization import normalize_ctc_words
from utils.word_audio_pool import export_audio_words


VOCAB = {'|': 0, 'a': 1, 'b': 2, 'ñ': 3, '<pad>': 4}


def fixture(words=None):
    words = ['12', 'aa', 'tri월illo', '...', 'ñ', '3'] if words is None else words
    transcript = normalize_skipping_unsupported(words, VOCAB, 4)
    expected = [4, 1, 1, 4, 1, 0, 3, 3, 4]
    logits = torch.full((1, len(expected), len(VOCAB)), -6.)
    for index, token in enumerate(expected):
        logits[0, index, token] = 6.
    config = SimpleNamespace(conv_kernel=[10, 3, 3, 3, 3, 2, 2], conv_stride=[5, 2, 2, 2, 2, 2, 2])
    timing = frame_timing(config, (len(expected) - 1) * 320 + 400, 16000, len(expected))
    metadata = {'format': 'ctc_emissions_v1', 'logits_shape': list(logits.shape),
                'ctc_model': 'synthetic', 'audio_source': {}, 'word_source': {},
                'blank_id': 4, 'vocabulary': VOCAB, 'timing': timing, **skip_metadata(transcript)}
    return logits.log_softmax(-1), torch.ones(1, len(expected), dtype=torch.bool), transcript, metadata


class SkipWordTests(unittest.TestCase):
    def test_strict_is_unchanged_and_skip_keeps_whole_word_ids(self):
        words = ['12', 'aa', 'tri월illo', '...', 'ñ', '3']
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            normalize_ctc_words(words, VOCAB, 4)
        result = normalize_skipping_unsupported(words, VOCAB, 4)
        self.assertEqual(result['words'], words)
        self.assertEqual(result['skipped_word_ids'], [0, 2, 5])
        self.assertEqual(result['ctc_text'], 'aa ñ')
        self.assertEqual(result['token_word_ids'], [1, 1, -1, 4])
        for i in (0, 2, 3, 5):
            row = result['word_mappings'][i]
            self.assertEqual(row['token_start'], row['token_end'])
            self.assertEqual(row['status'], 'no_ctc_units')
        self.assertEqual(normalize_skipping_unsupported(['aa', '...', 'ñ'], VOCAB, 4),
                         normalize_ctc_words(['aa', '...', 'ñ'], VOCAB, 4))

    def test_skip_does_not_hide_malformed_inputs(self):
        for words in (['two words'], [None]):
            with self.assertRaises(ValueError):
                normalize_skipping_unsupported(words, VOCAB, 4)
        with self.assertRaisesRegex(ValueError, 'delimiter'):
            normalize_skipping_unsupported(['12'], {'<pad>': 4}, 4)
        emissions, mask, _, metadata = fixture()
        transcript = normalize_skipping_unsupported(['12', '여'], VOCAB, 4)
        with self.assertRaisesRegex(ValueError, 'no CTC target'):
            align_ctc_units(emissions, mask, transcript, metadata)

    def test_alignment_and_windows_keep_null_skipped_rows(self):
        emissions, mask, transcript, metadata = fixture()
        original = copy.deepcopy(transcript)
        alignment, path, _ = align_ctc_units(emissions, mask, transcript, metadata)
        self.assertEqual(path.tolist(), [[4, 1, 1, 4, 1, 0, 3, 3, 4]])
        self.assertEqual([u['word_id'] for u in alignment['units']], [1, 1, -1, 4])
        windows = merge_word_windows(alignment)
        self.assertEqual(windows['words'], transcript['words'])
        self.assertEqual(windows['skipped_word_ids'], [0, 2, 5])
        self.assertEqual(windows['word_alignment_mask'], [False, True, False, False, True, False])
        for i in (0, 2, 5):
            self.assertIsNone(windows['word_windows'][i]['start_frame'])
            self.assertEqual(windows['word_windows'][i]['word_id'], i)
        self.assertEqual(transcript, original)

    def test_tampered_skips_and_missing_supported_units_rejected(self):
        emissions, mask, transcript, metadata = fixture()
        broken = copy.deepcopy(transcript)
        broken['skipped_word_ids'].append(1)
        with self.assertRaisesRegex(ValueError, 'skipped-word metadata'):
            align_ctc_units(emissions, mask, broken, metadata)
        alignment, _, _ = align_ctc_units(emissions, mask, transcript, metadata)
        alignment['units'].pop()
        with self.assertRaisesRegex(ValueError, 'targets'):
            merge_word_windows(alignment)

    def test_audit_skips_only_unsupported_characters_not_bad_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, words = root / 'audio.wav', root / 'words.csv'
            audio.write_bytes(b'fixture')
            words.write_text('word,start,end\naa,0,1\n12,1,2\n')
            record = {'audio': str(audio), 'words': str(words)}
            self.assertEqual(len(audit_record(record, VOCAB, 4)), 1)
            record['skip_unsupported_words'] = True
            self.assertEqual(audit_record(record, VOCAB, 4), [])
            words.write_text('word,start,end\naa,0,1\n12,3,2\n')
            self.assertIn('timestamps', audit_record(record, VOCAB, 4)[0]['error'])
            words.write_text('word,start,end\n12,0,1\n')
            self.assertIn('No alignable words', audit_record(record, VOCAB, 4)[0]['error'])

    def test_full_export_retains_text_context_and_masks_both_training_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_audio, source_words = root / 'recording.wav', root / 'words.csv'
            source_audio.write_bytes(b'audio identity fixture')
            emissions, mask, transcript, metadata = fixture()
            with source_words.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['word', 'start', 'end'])
                for i, word in enumerate(transcript['words']):
                    writer.writerow([word, i * .02, (i + 1) * .02])
            before = source_words.read_bytes()
            metadata.update(audio_source=source_identity(source_audio), word_source=source_identity(source_words))
            ctc = root / 'ctc'
            ctc.mkdir()
            (ctc / 'metadata.json').write_text(json.dumps(metadata))
            (ctc / 'transcript.json').write_text(json.dumps(transcript))
            torch.save(emissions, ctc / 'ctc_log_probs.pt')
            torch.save(mask, ctc / 'valid_frame_mask.pt')
            export_alignment(ctc, root / 'alignment')
            export_word_windows(root / 'alignment', root / 'windows')
            comparison = inspect_word_alignment(root / 'windows', root / 'comparison')
            self.assertEqual(comparison['skipped_word_ids'], [0, 2, 5])
            frames = root / 'frames'
            frames.mkdir()
            frame_meta = {'format': 'acoustic_frames_v1', 'source': metadata['audio_source'],
                          'audio_model': 'synthetic', 'augmentation': None, 'pooled': False,
                          'feature_shape': [9, 8], 'timing': metadata['timing']}
            (frames / 'metadata.json').write_text(json.dumps(frame_meta))
            torch.save(torch.randn(9, 8), frames / 'frames.pt')
            torch.save(mask[0], frames / 'valid_mask.pt')
            export_audio_words(frames, root / 'windows', root / 'audio_words')
            tokenizer_dir = root / 'tokenizer'
            tokenizer_dir.mkdir()
            vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'aa', 'ñ', '.', '12', '3']
            (tokenizer_dir / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
            tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True, model_max_length=32)
            model = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=8,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        intermediate_size=16, max_position_embeddings=32))
            export_text_words(root / 'audio_words', root / 'text_words', tokenizer, model,
                              'tiny-text', chunk_size=32)
            packaged = package_word_embeddings(root / 'audio_words', root / 'text_words', root / 'package')
            self.assertEqual(packaged['words'], transcript['words'])
            self.assertEqual(packaged['word_ids'], list(range(6)))
            self.assertEqual(packaged['skipped_word_ids'], [0, 2, 5])
            self.assertEqual(packaged['valid_paired_words'], 2)
            for suffix in (AUDIO_SUFFIX, TEXT_SUFFIX):
                saved_mask = torch.load(root / 'package' / ('recording' + suffix[:-3] + '_mask.pt'), weights_only=True)
                self.assertEqual(saved_mask.tolist(), [False, True, False, False, True, False])
            audio = torch.load(root / 'package/recording_word_audio.pt', weights_only=True)
            self.assertTrue((audio[[0, 2, 5]] == 0).all())
            text_meta = json.loads((root / 'text_words/metadata.json').read_text())
            self.assertEqual(text_meta['words'], transcript['words'])
            self.assertEqual(source_words.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
