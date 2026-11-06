"""Word-level text pooling tests with a real local fast tokenizer and tiny BERT."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from scripts.word_text_encoder import encode_word_text, word_chunks
from utils.prepro_word_text import export_text_words, read_audio_words


class WordTextTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'el', 'niño',
                 'gal', '##letas', '.', 'a', '##a', 'está']
        path = self.root / 'vocab.txt'
        path.write_text('\n'.join(vocab) + '\n', encoding='utf-8')
        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(self.root), local_files_only=True, do_lower_case=True,
            strip_accents=False, model_max_length=32,
        )
        self.model = BertModel(BertConfig(vocab_size=len(self.tokenizer), hidden_size=12,
                                         num_hidden_layers=1, num_attention_heads=3,
                                         intermediate_size=24, max_position_embeddings=32))
        self.words = ['galletas', '', 'niño', '...', 'galletas']

    def audio_fixture(self):
        directory = self.root / 'audio_words'
        directory.mkdir()
        mask = torch.tensor([True, False, True, False, True])
        vectors = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        vectors[~mask] = 0
        source = self.root / 'word_windows.json'
        source.write_text(json.dumps({'format': 'ctc_word_windows_v1', 'words': self.words,
                                      'word_alignment_mask': mask.tolist()}))
        metadata = {
            'format': 'audio_word_embeddings_v1', 'words': self.words,
            'feature_shape': [5, 3], 'valid_word_count': 3,
            'word_windows': [{'word_id': i, 'word': w, 'status': 'aligned' if mask[i] else 'no_ctc_units'}
                             for i, w in enumerate(self.words)],
            'audio_source': {}, 'word_source': {},
            'inputs': {'word_windows': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}},
        }
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        torch.save(vectors, directory / 'audio_word_embeddings.pt')
        torch.save(mask, directory / 'word_mask.pt')
        return directory

    def test_subword_means_match_direct_bert_and_preserve_words(self):
        vectors, mask, metadata = encode_word_text(self.words, self.tokenizer, self.model, chunk_size=32)
        batch = self.tokenizer(self.words, is_split_into_words=True, return_tensors='pt')
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state[0]
        self.assertEqual(metadata['word_mappings'][0]['subwords'], ['gal', '##letas'])
        self.assertEqual(metadata['words'], self.words)
        self.assertEqual(mask.tolist(), [True, False, True, True, True])
        for word_id in [0, 2, 3, 4]:
            positions = [i for i, owner in enumerate(batch.word_ids()) if owner == word_id]
            torch.testing.assert_close(vectors[word_id], hidden[positions].mean(0), rtol=0, atol=0)
        self.assertTrue((vectors[1] == 0).all())
        self.assertFalse(vectors.requires_grad)
        self.assertEqual([row['word_id'] for row in metadata['word_mappings']], list(range(5)))

    def test_long_transcript_chunks_never_split_or_drop_a_word(self):
        words = ['galletas'] * 20
        vectors, mask, metadata = encode_word_text(words, self.tokenizer, self.model, chunk_size=8)
        self.assertEqual(vectors.shape, (20, 12))
        self.assertTrue(mask.all())
        self.assertEqual(len(metadata['chunks']), 7)
        self.assertEqual(sum(row['subword_count'] for row in metadata['word_mappings']), 40)
        for row in metadata['word_mappings']:
            self.assertEqual(row['subwords'], ['gal', '##letas'])
            self.assertIsNotNone(row['chunk_index'])
        self.assertEqual(metadata['truncated_tokens'], 0)

    def test_special_tokens_are_excluded_but_original_special_spelling_is_kept(self):
        words = ['[MASK]', 'galletas']
        chunks, by_word = word_chunks(words, self.tokenizer, 8)
        self.assertEqual(by_word[0], [self.tokenizer.mask_token_id])
        self.assertIsNone(chunks[0]['word_ids'][0])
        self.assertIsNone(chunks[0]['word_ids'][-1])
        _, mask, result = encode_word_text(words, self.tokenizer, self.model, chunk_size=8)
        self.assertTrue(mask.all())
        self.assertEqual(result['word_mappings'][0]['subword_count'], 1)

    def test_empty_words_and_empty_transcript_keep_valid_shapes(self):
        for words in ([], ['', ' ']):
            vectors, mask, result = encode_word_text(words, self.tokenizer, self.model, chunk_size=8)
            self.assertEqual(vectors.shape, (len(words), 12))
            self.assertFalse(mask.any())
            self.assertTrue((vectors == 0).all())
            self.assertEqual(result['words'], words)

    def test_context_limits_and_slow_tokenizer_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Word 0'):
            word_chunks(['galletas'], self.tokenizer, 3)
        with self.assertRaisesRegex(ValueError, 'context'):
            encode_word_text(['el'], self.tokenizer, self.model, chunk_size=33)
        with self.assertRaisesRegex(ValueError, 'fast tokenizer'):
            word_chunks(['el'], SimpleNamespace(is_fast=False), 8)

    def test_export_masks_correspondence_and_no_overwrite(self):
        source = self.audio_fixture()
        original = {p.name: p.read_bytes() for p in source.iterdir()}
        output = self.root / 'text_words'
        result = export_text_words(source, output, self.tokenizer, self.model, 'tiny-test', chunk_size=32)
        self.assertEqual(result['feature_shape'], [5, 12])
        self.assertEqual(result['audio_feature_shape'], [5, 3])
        self.assertEqual(result['valid_paired_words'], 3)
        self.assertEqual(torch.load(output / 'text_word_mask.pt', weights_only=True).tolist(), [True, False, True, True, True])
        self.assertEqual(torch.load(output / 'paired_word_mask.pt', weights_only=True).tolist(), [True, False, True, False, True])
        self.assertEqual(json.loads((output / 'metadata.json').read_text()), json.loads(json.dumps(result)))
        self.assertEqual(original, {p.name: p.read_bytes() for p in source.iterdir()})
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            export_text_words(source, output, self.tokenizer, self.model, chunk_size=32)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_changed_word_source_or_wrong_audio_rows_are_rejected(self):
        source = self.audio_fixture()
        path = source / 'metadata.json'
        metadata = json.loads(path.read_text())
        metadata['word_windows'][0]['word_id'] = 4
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'word IDs'):
            read_audio_words(source)
        metadata['word_windows'][0]['word_id'] = 0
        path.write_text(json.dumps(metadata))
        word_source = self.root / 'word_windows.json'
        word_source.write_text(word_source.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Step 5'):
            read_audio_words(source)


if __name__ == '__main__':
    unittest.main()
