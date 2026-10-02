"""CTC preparation tests using synthetic audio and a tiny local checkpoint."""

import csv
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torchaudio
from transformers import (Wav2Vec2Config, Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor)

from scripts.ctc_encoder import SpanishCTCEncoder
from utils.prepro_ctc import export_ctc
from utils.transcript_normalization import normalize_ctc_words


VOCAB = {token: index for index, token in enumerate(
    ['<pad>', '<s>', '</s>', '<unk>', '|', "'", '-']
    + list('abcdefghijklmnopqrstuvwxyzáéíñóöúü'))}


class NormalizationTests(unittest.TestCase):
    def test_accents_repeated_words_and_word_ids(self):
        words = ['¡NIÑO!', 'esta\u0301', 'niño', '...','pingüino.']
        result = normalize_ctc_words(words, VOCAB, 0)
        self.assertEqual(result['words'], words)
        self.assertEqual(result['ctc_text'], 'niño está niño pingüino')
        self.assertEqual([w['word_id'] for w in result['word_mappings']], list(range(5)))
        self.assertEqual(result['word_mappings'][3]['status'], 'no_ctc_units')
        for word in result['word_mappings']:
            start, end = word['token_start'], word['token_end']
            self.assertEqual(result['tokens'][start:end], list(word['ctc_form']))
            self.assertEqual(result['token_word_ids'][start:end], [word['word_id']] * (end - start))
        for token, owner in zip(result['tokens'], result['token_word_ids']):
            if token == '|':
                self.assertEqual(owner, -1)
        self.assertNotIn(0, result['target_ids'])
        self.assertNotIn(VOCAB['<unk>'], result['target_ids'])
        self.assertEqual(result, normalize_ctc_words(words, VOCAB, 0))

    def test_unsupported_words_fail_with_index(self):
        for word in ('12', '€', 'café☃', 'dos palabras', 'ç'):
            with self.subTest(word=word), self.assertRaisesRegex(ValueError, '[Ww]ord 1'):
                normalize_ctc_words(['el', word], VOCAB, 0)

    def test_empty_words_are_preserved(self):
        result = normalize_ctc_words(['', '…'], VOCAB, 0)
        self.assertEqual(result['target_ids'], [])
        self.assertEqual(len(result['word_mappings']), 2)


class CTCEncoderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        # Release checkpoint-backed tensor mappings before removing NFS files.
        self.addCleanup(gc.collect)
        self.root = Path(self.directory.name)
        vocab_path = self.root / 'vocab.json'
        vocab_path.write_text(json.dumps(VOCAB), encoding='utf-8')
        tokenizer = Wav2Vec2CTCTokenizer(str(vocab_path), word_delimiter_token='|', do_lower_case=False)
        processor = Wav2Vec2Processor(
            Wav2Vec2FeatureExtractor(sampling_rate=16000, return_attention_mask=True), tokenizer,
        )
        model = Wav2Vec2ForCTC(Wav2Vec2Config(
            hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, conv_dim=(4,) * 7, vocab_size=len(VOCAB),
            num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
            mask_time_prob=0.0, pad_token_id=0,
        ))
        self.encoder = SpanishCTCEncoder(processor, model, 'tiny-test-model')
        self.waveform = torch.sin(torch.arange(16000) * 0.04) * 0.1

    def test_emissions_match_full_ctc_model(self):
        self.encoder.train()  # Inference must disable training-time masking/dropout.
        result = self.encoder(self.waveform)
        inputs = self.encoder.processor(self.waveform.numpy(), sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            reference = self.encoder.model(**inputs).logits
        torch.testing.assert_close(result['ctc_logits'], reference, rtol=0, atol=0)
        self.assertEqual(tuple(reference.shape), (1, 49, len(VOCAB)))
        torch.testing.assert_close(result['ctc_log_probs'].exp().sum(-1), torch.ones(1, 49))
        self.assertTrue(result['valid_frame_mask'].all())
        self.assertFalse(result['ctc_logits'].requires_grad)
        self.assertFalse(any(p.requires_grad for p in self.encoder.parameters()))

    def test_load_local_complete_checkpoint(self):
        checkpoint = self.root / 'checkpoint'
        self.encoder.model.save_pretrained(checkpoint)
        self.encoder.processor.save_pretrained(checkpoint)
        loaded = SpanishCTCEncoder.from_pretrained(str(checkpoint), local_files_only=True)
        torch.testing.assert_close(loaded(self.waveform)['ctc_logits'],
                                   self.encoder(self.waveform)['ctc_logits'], rtol=0, atol=0)

    def test_reject_missing_trained_head(self):
        with patch('scripts.ctc_encoder.Wav2Vec2ForCTC.from_pretrained',
                   return_value=(self.encoder.model, {'missing_keys': ['lm_head.weight']})):
            with self.assertRaisesRegex(ValueError, 'complete trained CTC'):
                SpanishCTCEncoder.from_pretrained('encoder-only', local_files_only=True)

    def test_input_and_vocabulary_validation(self):
        for waveform, rate in [(self.waveform, 8000), (self.waveform[None], 16000),
                               (torch.zeros(10), 16000), (torch.tensor([float('nan')]), 16000)]:
            with self.assertRaises(ValueError):
                self.encoder(waveform, rate)
        self.encoder.model.config.pad_token_id = 1
        with self.assertRaisesRegex(ValueError, 'blank'):
            SpanishCTCEncoder(self.encoder.processor, self.encoder.model)

    def test_export_and_preserve_existing_files(self):
        audio = self.root / 'audio.wav'
        words = self.root / 'words.csv'
        torchaudio.save(audio, self.waveform[None], 16000)
        with words.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['word', 'start', 'end', 'probability'])
            writer.writerows([['¡NIÑO!', 0.1, 0.2, 0.9], ['está', 0.3, 0.4, 0.9]])
        original = words.read_bytes()
        output = self.root / 'ctc'
        metadata, transcript = export_ctc(audio, words, output, self.encoder)
        saved = torch.load(output / 'ctc_log_probs.pt', weights_only=True)
        self.assertEqual(list(saved.shape), metadata['logits_shape'])
        self.assertFalse(metadata['alignment_computed'])
        self.assertEqual(transcript['words'], ['¡NIÑO!', 'está'])
        self.assertEqual(json.loads((output / 'transcript.json').read_text()), transcript)
        self.assertEqual(words.read_bytes(), original)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises(FileExistsError):
            export_ctc(audio, words, output, self.encoder)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})


if __name__ == '__main__':
    unittest.main()
