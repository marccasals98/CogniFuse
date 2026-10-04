"""Verify the Step 7/8 bridge against the actual existing loader and classifier."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils.package_word_embeddings import AUDIO_SUFFIX, TEXT_SUFFIX, package_word_embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from data import PrecomputedADDataset
from model import Classifier
from settings import TRAIN_DEFAULT_SETTINGS


class PackageWordTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.audio, self.text = self.root / 'audio', self.root / 'text'
        self.audio.mkdir()
        self.text.mkdir()
        self.output = self.root / 'packaged'
        self.words = ['hola', 'hola', '...', 'fin']
        self.audio_mask = torch.tensor([True, True, False, True])
        self.text_mask = torch.tensor([True, False, True, True])
        windows = self.root / 'windows.json'
        windows.write_text(json.dumps({'words': self.words, 'word_alignment_mask': self.audio_mask.tolist()}))
        audio_meta = {
            'format': 'audio_word_embeddings_v1', 'words': self.words,
            'feature_shape': [4, 8], 'valid_word_count': 3,
            'audio_source': {'path': '/source/recording.mp3'}, 'word_source': {},
            'word_windows': [{'word_id': i, 'word': w, 'status': 'aligned' if self.audio_mask[i] else 'no_ctc_units'}
                             for i, w in enumerate(self.words)],
            'inputs': {'word_windows': {'path': str(windows), 'sha256': self.sha(windows)}},
        }
        self.write_json(self.audio / 'metadata.json', audio_meta)
        audio = torch.randn(4, 8)
        audio[~self.audio_mask] = 0
        text = torch.randn(4, 8)
        text[~self.text_mask] = 0
        torch.save(audio, self.audio / 'audio_word_embeddings.pt')
        torch.save(self.audio_mask, self.audio / 'word_mask.pt')
        torch.save(text, self.text / 'text_word_embeddings.pt')
        torch.save(self.text_mask, self.text / 'text_word_mask.pt')
        torch.save(self.audio_mask & self.text_mask, self.text / 'paired_word_mask.pt')
        self.text_meta = {
            'format': 'text_word_embeddings_v1', 'words': self.words,
            'word_mappings': [{'word_id': i, 'word': w, 'status': 'encoded' if self.text_mask[i] else 'no_text_tokens'}
                              for i, w in enumerate(self.words)],
            'feature_shape': [4, 8], 'audio_feature_shape': [4, 8],
            'valid_audio_words': 3, 'valid_text_words': 3, 'valid_paired_words': 2,
            'audio_source': audio_meta['audio_source'], 'word_source': {},
            'text_model': 'test-encoder', 'model_revision': 'test-revision',
            'inputs': {name: {'sha256': self.sha(self.audio / name)} for name in
                       ('metadata.json', 'audio_word_embeddings.pt', 'word_mask.pt')},
        }
        self.write_json(self.text / 'metadata.json', self.text_meta)

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write_json(path, obj):
        path.write_text(json.dumps(obj))

    def export(self):
        return package_word_embeddings(self.audio, self.text, self.output)

    def test_exact_vectors_word_ids_paired_masks_and_no_overwrite(self):
        before = {p: p.read_bytes() for d in (self.audio, self.text) for p in d.iterdir()}
        result = self.export()
        self.assertEqual(result['words'], self.words)
        self.assertEqual(result['word_ids'], [0, 1, 2, 3])
        self.assertEqual(result['paired_mask'], [True, False, False, True])
        for directory, source, suffix in ((self.audio, 'audio_word_embeddings.pt', AUDIO_SUFFIX),
                                          (self.text, 'text_word_embeddings.pt', TEXT_SUFFIX)):
            torch.testing.assert_close(torch.load(directory / source, weights_only=True),
                                       torch.load(self.output / ('recording' + suffix), weights_only=True), rtol=0, atol=0)
            mask = torch.load(self.output / ('recording' + suffix[:-3] + '_mask.pt'), weights_only=True)
            self.assertEqual(mask.tolist(), result['paired_mask'])
        saved = {p: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(saved, {p: p.read_bytes() for p in self.output.iterdir()})
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_source_hash_mismatch_rejected_before_writing(self):
        torch.save(torch.randn(4, 8), self.audio / 'audio_word_embeddings.pt')
        # Keep invalid rows zero so the provenance check is what fails.
        audio = torch.load(self.audio / 'audio_word_embeddings.pt', weights_only=True)
        audio[~self.audio_mask] = 0
        torch.save(audio, self.audio / 'audio_word_embeddings.pt')
        with self.assertRaisesRegex(ValueError, 'different acoustic inputs'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_same_size_reordered_word_ids_rejected(self):
        self.text_meta['word_mappings'][0]['word_id'] = 1
        self.write_json(self.text / 'metadata.json', self.text_meta)
        with self.assertRaisesRegex(ValueError, 'word IDs'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_invalid_paired_mask_rejected(self):
        torch.save(torch.ones(4, dtype=torch.bool), self.text / 'paired_word_mask.pt')
        with self.assertRaisesRegex(ValueError, 'intersection'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_nonfinite_text_rejected(self):
        text = torch.load(self.text / 'text_word_embeddings.pt', weights_only=True)
        text[0, 0] = float('nan')
        torch.save(text, self.text / 'text_word_embeddings.pt')
        with self.assertRaisesRegex(ValueError, 'Invalid text'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_existing_loader_classifier_padding_and_backward(self):
        self.export()
        labels = self.root / 'labels.csv'
        labels.write_text('filename,patient_id,diagnosis\nrecording.mp3,test-patient,svPPA\n')
        params = SimpleNamespace(**{
            **TRAIN_DEFAULT_SETTINGS, 'precomputed_features_dir': str(self.output),
            'train_labels_path': str(labels), 'validation_labels_path': str(labels),
            'precomputed_audio_suffix': AUDIO_SUFFIX, 'precomputed_text_suffix': TEXT_SUFFIX,
            'speech_feature_extractor_output_vectors_dimension': 8,
            'text_feature_extractor_output_vectors_dimension': 8,
            'seq_to_seq_method': 'MultiHeadAttention', 'seq_to_seq_heads_number': 4,
            'seq_to_one_method': 'AttentionPooling', 'classifier_hidden_layers_width': 16,
        })
        dataset = PrecomputedADDataset(params, split='train', fold=1)
        item = dataset[0]
        self.assertEqual(item[1].item(), 2)
        # Different lengths exercise actual batch padding and the paired masks.
        shorter = (item[0][:1], item[1], item[2][:1], item[3][:1], item[4][:1])
        speech, targets, text, masks = dataset.collate_fn([item, shorter])
        with patch('model.SpeechFeatureExtractor', side_effect=AssertionError('No speech encoder needed')), \
             patch('model.TextFeatureExtractor', side_effect=AssertionError('No text encoder needed')):
            model = Classifier(params, torch.device('cpu')).eval()
        logits = model(speech, text, masks)
        self.assertEqual(logits.shape, (2, 3))
        self.assertTrue(torch.isfinite(logits).all())
        changed_speech, changed_text = speech.clone(), text.clone()
        changed_speech[~masks[:, 0]] = 10000
        changed_text[~masks[:, 1]] = -10000
        torch.testing.assert_close(logits, model(changed_speech, changed_text, masks), rtol=0, atol=0)
        single = model(speech[:1], text[:1], masks[:1])
        torch.testing.assert_close(logits[:1], single, rtol=1e-5, atol=1e-6)
        torch.nn.functional.cross_entropy(logits, targets).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for _, p in model.classifier_layer.named_parameters()))


if __name__ == '__main__':
    unittest.main()
