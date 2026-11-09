"""Frame packaging preserves word identity, source independence, and validity."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils.audio_frame_timing import frame_timing
from utils.ctc_word_windows import merge_word_windows
from utils.package_fusion_frames import (
    build_package, digest, mean_word_vectors, package_fusion_frames,
    read_fusion_package, schema_fingerprint,
)
from utils.transcript_normalization import normalize_ctc_words
from utils.word_audio_pool import pool_audio_words


class FusionFramePackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.output = self.root / 'exports' / 'pilot'
        self.stages = {name: self.source / name for name in
                       ('frames', 'audio_words', 'text_words', 'windows', 'alignment', 'comparison')}
        for directory in self.stages.values():
            directory.mkdir()

        def timing(stride):
            return frame_timing(SimpleNamespace(conv_kernel=[1], conv_stride=[stride]),
                                20, 10, (20 - 1) // stride + 1)

        self.words = ['a', '...', 'a']
        transcript = normalize_ctc_words(self.words, {'<pad>': 0, '|': 1, 'a': 2}, 0)
        units = [dict(target_index=i, token_id=token_id, token=token, word_id=word_id,
                      start_frame=start, end_frame_exclusive=end, start_time=start / 10,
                      end_time=end / 10, mean_log_emission=-0.2, emission_confidence=0.818730753)
                 for i, (token_id, token, word_id, start, end) in enumerate(
                     [(2, 'a', 0, 1, 3), (1, '|', -1, 3, 4), (2, 'a', 2, 5, 7)])]
        source = {'path': str(self.source / 'audio.wav'), 'size_bytes': 5, 'mtime_ns': 1}
        alignment = {'format': 'ctc_unit_alignment_v1', 'ctc_model': 'ctc', 'blank_id': 0,
                     'words': self.words, 'word_mappings': transcript['word_mappings'], 'units': units,
                     'aligned_frames': 20, 'timing': timing(1),
                     'frame_span_convention': 'start_inclusive_end_exclusive',
                     'audio_source': source, 'word_source': {'path': '/fixture/words.csv'},
                     'confidence_definition': 'test emission support'}
        windows = merge_word_windows(alignment)
        self.frames = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        frame_mask = torch.ones(10, dtype=torch.bool)
        frame_meta = {'format': 'acoustic_frames_v1', 'source': source, 'audio_model': 'audio',
                      'model_revision': 'audio-rev', 'pooled': False, 'augmentation': None,
                      'feature_shape': [10, 2], 'timing': timing(2)}
        self.means, audio_mask, audio_meta = pool_audio_words(self.frames, frame_mask, frame_meta, windows)
        self.text = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        self.text[2] = 0
        text_mask = torch.tensor([True, True, False])
        text_meta = {'words': self.words, 'pooling': 'mean_subwords', 'chunk_size': 200,
                     'tokenizer_sha256': 'tokenizer',
                     'word_mappings': [{'word_id': i, 'word': w} for i, w in enumerate(self.words)],
                     'chunks': [{'chunk_index': 0, 'word_start': 0, 'word_end_exclusive': 3}]}
        comparison = {'words': self.words, 'rows': [
            {'word_id': i, 'word': word, 'whisper_probability': None if i == 0 else .9,
             'delta_start': None if i == 1 else .1, 'delta_end': None if i == 1 else 0.0}
            for i, word in enumerate(self.words)]}
        self.write_json(self.source / 'config.json', {'acoustic_layer': 'last_hidden_state'})
        for stage, name, value in [('frames', 'metadata.json', frame_meta),
                                   ('audio_words', 'metadata.json', audio_meta),
                                   ('text_words', 'metadata.json', text_meta),
                                   ('windows', 'word_windows.json', windows),
                                   ('alignment', 'alignment.json', alignment),
                                   ('comparison', 'comparison.json', comparison)]:
            self.write_json(self.stages[stage] / name, value)
        for stage, name, value in [('frames', 'frames.pt', self.frames),
                                   ('frames', 'valid_mask.pt', frame_mask),
                                   ('audio_words', 'audio_word_embeddings.pt', self.means),
                                   ('text_words', 'text_word_embeddings.pt', self.text),
                                   ('text_words', 'text_word_mask.pt', text_mask)]:
            torch.save(value, self.stages[stage] / name)
        self.hashes = {str(p): digest(p) for p in self.source.rglob('*') if p.is_file()}
        self.audit = {'format': 'fusion_input_audit_v1', 'status': 'passed',
                      'recording_key': 'train/pilot', 'feature_root': str(self.source),
                      'stages': {k: str(v) for k, v in self.stages.items()},
                      'models': {k: {'id': k, 'revision': k + '-rev'} for k in ('audio', 'text', 'ctc')},
                      'verified_sha256': self.hashes}

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def export(self):
        # Source auditing is exercised on the real pilot; these tiny fixtures
        # isolate packaging edge cases without constructing a whole legacy batch.
        with patch('utils.package_fusion_frames.audit_recording', return_value=copy.deepcopy(self.audit)):
            return package_fusion_frames(self.source, 'pilot', self.output)

    def test_cross_grid_mapping_invalid_words_and_exact_reconstruction(self):
        metadata = self.export()
        loaded, tensors = read_fusion_package(self.output)
        self.assertEqual(loaded, metadata)
        self.assertEqual(loaded['words'], self.words)
        self.assertEqual(tensors['word_ids.pt'].tolist(), [0, 1, 2])
        # CTC indices were 1:3 and 5:7; acoustic indices use their own grid.
        self.assertEqual(tensors['word_frame_intervals.pt'].tolist(), [[1, 2], [-1, -1], [3, 4]])
        self.assertEqual(tensors['audio_word_mask.pt'].tolist(), [True, False, True])
        self.assertEqual(tensors['text_word_mask.pt'].tolist(), [True, True, False])
        self.assertEqual(tensors['paired_word_mask.pt'].tolist(), [True, False, False])
        torch.testing.assert_close(mean_word_vectors(tensors), self.means, rtol=0, atol=0)
        torch.testing.assert_close(tensors['frames.pt'], self.frames, rtol=0, atol=0)
        torch.testing.assert_close(tensors['text_words.pt'], self.text, rtol=0, atol=0)
        self.assertEqual(tuple(tensors['word_quality.pt'].shape), (3, 0))
        self.assertEqual(tuple(tensors['recording_quality.pt'].shape), (0,))
        self.assertIsNone(metadata['quality_sources']['comparison_rows'][0]['whisper_probability'])
        self.assertFalse(tensors['frames.pt'].requires_grad)
        self.assertEqual({p: digest(p) for p in self.hashes}, self.hashes)
        self.assertNotEqual((self.output / 'frames.pt').stat().st_ino,
                            (self.stages['frames'] / 'frames.pt').stat().st_ino)

    def test_relocation_and_source_removal_do_not_affect_loading(self):
        self.export()
        moved = self.root / 'relocated'
        shutil.move(self.output, moved)
        shutil.rmtree(self.source)
        metadata, tensors = read_fusion_package(moved)
        self.assertEqual(metadata['uid'], 'pilot')
        torch.testing.assert_close(mean_word_vectors(tensors), self.means, rtol=0, atol=0)

    def test_existing_export_is_refused_before_audit(self):
        self.export()
        before = {p.name: digest(p) for p in self.output.iterdir()}
        with patch('utils.package_fusion_frames.audit_recording', side_effect=AssertionError('No re-audit')):
            with self.assertRaises(FileExistsError):
                package_fusion_frames(self.source, 'pilot', self.output)
        self.assertEqual(before, {p.name: digest(p) for p in self.output.iterdir()})

    def test_changed_source_is_rejected_before_creating_export_directory(self):
        def changed(audit):
            result = build_package(audit)
            (self.source / 'config.json').write_text('changed')
            return result
        with patch('utils.package_fusion_frames.build_package', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'source changed while packaging'):
                self.export()
        self.assertFalse(self.output.parent.exists())

    def test_tensor_tampering_is_detected(self):
        self.export()
        with (self.output / 'frames.pt').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Tensor hash differs: frames.pt'):
            read_fusion_package(self.output)

    def test_rehashed_invalid_intervals_and_masks_are_still_rejected(self):
        self.export()
        for name, value, message in [
            ('word_frame_intervals.pt', torch.tensor([[1, 20], [-1, -1], [3, 4]]), 'Invalid acoustic interval'),
            ('word_frame_intervals.pt', torch.tensor([[1, 2], [0, 0], [3, 4]]), 'Invalid acoustic sentinel'),
            ('paired_word_mask.pt', torch.tensor([True, True, True]), 'Invalid paired word mask'),
        ]:
            with self.subTest(name=name, message=message):
                original = (self.output / name).read_bytes()
                metadata = json.loads((self.output / 'metadata.json').read_text())
                torch.save(value, self.output / name)
                changed = copy.deepcopy(metadata)
                changed['tensors'][name]['sha256'] = digest(self.output / name)
                self.write_json(self.output / 'metadata.json', changed)
                with self.assertRaisesRegex(ValueError, message):
                    read_fusion_package(self.output)
                (self.output / name).write_bytes(original)
                self.write_json(self.output / 'metadata.json', metadata)

    def test_schema_and_nonfinite_data_are_rejected(self):
        self.export()
        metadata = json.loads((self.output / 'metadata.json').read_text())
        changed = copy.deepcopy(metadata)
        changed['schema']['audio_width'] = 999
        self.write_json(self.output / 'metadata.json', changed)
        with self.assertRaisesRegex(ValueError, 'Schema fingerprint differs'):
            read_fusion_package(self.output)
        changed['schema_sha256'] = schema_fingerprint(changed['schema'])
        self.write_json(self.output / 'metadata.json', changed)
        with self.assertRaisesRegex(ValueError, 'Encoder widths differ'):
            read_fusion_package(self.output)
        values = self.text.clone()
        values[0, 0] = float('nan')
        torch.save(values, self.output / 'text_words.pt')
        metadata['tensors']['text_words.pt']['sha256'] = digest(self.output / 'text_words.pt')
        self.write_json(self.output / 'metadata.json', metadata)
        with self.assertRaisesRegex(ValueError, 'Nonfinite tensor: text_words.pt'):
            read_fusion_package(self.output)

    def test_in_bounds_interval_with_wrong_encoder_grid_is_rejected(self):
        self.export()
        metadata = json.loads((self.output / 'metadata.json').read_text())
        intervals = torch.tensor([[2, 3], [-1, -1], [3, 4]])
        torch.save(intervals, self.output / 'word_frame_intervals.pt')
        metadata['tensors']['word_frame_intervals.pt']['sha256'] = digest(
            self.output / 'word_frame_intervals.pt')
        metadata['word_windows'][0].update(audio_start_frame=2, audio_end_frame_exclusive=3)
        self.write_json(self.output / 'metadata.json', metadata)
        with self.assertRaisesRegex(ValueError, 'disagrees with timing-based mapping'):
            read_fusion_package(self.output)

    def test_interrupted_publication_has_no_completion_metadata(self):
        import os
        link = os.link
        def fail_on_text(source, target):
            if Path(source).name == 'text_words.pt':
                raise OSError('simulated publication failure')
            return link(source, target)
        with patch('utils.package_fusion_frames.os.link', side_effect=fail_on_text):
            with self.assertRaisesRegex(OSError, 'simulated publication failure'):
                self.export()
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / 'metadata.json').exists())
        self.assertFalse(list(self.output.parent.glob('.fusion-package-*')))
        with self.assertRaises(FileNotFoundError):
            read_fusion_package(self.output)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
