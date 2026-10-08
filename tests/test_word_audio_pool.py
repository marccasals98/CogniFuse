"""Mean-pooling and cross-grid mapping tests using known acoustic vectors."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from utils.audio_frame_timing import frame_timing
from utils.ctc_word_windows import merge_word_windows
from utils.transcript_normalization import normalize_ctc_words
from utils.word_audio_pool import export_audio_words, map_frame_window, pool_audio_words


def timing(stride=1):
    count = (20 - 1) // stride + 1
    return frame_timing(SimpleNamespace(conv_kernel=[1], conv_stride=[stride]), 20, 10, count)


def fixture(context=0, stride=1):
    transcript = normalize_ctc_words(['a', '...', 'a'], {'<pad>': 0, '|': 1, 'a': 2}, 0)
    units = [
        {'target_index': 0, 'token_id': 2, 'token': 'a', 'word_id': 0,
         'start_frame': 1, 'end_frame_exclusive': 3, 'start_time': .1, 'end_time': .3},
        {'target_index': 1, 'token_id': 1, 'token': '|', 'word_id': -1,
         'start_frame': 3, 'end_frame_exclusive': 4, 'start_time': .3, 'end_time': .4},
        {'target_index': 2, 'token_id': 2, 'token': 'a', 'word_id': 2,
         'start_frame': 5, 'end_frame_exclusive': 7, 'start_time': .5, 'end_time': .7},
    ]
    source = {'path': '/fixture/recording.wav', 'size_bytes': 123, 'mtime_ns': 456}
    alignment = {
        'format': 'ctc_unit_alignment_v1', 'ctc_model': 'test-ctc', 'blank_id': 0,
        'words': transcript['words'], 'word_mappings': transcript['word_mappings'], 'units': units,
        'aligned_frames': 20, 'timing': timing(),
        'frame_span_convention': 'start_inclusive_end_exclusive',
        'audio_source': source, 'word_source': {'path': '/fixture/words.csv'},
    }
    windows = merge_word_windows(alignment, context)
    t = timing(stride)
    features = torch.arange(t['frame_count'] * 2, dtype=torch.float32).reshape(-1, 2)
    metadata = {'format': 'acoustic_frames_v1', 'source': dict(source),
                'audio_model': 'test-acoustic', 'augmentation': None, 'pooled': False,
                'feature_shape': list(features.shape), 'timing': t}
    return features, torch.ones(len(features), dtype=torch.bool), metadata, windows, alignment


class WordAudioPoolingTests(unittest.TestCase):
    def test_exact_means_and_original_word_order(self):
        features, mask, metadata, windows, _ = fixture()
        before = features.clone()
        embeddings, word_mask, output = pool_audio_words(features, mask, metadata, windows)
        self.assertEqual(tuple(embeddings.shape), (3, 2))
        self.assertEqual(output['words'], ['a', '...', 'a'])
        self.assertEqual(word_mask.tolist(), [True, False, True])
        torch.testing.assert_close(embeddings[0], features[1:3].mean(0), rtol=0, atol=0)
        torch.testing.assert_close(embeddings[2], features[5:7].mean(0), rtol=0, atol=0)
        self.assertTrue((embeddings[1] == 0).all())
        self.assertIsNone(output['word_windows'][1]['audio_start_frame'])
        self.assertEqual(output['word_windows'][2]['mapping_method'], 'identical_grid')
        torch.testing.assert_close(features, before)
        self.assertFalse(embeddings.requires_grad)

    def test_context_windows_are_used(self):
        features, mask, metadata, windows, _ = fixture(context=2)
        embeddings, _, output = pool_audio_words(features, mask, metadata, windows)
        torch.testing.assert_close(embeddings[0], features[0:5].mean(0))
        torch.testing.assert_close(embeddings[2], features[3:9].mean(0))
        self.assertEqual(output['context_frames'], 2)

    def test_different_grids_use_time_not_ctc_indices(self):
        features, mask, metadata, windows, _ = fixture(stride=2)
        embeddings, _, output = pool_audio_words(features, mask, metadata, windows)
        torch.testing.assert_close(embeddings[0], features[1])
        torch.testing.assert_close(embeddings[2], features[3])
        self.assertEqual(output['word_windows'][2]['audio_start_frame'], 3)
        self.assertEqual(output['word_windows'][2]['ctc_start_frame'], 5)
        self.assertEqual(output['word_windows'][2]['mapping_method'], 'frame_centers_in_interval')

    def test_short_interval_fallback_stays_inside_valid_frames(self):
        self.assertEqual(map_frame_window(1, 2, timing(), timing(4)), (0, 1, 'nearest_center_fallback'))
        self.assertEqual(map_frame_window(19, 20, timing(), timing(4)), (4, 5, 'nearest_center_fallback'))
        with self.assertRaises(ValueError):
            map_frame_window(19, 21, timing(), timing(4))

    def test_padded_frames_never_enter_mean(self):
        features, mask, metadata, windows, _ = fixture(context=100)
        expected = features.mean(0)
        features = torch.cat((features, torch.full((3, 2), float('nan'))))
        mask = torch.cat((mask, torch.zeros(3, dtype=torch.bool)))
        metadata['feature_shape'] = list(features.shape)
        embeddings, _, output = pool_audio_words(features, mask, metadata, windows)
        torch.testing.assert_close(embeddings[0], expected)
        torch.testing.assert_close(embeddings[2], expected)
        self.assertEqual(output['word_windows'][0]['audio_end_frame_exclusive'], 20)

    def test_invalid_inputs_fail(self):
        for issue in ('source', 'duration', 'mask_hole', 'nonfinite', 'word_order', 'context', 'augmentation'):
            with self.subTest(issue=issue):
                features, mask, metadata, windows, _ = fixture()
                if issue == 'source':
                    metadata['source']['path'] = '/different.wav'
                elif issue == 'duration':
                    windows['timing'] = frame_timing(SimpleNamespace(conv_kernel=[1], conv_stride=[1]), 40, 10, 40)
                elif issue == 'mask_hole':
                    mask[1] = False
                elif issue == 'nonfinite':
                    features[1, 0] = float('nan')
                elif issue == 'word_order':
                    windows['word_windows'][0]['word_id'] = 2
                elif issue == 'context':
                    windows['word_windows'][0]['expanded_start_frame'] = 0
                else:
                    metadata['augmentation'] = 'speed'
                with self.assertRaises(ValueError):
                    pool_audio_words(features, mask, metadata, windows)

    def test_export_preserves_inputs_and_refuses_overwrite(self):
        features, mask, metadata, windows, alignment = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir, window_dir, output = root / 'frames', root / 'windows', root / 'pooled'
            audio_dir.mkdir()
            window_dir.mkdir()
            source_path = root / 'alignment.json'
            source_path.write_text(json.dumps(alignment))
            windows['alignment_source'] = {'path': str(source_path), 'sha256': hashlib.sha256(source_path.read_bytes()).hexdigest()}
            (audio_dir / 'metadata.json').write_text(json.dumps(metadata))
            (window_dir / 'word_windows.json').write_text(json.dumps(windows))
            torch.save(features, audio_dir / 'frames.pt')
            torch.save(mask, audio_dir / 'valid_mask.pt')
            paths = [p for p in root.rglob('*') if p.is_file()]
            before = {p: p.read_bytes() for p in paths}
            exported = export_audio_words(audio_dir, window_dir, output)
            self.assertEqual(exported['feature_shape'], [3, 2])
            self.assertEqual(torch.load(output / 'word_mask.pt', weights_only=True).tolist(), [True, False, True])
            self.assertEqual(json.loads((output / 'metadata.json').read_text()), exported)
            self.assertEqual(before, {p: p.read_bytes() for p in paths})
            saved = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                export_audio_words(audio_dir, window_dir, output)
            self.assertEqual(saved, {p.name: p.read_bytes() for p in output.iterdir()})
            source_path.write_text(source_path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'alignment changed'):
                export_audio_words(audio_dir, window_dir, root / 'new-output')
            self.assertFalse((root / 'new-output').exists())


if __name__ == '__main__':
    unittest.main()
