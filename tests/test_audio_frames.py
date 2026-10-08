"""Offline frame-export checks; no model downloads or clinical data required."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
import torchaudio
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from utils.audio_frame_timing import frame_interval_seconds, frame_timing, time_to_frame_index
from utils.prepro_audio_frames import export_recording_frames, extract_recording_frames
from utils.prepro_embeddings import extract_audio_features


class AudioProcessor:
    """Use the real HF waveform processor without an unused ASR tokenizer."""

    def __init__(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=False,
        )

    def __call__(self, *args, **kwargs):
        return self.feature_extractor(*args, **kwargs)


class FrameTimingTests(unittest.TestCase):
    def test_standard_grid_and_roundtrip(self):
        timing = frame_timing(Wav2Vec2Config(), 640000, 16000, 1999)
        self.assertEqual(timing['frame_stride_samples'], 320)
        self.assertEqual(timing['receptive_field_samples'], 400)
        self.assertEqual(frame_interval_seconds(0, timing), (0.0, 0.025))
        for index in (0, 1, 123, 1998):
            start, end = frame_interval_seconds(index, timing)
            self.assertEqual(time_to_frame_index((start + end) / 2, timing), index)
            self.assertLessEqual(end, 40)
        self.assertEqual(time_to_frame_index(0, timing), 0)
        self.assertEqual(time_to_frame_index(40, timing), 1998)

    def test_invalid_geometry_and_boundaries(self):
        config = Wav2Vec2Config()
        with self.assertRaises(ValueError):
            frame_timing(config, 640000, 16000, 2000)
        config.add_adapter = True
        with self.assertRaises(ValueError):
            frame_timing(config, 640000, 16000, 1999)
        timing = frame_timing(Wav2Vec2Config(), 16000, 16000, 49)
        for index in (-1, 49):
            with self.assertRaises(ValueError):
                frame_interval_seconds(index, timing)
        for seconds in (-1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                time_to_frame_index(seconds, timing)


class FrameExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small actual encoder, with the same temporal geometry as the baseline.
        config = Wav2Vec2Config(
            hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, conv_dim=(4,) * 7,
            num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
            mask_time_prob=0.0,
        )
        cls.model = Wav2Vec2Model(config).eval()
        cls.processor = AudioProcessor()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.audio_path = self.root / 'recording.wav'
        # Stereo and a non-integer resampling ratio exercise sample counts.
        samples = torch.arange(5513) / 11025
        signal = torch.sin(2 * torch.pi * 220 * samples) * 0.1
        torchaudio.save(self.audio_path, torch.stack((signal, signal * 0.5)), 11025)

    def test_features_match_existing_extractor(self):
        original, _ = extract_audio_features(
            self.audio_path, 'wav2vec2', (self.processor, self.model), 'cpu',
        )
        frames, mask, timing = extract_recording_frames(
            self.audio_path, self.processor, self.model, 'cpu',
        )
        torch.testing.assert_close(frames, original, rtol=0, atol=0)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(mask.all())
        self.assertEqual(len(mask), len(frames))
        self.assertEqual(timing['source_channels'], 2)
        self.assertEqual(timing['num_samples'], 8001)
        self.assertEqual(timing['padding'], 'none')
        self.assertLessEqual(frame_interval_seconds(len(frames) - 1, timing)[1],
                             timing['audio_duration_seconds'])

    def test_saved_tensors_and_no_overwrite(self):
        output = self.root / 'frames'
        export_recording_frames(self.audio_path, output, self.processor, self.model, 'cpu')
        frames = torch.load(output / 'frames.pt', weights_only=True)
        mask = torch.load(output / 'valid_mask.pt', weights_only=True)
        metadata = json.loads((output / 'metadata.json').read_text())
        self.assertEqual(metadata['feature_shape'], list(frames.shape))
        self.assertEqual(metadata['timing']['valid_frame_count'], int(mask.sum()))
        self.assertFalse(metadata['pooled'])
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            export_recording_frames(self.audio_path, output, self.processor, self.model, 'cpu')
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_short_audio_fails_before_creating_output(self):
        torchaudio.save(self.audio_path, torch.zeros(1, 100), 16000)
        output = self.root / 'frames'
        with self.assertRaises(ValueError):
            export_recording_frames(self.audio_path, output, self.processor, self.model, 'cpu')
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
