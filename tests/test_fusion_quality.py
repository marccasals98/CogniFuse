"""Raw measurement extraction, portable enrichment, and leakage-free statistics."""

import copy
import json
from pathlib import Path
import shutil
import unittest
import wave
from unittest.mock import Mock

import numpy as np
import torch

import test_frame_inputs as frame_fixtures
from scripts.data import make_precomputed_dataset
from scripts.frame_inputs import collate_frame_items, frame_item
from scripts.reliability import (fit_quality_normalization, transform_quality,
                                 validate_training_quality_state)
from scripts.evaluate import FrameEvaluationDataset, evaluate, parse_args
from utils.fusion_quality import recording_quality, quality_schema, word_quality
from utils.extract_fusion_quality import enrich_package
from utils.package_fusion_frames import digest, read_fusion_package
from train import Trainer


class FusionQualityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = frame_fixtures.FrameInputTests('test_real_training_entrypoint_one_epoch')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.audio = self.root / 'waveforms'
        self.audio.mkdir()
        self.ready = self.root / 'quality_features'
        for index, uid in enumerate(('a1', 'a2', 'c1', 'c2', 't1')):
            rate = 16000
            t = torch.arange(6400) / rate
            waveform = .2 * torch.sin(2 * torch.pi * 220 * t)
            waveform += .02 * torch.randn(6400, generator=torch.Generator().manual_seed(index))
            path = self.audio / (uid + '.wav')
            with wave.open(str(path), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(rate)
                stream.writeframes((waveform.numpy() * 32767).astype('<i2').tobytes())
            package = self.fixture.features / uid
            metadata_path = package / 'metadata.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['provenance']['verified_sha256'][str(path)] = digest(path)
            metadata['quality_sources']['comparison_rows'][0]['whisper_probability'] = .2 if uid == 'a2' else .8
            metadata_path.write_text(json.dumps(metadata))

    def enrich(self, uid='a2'):
        return enrich_package(self.fixture.features / uid, self.ready / uid)

    def ready_dataset(self):
        for uid in ('a1', 'a2', 'c1', 'c2', 't1'):
            if not (self.ready / uid).exists():
                self.enrich(uid)
        metadata = json.loads((self.ready / 'a2/metadata.json').read_text())
        self.fixture.features = self.ready
        params = self.fixture.params
        params.precomputed_features_dir = str(self.ready)
        params.frame_package_schema = metadata['schema']
        params.frame_package_schema_sha256 = metadata['schema_sha256']
        return make_precomputed_dataset(params, split='train', fold=0)

    def replace_tensor(self, directory, name, value):
        path = directory / 'metadata.json'
        metadata = json.loads(path.read_text())
        torch.save(value, directory / name)
        metadata['tensors'][name]['sha256'] = digest(directory / name)
        if name == 'paired_word_mask.pt':
            metadata['valid_counts']['paired_words'] = int(value.sum())
        if name == 'text_word_mask.pt':
            metadata['valid_counts']['text_words'] = int(value.sum())
        path.write_text(json.dumps(metadata))

    def test_word_confidence_missingness_and_weighted_ctc_aggregation(self):
        metadata = json.loads((self.fixture.features / 'a2/metadata.json').read_text())
        metadata['quality_sources']['comparison_rows'][0]['whisper_probability'] = None
        # Same character word, different nonblank span lengths: weighted log mean.
        first = metadata['quality_sources']['ctc_units'][0]
        first['mean_log_emission'] = -1
        extra = copy.deepcopy(first)
        extra.update(start_frame=3, end_frame_exclusive=4, mean_log_emission=-4)
        metadata['quality_sources']['ctc_units'].append(extra)
        values, observed = word_quality(metadata)
        self.assertEqual(tuple(values.shape), (3, 4))
        self.assertFalse(observed[0, 0])
        self.assertEqual(values[0, 0].item(), 0)
        torch.testing.assert_close(values[0, 1], torch.tensor(np.exp(-2), dtype=torch.float32))
        self.assertFalse(observed[1, 1])
        self.assertTrue(observed[2, 1])
        self.assertTrue(observed[0, 3])  # Observed zero boundary difference is valid.
        self.assertEqual(values[0, 3].item(), 0)
        metadata['quality_sources']['comparison_rows'][0]['word_id'] = 2
        with self.assertRaisesRegex(ValueError, 'word IDs differ'):
            word_quality(metadata)

    def test_recording_clipping_silence_and_short_recordings(self):
        source = torch.stack((torch.ones(3200), -torch.ones(3200)))
        values, observed, _ = recording_quality(source, 16000)
        self.assertEqual(values[1].item(), 1)  # Before mono cancellation.
        self.assertEqual(observed.tolist(), [False, True, False])
        values, observed, _ = recording_quality(torch.zeros(1, 3200), 16000)
        self.assertEqual(values.tolist(), [0, 0, 0])
        self.assertEqual(observed.tolist(), [False, True, False])
        _, observed, info = recording_quality(torch.ones(1, 100), 16000)
        self.assertEqual(observed.tolist(), [False, True, False])
        self.assertIn('too_few', info['missing_reason'])

    def test_noise_proxies_respond_to_controlled_noise(self):
        rate = 16000
        t = torch.arange(rate) / rate
        tone = .3 * torch.sin(2 * torch.pi * 200 * t)
        noise = torch.randn(rate, generator=torch.Generator().manual_seed(2))
        speech = tone * (t > .5)
        clean, _, _ = recording_quality((speech + .001 * noise)[None], rate)
        noisy, _, _ = recording_quality((speech + .1 * noise)[None], rate)
        self.assertGreater(float(clean[0]), float(noisy[0]))
        harmonic, _, _ = recording_quality(tone[None], rate)
        broadband, _, _ = recording_quality((.1 * noise)[None], rate)
        self.assertGreater(float(broadband[2]), float(harmonic[2]) + .1)

    def test_enrichment_preserves_features_and_loads_without_sources(self):
        source = self.fixture.features / 'a2'
        before = {p.name: digest(p) for p in source.iterdir()}
        metadata = self.enrich()
        self.assertEqual(metadata['schema']['quality'], quality_schema())
        _, tensors = read_fusion_package(self.ready / 'a2')
        self.assertEqual(tuple(tensors['word_quality.pt'].shape), (3, 4))
        self.assertEqual(tuple(tensors['recording_quality.pt'].shape), (3,))
        self.assertEqual(tensors['recording_quality_observed.pt'].tolist(), [True] * 3)
        for name, checksum in before.items():
            if name != 'metadata.json' and 'quality' not in name:
                self.assertEqual(digest(self.ready / 'a2' / name), checksum)
        self.assertEqual({p.name: digest(p) for p in source.iterdir()}, before)
        shutil.rmtree(self.audio)
        shutil.rmtree(source)
        read_fusion_package(self.ready / 'a2')
        with self.assertRaises(FileExistsError):
            self.enrich()

    def test_wrong_audio_and_rehashed_quality_tampering_are_rejected(self):
        path = self.audio / 'a2.wav'
        original = path.read_bytes()
        path.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'differs from the original'):
            self.enrich()
        self.assertFalse(self.ready.exists())
        path.write_bytes(original)
        self.enrich()
        directory = self.ready / 'a2'
        _, tensors = read_fusion_package(directory)
        changed = tensors['word_quality.pt'].clone()
        changed[0, 0] = .99
        self.replace_tensor(directory, 'word_quality.pt', changed)
        with self.assertRaisesRegex(ValueError, 'disagree with original evidence'):
            read_fusion_package(directory)

    def test_fit_uses_training_fold_and_recordings_once(self):
        dataset = self.ready_dataset()
        for uid, values in (('a2', [10., .1, .2]), ('c2', [30., .3, .8]), ('a1', [60., 1., 1.])):
            self.replace_tensor(self.ready / uid, 'recording_quality.pt', torch.tensor(values))
        # More valid words in c2 must not increase its recording-level weight.
        self.replace_tensor(self.ready / 'c2', 'text_word_mask.pt', torch.tensor([True, True, True]))
        self.replace_tensor(self.ready / 'c2', 'paired_word_mask.pt', torch.tensor([True, False, True]))
        state = fit_quality_normalization(dataset)
        self.assertEqual(state['fit_uids'], ['a2', 'c2'])
        self.assertEqual(state['recording']['observations'], [2, 2, 2])
        np.testing.assert_allclose(state['recording']['mean'], [20, .2, .5], atol=1e-6)
        before = copy.deepcopy(state)
        self.replace_tensor(self.ready / 'a1', 'recording_quality.pt', torch.tensor([0., 0., 0.]))
        self.assertEqual(fit_quality_normalization(dataset), before)
        validation = make_precomputed_dataset(self.fixture.params, split='val', fold=0)
        with self.assertRaisesRegex(ValueError, 'only be fitted on ready training'):
            fit_quality_normalization(validation)
        another_fold = make_precomputed_dataset(self.fixture.params, split='train', fold=1)
        with self.assertRaisesRegex(ValueError, 'different training split'):
            validate_training_quality_state(another_fold, state)

    def test_missing_quality_indicators_and_batch_padding(self):
        dataset = self.ready_dataset()
        state = fit_quality_normalization(dataset)
        _, tensors = read_fusion_package(self.ready / 'a1')
        missing, zero = copy.deepcopy(tensors), copy.deepcopy(tensors)
        missing['word_quality.pt'][0, 0] = 0
        missing['word_quality_observed.pt'][0, 0] = False
        zero['word_quality.pt'][0, 0] = 0
        a = transform_quality(missing, state, dataset.schema_sha256)
        b = transform_quality(zero, state, dataset.schema_sha256)
        self.assertEqual(a[0, 0].item(), 0)
        self.assertEqual(a[0, 7].item(), 0)
        self.assertEqual(b[0, 7].item(), 1)
        self.assertTrue((a[~tensors['paired_word_mask.pt']] == 0).all())
        item = frame_item(self.ready / 'a1', 1, quality_state=state)
        short = copy.deepcopy(item)
        for key in ('word_ids', 'word_frame_intervals', 'text_words', 'audio_word_mask',
                    'text_word_mask', 'paired_word_mask', 'word_quality', 'word_quality_observed',
                    'reliability_features'):
            short[key] = short[key][:1]
        batch = collate_frame_items([item, short])
        self.assertEqual(tuple(batch['reliability_features'].shape), (2, 3, 14))
        self.assertTrue((batch['reliability_features'][1, 1:] == 0).all())

    def test_no_observed_training_feature_fails_explicitly(self):
        dataset = self.ready_dataset()
        for uid in ('a2', 'c2'):
            path = self.ready / uid / 'metadata.json'
            metadata = json.loads(path.read_text())
            metadata['quality_sources']['comparison_rows'][0]['whisper_probability'] = None
            path.write_text(json.dumps(metadata))
            q, values = word_quality(metadata)
            self.replace_tensor(self.ready / uid, 'word_quality.pt', q)
            self.replace_tensor(self.ready / uid, 'word_quality_observed.pt', values)
        with self.assertRaisesRegex(ValueError, 'No observed training values.*asr_confidence'):
            fit_quality_normalization(dataset)

    def test_training_checkpoint_and_evaluator_restore_normalization(self):
        self.ready_dataset()
        checkpoint_path, saved = self.fixture.run_training_epoch()
        state = saved['settings'].frame_quality_normalization
        self.assertEqual(state['fit_uids'], ['a2', 'c2'])
        self.assertEqual(state['recording']['observations'], [2] * 3)
        dataset = FrameEvaluationDataset(self.ready / 'a1', saved['settings'])
        item = dataset[0]
        self.assertEqual(tuple(item['reliability_features'].shape), (3, 14))
        output = self.root / 'quality-evaluation'
        evaluate(parse_args(['--checkpoint', str(checkpoint_path), '--dataset_to_evaluate', str(self.ready / 'a1'),
                             '--output_dir', str(output), '--device', 'cpu', '--threads', '1']))
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['checkpoint_settings']['frame_quality_normalization'], state)
        wrong = copy.deepcopy(saved['settings'])
        del wrong.frame_quality_normalization
        with self.assertRaisesRegex(ValueError, 'saved training normalization'):
            FrameEvaluationDataset(self.ready / 'a1', wrong)

    def test_normalization_receipts_stay_in_checkpoint_and_validate_resume(self):
        dataset = self.ready_dataset()
        state = fit_quality_normalization(dataset)
        trainer = Trainer.__new__(Trainer)
        trainer.params = self.fixture.params
        trainer.params.frame_quality_normalization = state
        trainer.total_trainable_params, trainer.gpus_count = 0, 0
        trainer.wandb_run = Mock()
        trainer.config_wandb()
        public = trainer.wandb_config['frame_quality_normalization']
        self.assertNotIn('fit_uids', public)
        self.assertNotIn('fit_metadata_sha256', public)
        self.assertEqual(trainer.params.frame_quality_normalization, state)
        self.assertEqual(public['training_recordings'], 2)
        path = self.ready / 'a2/metadata.json'
        metadata = json.loads(path.read_text())
        metadata['quality_extraction']['note'] = 'changed after fitting'
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'changed since quality normalization'):
            validate_training_quality_state(dataset, state)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
