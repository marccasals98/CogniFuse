"""Forced-alignment correctness and cache-preservation checks without models."""

import copy
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from utils.audio_frame_timing import frame_timing
from utils.ctc_forced_alignment import align_ctc_units, export_alignment
from utils.transcript_normalization import normalize_ctc_words


def fixture():
    vocabulary = {'|': 0, 'a': 1, 'b': 2, 'ñ': 3, '<pad>': 4}
    transcript = normalize_ctc_words(['aa', 'ñ'], vocabulary, 4)
    expected = [4, 1, 1, 4, 1, 0, 3, 3, 4]
    logits = torch.full((1, len(expected), len(vocabulary)), -6.0)
    for i, token in enumerate(expected):
        logits[0, i, token] = 6.0
    config = SimpleNamespace(conv_kernel=[10, 3, 3, 3, 3, 2, 2], conv_stride=[5, 2, 2, 2, 2, 2, 2])
    timing = frame_timing(config, (len(expected) - 1) * 320 + 400, 16000, len(expected))
    metadata = {
        'format': 'ctc_emissions_v1', 'logits_shape': list(logits.shape),
        'ctc_model': 'synthetic', 'audio_source': {}, 'word_source': {},
        'blank_id': 4, 'vocabulary': vocabulary, 'timing': timing,
    }
    return logits.log_softmax(-1), torch.ones(1, len(expected), dtype=torch.bool), transcript, metadata


class ForcedAlignmentTests(unittest.TestCase):
    def test_repeats_delimiters_and_nonzero_blank(self):
        emissions, mask, transcript, metadata = fixture()
        result, path, scores = align_ctc_units(emissions, mask, transcript, metadata)
        self.assertEqual(path.tolist(), [[4, 1, 1, 4, 1, 0, 3, 3, 4]])
        self.assertEqual([(u['start_frame'], u['end_frame_exclusive']) for u in result['units']],
                         [(1, 3), (4, 5), (5, 6), (6, 8)])
        self.assertEqual([u['word_id'] for u in result['units']], [0, 0, -1, 1])
        self.assertEqual(result['words'], ['aa', 'ñ'])
        torch.testing.assert_close(scores, emissions.gather(-1, path[..., None]).squeeze(-1))
        for unit in result['units']:
            self.assertTrue(0 < unit['emission_confidence'] <= 1)
            self.assertLessEqual(unit['end_time'], metadata['timing']['audio_duration_seconds'])
        self.assertAlmostEqual(result['units'][0]['start_time'], 0.02)
        self.assertAlmostEqual(result['units'][0]['end_time'], 0.065)

    def test_path_matches_exhaustive_optimum(self):
        emissions, mask, transcript, metadata = fixture()
        # Ambiguous emissions: compare the standard aligner with all 5^9 paths
        # reduced to three active symbols and five frames (only 3^5 paths).
        emissions = torch.tensor([[[0., 2., 1.], [1., 0., 2.], [0., 2., 1.],
                                   [1., 1., 2.], [0., 2., 1.]]]).log_softmax(-1)
        vocabulary = {'|': 0, 'a': 1, '<pad>': 2}
        transcript = normalize_ctc_words(['aa'], vocabulary, 2)
        metadata.update(vocabulary=vocabulary, blank_id=2, logits_shape=[1, 5, 3])
        config = SimpleNamespace(conv_kernel=[1], conv_stride=[1])
        metadata['timing'] = frame_timing(config, 5, 16000, 5)
        result, _, _ = align_ctc_units(emissions, torch.ones(1, 5, dtype=torch.bool), transcript, metadata)
        best = -float('inf')
        for candidate in itertools.product(range(3), repeat=5):
            collapsed = [token for token, _ in itertools.groupby(candidate) if token != 2]
            if collapsed == [1, 1]:
                best = max(best, sum(float(emissions[0, i, token]) for i, token in enumerate(candidate)))
        self.assertAlmostEqual(result['path_log_score'], best, places=5)

    def test_padding_is_excluded(self):
        emissions, mask, transcript, metadata = fixture()
        emissions = torch.cat((emissions, torch.full((1, 2, 5), float('nan'))), dim=1)
        mask = torch.cat((mask, torch.zeros(1, 2, dtype=torch.bool)), dim=1)
        metadata['logits_shape'] = list(emissions.shape)
        result, path, _ = align_ctc_units(emissions, mask, transcript, metadata)
        self.assertEqual(path.shape, (1, 9))
        self.assertTrue(all(unit['end_frame_exclusive'] <= 9 for unit in result['units']))

    def test_bad_probabilities_masks_and_timing_fail(self):
        for kind in ('logits', 'nan', 'mask_hole', 'empty_mask', 'timing'):
            with self.subTest(kind=kind):
                emissions, mask, transcript, metadata = fixture()
                if kind == 'logits':
                    emissions += 2
                elif kind == 'nan':
                    emissions[0, 0, 0] = float('nan')
                elif kind == 'mask_hole':
                    mask[0, 1] = False
                elif kind == 'empty_mask':
                    mask[:] = False
                else:
                    metadata['timing']['frame_stride_seconds'] = 0.1
                with self.assertRaises(ValueError):
                    align_ctc_units(emissions, mask, transcript, metadata)

    def test_corrupted_mapping_and_impossible_targets_fail(self):
        emissions, mask, transcript, metadata = fixture()
        corrupt = copy.deepcopy(transcript)
        corrupt['token_word_ids'][0] = 1
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            align_ctc_units(emissions, mask, corrupt, metadata)
        long_transcript = normalize_ctc_words(['aaaaaaaaaa'], metadata['vocabulary'], 4)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            align_ctc_units(emissions, mask, long_transcript, metadata)

    def test_export_preserves_inputs_and_refuses_overwrite(self):
        emissions, mask, transcript, metadata = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'ctc', root / 'alignment'
            source.mkdir()
            (source / 'metadata.json').write_text(json.dumps(metadata))
            (source / 'transcript.json').write_text(json.dumps(transcript))
            torch.save(emissions, source / 'ctc_log_probs.pt')
            torch.save(mask, source / 'valid_frame_mask.pt')
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            result = export_alignment(source, output)
            self.assertEqual(len(result['input_sha256']), 4)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertEqual(json.loads((output / 'alignment.json').read_text()), result)
            with self.assertRaises(FileExistsError):
                export_alignment(source, output)
            self.assertEqual(torch.load(output / 'path.pt', weights_only=True).shape, (1, 9))


if __name__ == '__main__':
    unittest.main()
