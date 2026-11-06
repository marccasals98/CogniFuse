"""Scientific controls for the ADReSSo paper adaptation."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from utils.adresso_paper import (Classifier, PaperModel, WordDataset,
                                checkpoint_candidate, experiment_matrix)


SETTINGS = dict(precomputed_features_dir='unused',
    speech_feature_extractor_output_vectors_dimension=8,
    text_feature_extractor_output_vectors_dimension=8,
    speech_adapter='NoneAdapter', text_adapter='NoneAdapter',
    seq_to_seq_method='MultiHeadAttention', seq_to_seq_heads_number=8,
    skip_connections=False, seq_to_seq_input_dropout=.3,
    seq_to_one_input_dropout=.3, seq_to_one_method='AttentionPooling',
    classifier_layer_drop_out=.3, classifier_hidden_layers=2,
    classifier_hidden_layers_width=16, number_classes=2)


class PaperControls(unittest.TestCase):
    def test_baseline_has_identical_production_weights_and_predictions(self):
        torch.manual_seed(3407)
        production = Classifier(SimpleNamespace(**SETTINGS), 'cpu').eval()
        torch.manual_seed(3407)
        baseline = PaperModel(experiment_matrix()[0], SETTINGS).eval()
        self.assertEqual(list(production.state_dict()), list(baseline.state_dict()))
        for key, tensor in production.state_dict().items():
            self.assertTrue(torch.equal(tensor, baseline.state_dict()[key]))
        audio, text = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        with torch.no_grad():
            expected = production(audio, text, torch.stack([mask, mask], 1))
            actual = baseline(audio, text, mask, mask, torch.zeros(2, 4, 3), torch.ones(2, 4))
        self.assertTrue(torch.equal(expected, actual))

    def test_alignment_control_does_not_change_other_ablations_masks(self):
        data = dict(audio={'ctc_0': torch.ones(3, 8)}, text={'distilbert': torch.ones(3, 8)},
                    mask=torch.tensor([1, 1, 1], dtype=torch.bool),
                    alignment_mask=torch.tensor([1, 0, 1], dtype=torch.bool),
                    timing=torch.zeros(3, 3), confidence=torch.ones(3))
        records = [{'uid': 'sample', 'label': 0}]
        matrix = experiment_matrix()
        for name, expected in [('global', data['mask']), ('alignment_control', data['alignment_mask'])]:
            config = next(c for c in matrix if c['name'] == name)
            sample = WordDataset(records, {'sample': data}, config, '.')[0]
            self.assertTrue(torch.equal(sample[2], expected))
            self.assertTrue(torch.equal(sample[3], expected))
        self.assertEqual(len(matrix), 26)
        self.assertTrue(all(c['encoder'] == 'distilbert' for c in matrix))

    def test_encoder_repeat_changes_only_the_encoder_in_the_intervention_matrix(self):
        original = experiment_matrix()
        repeated = experiment_matrix('deberta_v3_large')
        self.assertEqual(repeated, [dict(c, encoder='deberta_v3_large') for c in original])
        settings = {**SETTINGS, 'classifier_hidden_layers': 1}
        torch.manual_seed(3407)
        expected = Classifier(SimpleNamespace(**settings), 'cpu').eval()
        torch.manual_seed(3407)
        actual = PaperModel(repeated[0], settings).eval()
        for key, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[key]))
        mask = torch.ones(1, 4, dtype=torch.bool)
        audio, text = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        with torch.no_grad():
            self.assertTrue(torch.equal(expected(audio, text, torch.stack([mask, mask], 1)),
                actual(audio, text, mask, mask, torch.zeros(1, 4, 3), torch.ones(1, 4))))

    def test_validation_ties_and_regressions_do_not_rewrite_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, journal = Path(directory) / 'best.pt', Path(directory) / 'journal.jsonl'
            model = torch.nn.Linear(2, 2)
            best = checkpoint_candidate(checkpoint, journal, model, .8, None, 100, 0)
            original, mtime = checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns
            for step, score in [(200, .8), (300, .7)]:
                best = checkpoint_candidate(checkpoint, journal, model, score, best, step, 1)
                self.assertEqual(checkpoint.read_bytes(), original)
                self.assertEqual(checkpoint.stat().st_mtime_ns, mtime)
            self.assertEqual(checkpoint_candidate(checkpoint, journal, model, .9, best, 400, 2), .9)


if __name__ == '__main__':
    unittest.main()
