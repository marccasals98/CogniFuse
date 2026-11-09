"""Verify recording joins, ensemble decisions and missing-result reporting."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.evaluate_adresso_selected import ensemble_predictions, summarize_ensemble
from utils.adresso_test_report import TEST_HEADERS, test_columns


class ADReSSoEnsembleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir='/tmp', prefix='adresso-ensemble-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.directories = []
        for fold in range(5):
            folder = self.root / str(fold)
            folder.mkdir()
            (folder / 'manifest.json').write_text('{}')
            self.write_predictions(folder, fold)
            self.directories.append(folder)

    def write_predictions(self, folder, fold, duplicate=False, wrong_label=False):
        # Three weak voters prefer control, two confident voters prefer AD.
        p = .49 if fold < 3 else .99
        rows = [('a', 1, 1 - p, p), ('b', 0, .9, .1)]
        if fold % 2:
            rows.reverse()
        if duplicate:
            rows.append(rows[0])
        if wrong_label:
            rows = [(uid, 0, p0, p1) for uid, _, p0, p1 in rows]
        with (folder / 'predictions.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['uid', 'label', 'probability_0', 'probability_1'])
            writer.writerows(rows)

    def test_probability_average_matches_ids_and_differs_from_majority_vote(self):
        truth, probabilities = ensemble_predictions(self.directories, ['a', 'b'])
        self.assertEqual(truth, [1, 0])
        np.testing.assert_allclose(probabilities, [[.31, .69], [.9, .1]])
        metrics = summarize_ensemble(self.directories, ['a', 'b'], self.root / 'ensemble', samples=100)
        self.assertEqual(metrics['macro_f1'], 1)
        self.assertEqual(metrics['confusion_matrix'], [[1, 0], [0, 1]])
        with self.assertRaises(FileExistsError):
            summarize_ensemble(self.directories, ['a', 'b'], self.root / 'ensemble', samples=100)

    def test_duplicate_missing_and_incomplete_folds_fail(self):
        with self.assertRaises(ValueError):
            ensemble_predictions(self.directories[:4], ['a', 'b'])
        with self.assertRaises(ValueError):
            ensemble_predictions(self.directories, ['a', 'b', 'c'])
        self.write_predictions(self.directories[0], 0, duplicate=True)
        with self.assertRaises(ValueError):
            ensemble_predictions(self.directories, ['a', 'b'])

    def test_conflicting_labels_and_invalid_probabilities_fail(self):
        self.write_predictions(self.directories[0], 0, wrong_label=True)
        with self.assertRaisesRegex(ValueError, 'Ground truth'):
            ensemble_predictions(self.directories, ['a', 'b'])
        self.write_predictions(self.directories[0], 0)
        path = self.directories[0] / 'predictions.csv'
        path.write_text(path.read_text().replace('0.49', 'nan'))
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            ensemble_predictions(self.directories, ['a', 'b'])

    def test_reporting_never_fabricates_missing_scores_and_matches_exact_source(self):
        result = self.root / 'development/results'
        missing = test_columns(result, self.root)
        self.assertEqual(missing, ['Not included'] + [''] * (len(TEST_HEADERS) - 1))
        metrics = summarize_ensemble(self.directories, ['a', 'b'], self.root / 'ensemble', samples=100)
        system = {'training_results': str(result), 'ensemble_metrics': metrics,
                  'ensemble_directory': str(self.root / 'ensemble'),
                  'fold_macro_f1_mean': .8, 'fold_macro_f1_sd': .1}
        (self.root / 'summary.json').write_text(json.dumps({'status': 'complete', 'systems': [system]}))
        cells = test_columns(result, self.root)
        self.assertEqual(cells[0], 'Complete')
        self.assertEqual(cells[2:5], [2, 1, 1])
        self.assertEqual(test_columns(self.root / 'other/results', self.root), missing)


if __name__ == '__main__':
    unittest.main()
