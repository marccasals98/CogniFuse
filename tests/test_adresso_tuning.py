"""Guard the sweep's completion and strict-checkpoint audit requirements."""

import json
from pathlib import Path
import tempfile
import unittest

from utils.tune_adresso import audit_checkpoint, summarize, trial_grid


class TuningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir='/tmp', prefix='adresso-tuning-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def journal(self, decisions):
        checkpoint = self.root / 'model.chkpt'
        checkpoint.write_bytes(b'checkpoint fixture')
        (self.root / 'checkpoint_decisions.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in decisions))
        return {'checkpoint': str(checkpoint), 'epochs': 60, 'training_recordings': 132,
                'selected_step': 100, 'metrics': {'macro_f1': 0.8}}

    def test_grid_covers_each_requested_combination_once(self):
        trials = trial_grid()
        combinations = {(t['seq_to_seq_heads_number'], t['seq_to_seq_input_dropout'],
                         t['classifier_hidden_layers']) for t in trials}
        self.assertEqual(len(combinations), 27)
        self.assertEqual(len({t['id'] for t in trials}), 27)
        self.assertEqual({t['seq_to_seq_heads_number'] for t in trials}, {2, 4, 8})
        self.assertEqual({t['seq_to_seq_input_dropout'] for t in trials}, {0., .1, .3})
        self.assertEqual({t['classifier_hidden_layers'] for t in trials}, {1, 2, 4})

    def test_audit_accepts_initial_save_and_skips_ties_and_decreases(self):
        decisions = [
            {'completed_steps': 100, 'validation_macro_f1': .8,
             'previous_best_validation_macro_f1': None, 'eligible': True, 'saved': True},
            {'completed_steps': 200, 'validation_macro_f1': .8,
             'previous_best_validation_macro_f1': .8, 'eligible': True, 'saved': False},
            {'completed_steps': 300, 'validation_macro_f1': .7,
             'previous_best_validation_macro_f1': .8, 'eligible': True, 'saved': False},
        ]
        fold = self.journal(decisions)
        audit_checkpoint(fold)
        for index in (1, 2):
            broken = [dict(row) for row in decisions]
            broken[index]['saved'] = True
            self.journal(broken)
            with self.assertRaisesRegex(ValueError, 'strict improvement'):
                audit_checkpoint(fold)

    def test_audit_rejects_selected_step_or_schedule_mismatch(self):
        decisions = [{'completed_steps': 100, 'validation_macro_f1': .8,
                      'previous_best_validation_macro_f1': None,
                      'eligible': True, 'saved': True}]
        fold = self.journal(decisions)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            audit_checkpoint({**fold, 'selected_step': 200})
        decisions[0]['completed_steps'] = 99
        self.journal(decisions)
        with self.assertRaisesRegex(ValueError, 'schedule'):
            audit_checkpoint(fold)

    def test_incomplete_sweep_cannot_claim_a_final_winner(self):
        (self.root / 'baseline').mkdir()
        (self.root / 'manifest.json').write_text(json.dumps({
            'trials': trial_grid(), 'baseline_results_directory': 'baseline'}))
        (self.root / 'baseline/summary.json').write_text(json.dumps({
            'mean_macro_f1': .8, 'std_macro_f1': .1}))
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            summarize(self.root, require_complete=True)
        report = json.loads((self.root / 'leaderboard.json').read_text())
        self.assertEqual(report['completed_trials'], 0)
        self.assertEqual(report['status'], 'incomplete')
        self.assertIsNone(report['best_completed_trial'])

    def test_complete_sweep_writes_winning_parameters_and_compares_baseline(self):
        trial = trial_grid()[0]
        parameters = {k: v for k, v in trial.items() if k not in ('id', 'index')}
        fixed = {'num_folds': 5, 'max_epochs': 60, 'eval_and_save_best_model_every': 100}
        baseline = self.root / 'baseline'
        baseline.mkdir()
        (baseline / 'splits.json').write_text('[]')
        (baseline / 'config.json').write_text(json.dumps({**fixed, **parameters}))
        (self.root / 'manifest.json').write_text(json.dumps({
            'trials': [trial], 'fixed_parameters': fixed,
            'baseline_results_directory': str(baseline)}))
        results = self.root / 'trials' / trial['id'] / 'results'
        results.mkdir(parents=True)
        (results / 'config.json').write_text(json.dumps({**fixed, **parameters}))
        (results / 'splits.json').write_text('[]')
        summary = {'num_folds': 5, 'recordings': 166, 'mean_macro_f1': .9,
                   'std_macro_f1': 0., 'fold_macro_f1': [.9] * 5,
                   'out_of_fold': {'macro_f1': .9}}
        (results / 'summary.json').write_text(json.dumps(summary))
        fold = self.journal([{'completed_steps': 100, 'validation_macro_f1': .9,
                             'previous_best_validation_macro_f1': None,
                             'eligible': True, 'saved': True}])
        fold.update(metrics={'macro_f1': .9}, selected_epoch=1)
        for index in range(5):
            (results / f'fold_{index}.json').write_text(json.dumps(fold))
        for baseline_score, expected_winner in ((.8, trial['id']), (.95, 'original_baseline')):
            with self.subTest(baseline_score=baseline_score):
                (baseline / 'summary.json').write_text(json.dumps({
                    'mean_macro_f1': baseline_score, 'std_macro_f1': .1}))
                report = summarize(self.root, require_complete=True)
                self.assertEqual(report['status'], 'complete')
                winner = json.loads((self.root / 'best_parameters.json').read_text())
                self.assertEqual(winner['winner'], expected_winner)
                self.assertEqual(winner['parameters'], {**fixed, **parameters})
                self.assertEqual(winner['mean_macro_f1'], max(.9, baseline_score))

        # A repeat using different input representations must pick its own best
        # trial even if the historical token-feature reference scored higher.
        manifest = json.loads((self.root / 'manifest.json').read_text())
        manifest['baseline_is_matched'] = False
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        summarize(self.root, require_complete=True)
        winner = json.loads((self.root / 'best_parameters.json').read_text())
        self.assertEqual(winner['winner'], trial['id'])
        self.assertEqual(winner['mean_macro_f1'], .9)


if __name__ == '__main__':
    unittest.main()
