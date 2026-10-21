"""Exercise checkpoint persistence/reloading and the real sequential CV command."""
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from train import ArgsParser, Trainer
import cross_validation as cv


class Recordings:
    label_names = ['lvPPA', 'nfPPA', 'svPPA']

    def __init__(self, prefix):
        self.segments = [{'uid': f'{prefix}{i}', 'patient_id': f'{prefix}{i}', 'label': i}
                         for i in range(3)]
        self.df = pd.DataFrame(self.segments)

    def __len__(self):
        return len(self.segments)


class Generator:
    def __init__(self, prefix):
        self.dataset = Recordings(prefix)

    def __len__(self):
        return 3


class TinyTrainer(Trainer):
    """Use real checkpoint I/O with deterministic, lightweight classifier weights."""
    def __init__(self, root, interval=10):
        self.params = SimpleNamespace(
            model_output_folder=str(root), model_name='tiny',
            fold_results_path=str(root / 'fold.json'), fold=0, num_folds=2,
            random_seed=1234, torch_seed=3407, max_epochs=2,
            eval_and_save_best_model_every=interval, max_overfitting_allowed=None,
            load_checkpoint=False,
        )
        self.is_distributed = False
        self.is_main_process = True
        self.training_generator = Generator('train')
        self.evaluating_generator = Generator('val')
        self.net = torch.nn.Linear(3, 3, bias=False)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=.1)
        self.wandb_run = None
        self.start_datetime = 'fixture'
        self.total_trainable_params = 9
        self.initialize_training_variables()
        self.epoch = 0
        self.batch_number = 0
        self.train_loss = 1.0
        self.weights(1)

    def weights(self, sign):
        with torch.no_grad():
            self.net.weight.copy_(sign * torch.eye(3))

    def evaluate(self):
        with torch.no_grad():
            self.validation_predictions = self.net(torch.eye(3)).argmax(1).tolist()
        self.validation_labels = [0, 1, 2]
        self.training_eval_metric = self.validation_eval_metric = float(f1_score(
            self.validation_labels, self.validation_predictions,
            labels=[0, 1, 2], average='macro', zero_division=0))
        self.has_evaluation = True


class BestCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir='/tmp', prefix='best-cv-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trainer = TinyTrainer(self.root)

    def result(self):
        return json.loads((self.root / 'fold.json').read_text())

    def test_reload_earlier_best_and_log_terminal_score_separately(self):
        t = self.trainer
        t.step = 9
        t.eval_and_save_best_model()
        t.weights(-1)
        t.epoch = 1
        t.step = 20
        t.train = Mock()
        t.wandb_run = Mock()
        t.wandb_run.summary = {}
        t.wandb_run.settings.mode = 'offline'
        with patch('train.wandb.finish'):
            t.main()
        r = self.result()
        self.assertEqual(r['metrics']['macro_f1'], 1.)
        self.assertEqual(r['final_metrics']['validation_macro_f1'], 0.)
        self.assertEqual(r['selected_epoch'], 1)
        self.assertEqual(r['selected_step'], 10)
        self.assertEqual(r['protocol'], 'best_validation_checkpoint')
        self.assertEqual([p['prediction'] for p in r['predictions']], [0, 1, 2])
        torch.testing.assert_close(t.net.weight, torch.eye(3))
        logged = t.wandb_run.log.call_args.args[0]
        self.assertEqual(logged['cv/terminal_epoch_validation_macro_f1'], 0.)
        self.assertNotIn('cv/final_validation_macro_f1', logged)
        self.assertEqual(logged['cv/selected_validation_macro_f1'], 1.)
        for name, value in r['metrics']['per_class_f1'].items():
            self.assertEqual(logged[f'cv/selected_validation_f1/{name}'], value)
            self.assertEqual(value, 1.)
        self.assertEqual(t.wandb_run.summary, logged)
        checkpoint = torch.load(r['checkpoint'], weights_only=False)
        self.assertEqual(checkpoint['selection_protocol'], r['protocol'])
        self.assertEqual(checkpoint['training_variables']['best_model_step'], 10)

    def test_exact_update_interval_and_ties_retain_earlier_checkpoint(self):
        t = self.trainer
        t.step = 8
        t.eval_and_save_best_model()
        self.assertFalse((self.root / 'tiny/tiny.chkpt').exists())
        t.step = 9
        t.eval_and_save_best_model()
        t.step = 19
        t.epoch = 1
        t.eval_and_save_best_model()
        t.step = 20
        t.finish_cv_fold()
        self.assertEqual(self.result()['selected_step'], 10)
        self.assertEqual(self.result()['selected_epoch'], 1)

    def test_initial_zero_score_is_saved_and_reloaded(self):
        t = self.trainer
        t.weights(-1)
        t.step = 9
        t.eval_and_save_best_model()
        t.step = 15
        t.finish_cv_fold()
        self.assertEqual(self.result()['metrics']['macro_f1'], 0.)
        self.assertEqual(self.result()['selected_step'], 10)

    def test_terminal_model_can_improve_between_evaluations(self):
        t = self.trainer
        t.weights(-1)
        t.step = 9
        t.eval_and_save_best_model()
        t.weights(1)
        t.step = 15
        t.epoch = 1
        t.finish_cv_fold()
        self.assertEqual(self.result()['selected_step'], 15)
        self.assertEqual(self.result()['metrics']['macro_f1'], 1.)

    def test_short_budget_and_explicit_final_only_mode(self):
        for interval in (100, 0):
            with self.subTest(interval=interval):
                t = TinyTrainer(self.root / str(interval), interval)
                t.step = 2
                t.finish_cv_fold()
                r = json.loads(Path(t.params.fold_results_path).read_text())
                self.assertEqual(r['selected_step'], 2)
                self.assertEqual(r['evaluation_interval_steps'], interval)
                self.assertEqual(r['protocol'], 'best_validation_checkpoint' if interval else 'fixed_epochs_final_model')
                self.assertEqual(r['metrics']['macro_f1'], 1.)

    def test_no_eligible_checkpoint_fails_explicitly(self):
        t = self.trainer
        t.params.max_overfitting_allowed = -.1
        with self.assertRaisesRegex(RuntimeError, 'No eligible'):
            t.finish_cv_fold()

    def test_training_seed_defaults_to_3407_independently_of_fold_seed(self):
        parser = ArgsParser()
        parser.main([])
        p = parser.arguments
        self.assertEqual((p.random_seed, p.torch_seed), (1234, 3407))
        with patch('train.torch.cuda.manual_seed') as cuda_seed, patch('train.torch.cuda.manual_seed_all') as cuda_seed_all:
            self.trainer.set_random_seed(p)
            cuda_seed.assert_called_with(3407)
            cuda_seed_all.assert_called_with(3407)
        actual = torch.rand(3)
        self.assertEqual(random.random(), random.Random(1234).random())
        self.assertEqual(np.random.rand(), np.random.RandomState(1234).rand())
        torch.manual_seed(3407)
        torch.testing.assert_close(actual, torch.rand(3))
        parser = ArgsParser()
        parser.main(['--random_seed', '42'])
        self.assertEqual(parser.arguments.torch_seed, 3407)
        legacy = SimpleNamespace(random_seed=42)
        self.trainer.set_random_seed(legacy)
        self.assertEqual(legacy.torch_seed, 3407)

    def test_explicit_torch_seed_overrides_default_without_changing_fold_seed(self):
        parser = ArgsParser()
        parser.main(['--torch_seed', '1234'])
        params = parser.arguments
        self.trainer.set_random_seed(params)
        actual = torch.rand(3)
        torch.manual_seed(1234)
        torch.testing.assert_close(actual, torch.rand(3))
        self.assertEqual(params.random_seed, 1234)

    def test_summary_rejects_mixed_protocols_intervals_and_seeds(self):
        t = self.trainer
        t.step = 2
        t.finish_cv_fold()
        a = self.result()
        b = json.loads(json.dumps(a))
        b.update(fold=1, training_patients=a['validation_patients'], validation_patients=a['training_patients'])
        b['predictions'] = [{**p, 'uid': f"other{i}", 'patient_id': a['training_patients'][i]}
                            for i, p in enumerate(b['predictions'])]
        self.assertEqual(cv.summarize_results([a, b])['mean_macro_f1'], 1.)
        for key, value in [('protocol', 'fixed_epochs_final_model'), ('evaluation_interval_steps', 0), ('torch_seed', 1234)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                cv.summarize_results([a, {**b, key: value}])
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            cv.selection_protocol(-1)

    def test_wandb_experiment_summary_matches_cv_statistics(self):
        results = [
            {'fold': 0, 'training_macro_f1': .85,
             'metrics': {'macro_f1': .6, 'per_class_f1': {'lvPPA': .4, 'nfPPA': .6, 'svPPA': .8}}},
            {'fold': 1, 'training_macro_f1': .95,
             'metrics': {'macro_f1': .8, 'per_class_f1': {'lvPPA': .7, 'nfPPA': .8, 'svPPA': .9}}},
        ]
        summary = {'mean_macro_f1': .7, 'std_macro_f1': float(np.std([.6, .8], ddof=1)),
                   'num_folds': 2, 'out_of_fold': {'macro_f1': .72,
                       'per_class_f1': {'lvPPA': .6, 'nfPPA': .72, 'svPPA': .84}},
                   'protocol': 'best_validation_checkpoint'}
        params = SimpleNamespace(**vars(self.trainer.params), use_weights_and_biases=True,
                                 wandb_dir=str(self.root), train_labels_path='private/patients.csv')
        fake_run = Mock(id='summary-id', url='https://wandb.ai/test/project/runs/summary-id')
        fake_run.summary = {}
        with patch('cross_validation.wandb.init') as initialize:
            initialize.return_value.__enter__.return_value = fake_run
            cv.log_cv_summary(params, summary, results, self.root, 'CV-fixture')
        metrics = fake_run.log.call_args.args[0]
        self.assertEqual(metrics['cv/final_validation_macro_f1'], summary['mean_macro_f1'])
        self.assertEqual(metrics['cv/std_validation_macro_f1'], summary['std_macro_f1'])
        self.assertEqual(metrics['cv/completed_folds'], 2)
        self.assertEqual(metrics['cv/pooled_macro_f1'], .72)
        for name, value in summary['out_of_fold']['per_class_f1'].items():
            self.assertEqual(metrics[f'cv/pooled_f1/{name}'], value)
        for result in results:
            prefix = f"cv/fold_{result['fold']}"
            self.assertEqual(metrics[f'{prefix}_macro_f1'], result['metrics']['macro_f1'])
            for name, value in result['metrics']['per_class_f1'].items():
                self.assertEqual(metrics[f'{prefix}_f1/{name}'], value)
        self.assertAlmostEqual(metrics['cv/final_training_macro_f1'], .9)
        self.assertEqual(fake_run.summary, metrics)
        options = initialize.call_args.kwargs
        self.assertEqual(options['group'], 'CV-fixture')
        self.assertEqual(options['job_type'], 'cross_validation')
        self.assertNotIn('train_labels_path', options['config'])
        saved = json.loads((self.root / 'wandb_cv_summary.json').read_text())
        self.assertEqual(saved['metrics'], metrics)
        self.assertEqual(saved['url'], fake_run.url)

    def test_disabled_wandb_creates_no_experiment_run(self):
        with patch('cross_validation.wandb.init') as initialize:
            cv.log_cv_summary(SimpleNamespace(use_weights_and_biases=False), {}, [], self.root, 'CV-test')
        initialize.assert_not_called()
        self.assertFalse((self.root / 'wandb_cv_summary.json').exists())

    def test_fold_wandb_runs_share_the_cv_group(self):
        with patch.dict(os.environ, {'COGNIFUSE_CV_GROUP': 'CV-fixture'}), \
                patch('train.wandb.init') as initialize, patch('train.wandb.run', Mock()):
            self.trainer.init_wandb(SimpleNamespace(wandb_dir=str(self.root)))
        self.assertEqual(initialize.call_args.kwargs['group'], 'CV-fixture')
        self.assertEqual(initialize.call_args.kwargs['job_type'], 'cv_fold')

    def test_real_two_fold_training_and_summary(self):
        features = self.root / 'features'
        features.mkdir()
        labels = self.root / 'labels.csv'
        rows = ['filename,patient_id,diagnosis']
        for label, name in enumerate(Recordings.label_names):
            for i in range(2):
                uid = f'r{label}{i}'
                rows.append(f'{uid}.wav,{uid},{name}')
                for suffix in ('distil', 'distil_audio'):
                    torch.save(torch.arange(12).float().reshape(3, 4) + label,
                               features / f'{uid}{suffix}.pt')
                    torch.save(torch.ones(3, dtype=torch.bool), features / f'{uid}{suffix}_mask.pt')
        labels.write_text('\n'.join(rows) + '\n')
        output = self.root / 'cv'
        command = [sys.executable, str(SCRIPTS / 'train.py'), '--cross_validate',
                   '--precomputed_features_dir', str(features), '--train_labels_path', str(labels),
                   '--validation_labels_path', str(labels), '--num_folds', '2', '--max_epochs', '2',
                   '--training_batch_size', '2', '--evaluation_batch_size', '1',
                   '--speech_adapter', 'NoneAdapter', '--text_adapter', 'NoneAdapter',
                   '--seq_to_seq_method', 'NoneSeqToSeq', '--seq_to_one_method', 'StatisticalPooling',
                   '--classifier_hidden_layers', '0', '--classifier_hidden_layers_width', '4',
                   '--random_seed', '1234', '--torch_seed', '3407',
                   '--eval_and_save_best_model_every', '2', '--no-use_weights_and_biases',
                   '--cross_validation_output_dir', str(output),
                   '--model_output_folder', str(self.root / 'models'),
                   '--log_file_folder', str(self.root / 'logs')]
        env = {k: v for k, v in os.environ.items() if k not in {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'} and not k.startswith('TORCHELASTIC_')}
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', TMPDIR='/tmp')
        completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stdout[-2500:] + completed.stderr[-4500:])
        config = json.loads((output / 'config.json').read_text())
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(config['eval_and_save_best_model_every'], 2)
        self.assertEqual(summary['protocol'], 'best_validation_checkpoint')
        self.assertEqual(summary['torch_seed'], 3407)
        folds = [json.loads((output / f'fold_{i}.json').read_text()) for i in range(2)]
        vals = []
        for fold in folds:
            self.assertEqual(fold['evaluation_interval_steps'], 2)
            self.assertTrue(Path(fold['checkpoint']).is_file())
            self.assertFalse(set(fold['training_patients']) & set(fold['validation_patients']))
            vals.append(fold['metrics']['macro_f1'])
            self.assertAlmostEqual(vals[-1], f1_score([p['label'] for p in fold['predictions']],
                [p['prediction'] for p in fold['predictions']], labels=[0, 1, 2], average='macro', zero_division=0))
        self.assertAlmostEqual(summary['mean_macro_f1'], np.mean(vals))
        self.assertAlmostEqual(summary['std_macro_f1'], np.std(vals, ddof=1))


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    unittest.main()
