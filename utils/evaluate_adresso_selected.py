"""Evaluate a frozen, validation-selected ADReSSo shortlist and its CV ensembles.

Run `prepare` before test inference to record the model choices and source hashes.
No fitting, architecture ranking, threshold tuning, or checkpoint selection uses test scores.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.adresso import ADReSSoIndex
from scripts.evaluate import bootstrap_macro_f1, evaluate, file_identity, parse_args as evaluation_args
from utils.adresso_test_report import DEFAULT_ROOT

CLASSES = ['cn', 'ad']
LABELS = REPO / 'outputs/adresso_evaluation/official_labels/test_results_task1_groundtruth.csv'
DATASET = Path('/home/usuaris/veussd/marc.casals/datasets/ADReSSo21')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(root):
    old = REPO / 'outputs/adresso_architecture_sweep_20261029'
    new = REPO / 'outputs/adresso_architecture_sweep_words_20261029'
    old_choice, new_choice = read(old / 'best_parameters.json'), read(new / 'best_parameters.json')
    for choice in (old_choice, new_choice):
        if choice['official_test_used'] or 'validation' not in choice['selection_metric']:
            raise ValueError('Shortlist must be selected only by development validation')
    originals = REPO / 'outputs/adresso_full_whisper_distil_pauses'
    words = REPO / 'outputs/adresso_ctc_distilbert_words_20261029'
    definitions = [
        ('baseline', 'B0 — original baseline', 'Original token features',
         REPO / 'outputs/adresso_baseline_cv/run_2698224/results', originals, None),
        ('original_winner', old_choice['winner'], 'Original token features',
         old / 'trials' / old_choice['winner'] / 'results', originals, old / 'best_parameters.json'),
        ('word_winner', new_choice['winner'], 'CTC word features',
         new / 'trials' / new_choice['winner'] / 'results', words, new / 'best_parameters.json'),
    ]
    index = ADReSSoIndex(DATASET)
    index.development_folds(5)
    test_ids = sorted(r['uid'] for r in index.test)
    systems = []
    for identity, name, representation, results, features, selection in definitions:
        summary = read(results / 'summary.json')
        if summary['num_folds'] != 5 or summary['recordings'] != len(index.training):
            raise ValueError('Expected complete five-fold development results')
        config = read(results / 'config.json')
        if Path(config['precomputed_features_dir']).resolve() != features / 'train/embeddings':
            raise ValueError('Training and test representations differ')
        folds = []
        for fold in range(5):
            source = results / f'fold_{fold}.json'
            record = read(source)
            checkpoint = Path(record['checkpoint'])
            if record['fold'] != fold or record['protocol'] != 'best_validation_checkpoint':
                raise ValueError('Expected explicitly validation-selected fold checkpoints')
            folds.append({'fold': fold, 'checkpoint': file_identity(checkpoint),
                          'development_result': file_identity(source)})
        systems.append({'id': identity, 'name': name, 'representation': representation,
                        'training_results': str(results), 'test_features': str(features / 'test/embeddings'),
                        'validation_macro_f1': summary['mean_macro_f1'], 'folds': folds,
                        'selection_source': file_identity(selection or results / 'config.json')})
    root.mkdir(parents=True, exist_ok=False)
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'selection': 'Original baseline and the validation-selected winner of each completed search',
            'ensemble_policy': 'Unweighted arithmetic mean of five checkpoint probability vectors; argmax, no threshold tuning',
            'bootstrap_samples': 10000, 'bootstrap_seed': 1234,
            'dataset_root': str(index.root), 'test_ids': test_ids, 'labels': file_identity(LABELS),
            'systems': systems}
    write(root / 'plan.json', plan)
    print(f'Frozen {len(systems)} systems / 15 checkpoints before test inference: {root}', flush=True)


def ensemble_predictions(directories, expected_ids):
    """Match probability rows by recording ID, checking labels and normalisation."""
    arrays, truth = [], None
    if len(directories) != 5:
        raise ValueError('A CV ensemble requires all five folds')
    for directory in directories:
        with (Path(directory) / 'predictions.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        by_uid = {row['uid']: row for row in rows}
        if len(by_uid) != len(rows) or set(by_uid) != set(expected_ids):
            raise ValueError('Fold predictions must cover each official test recording exactly once')
        labels = [int(by_uid[uid]['label']) for uid in expected_ids]
        probabilities = np.array([[float(by_uid[uid][f'probability_{c}']) for c in range(2)]
                                  for uid in expected_ids])
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or (probabilities > 1).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-6)):
            raise ValueError('Invalid checkpoint probabilities')
        if truth is not None and labels != truth:
            raise ValueError('Ground truth differs between folds')
        if not set(labels) <= {0, 1}:
            raise ValueError('Invalid diagnosis labels')
        truth = labels
        arrays.append(probabilities)
    return truth, np.mean(arrays, axis=0)


def summarize_ensemble(directories, expected_ids, destination, samples=10000, seed=1234):
    truth, probabilities = ensemble_predictions(directories, expected_ids)
    predictions = probabilities.argmax(1).tolist()
    interval, distribution = bootstrap_macro_f1(truth, predictions, 2, samples=samples, seed=seed)
    metrics = {'status': 'evaluated', 'recordings': len(expected_ids), 'class_names': CLASSES,
               'label_map': {'cn': 0, 'ad': 1},
               'macro_f1': float(f1_score(truth, predictions, labels=[0, 1], average='macro', zero_division=0)),
               'accuracy': float(accuracy_score(truth, predictions)),
               'per_class_f1': dict(zip(CLASSES, f1_score(truth, predictions, labels=[0, 1], average=None, zero_division=0).tolist())),
               'confusion_matrix': confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
               'confusion_matrix_axes': 'rows=true, columns=predicted; class_names order',
               'macro_f1_bootstrap': interval}
    destination.mkdir(exist_ok=False)
    write(destination / 'metrics.json', metrics)
    write(destination / 'manifest.json', {
        'policy': 'Mean of probability vectors of all five validation-selected checkpoints, then argmax',
        'sources': [file_identity(Path(d) / name) for d in directories for name in ('manifest.json', 'predictions.csv')],
        'test_used_for_selection': False,
        'uncertainty_scope': 'Recording bootstrap conditional on the fixed five-model ensemble; excludes training variability'})
    with (destination / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['uid', 'prediction', 'predicted_class', 'probability_0', 'probability_1', 'label', 'true_class'])
        for uid, prediction, probability, label in zip(expected_ids, predictions, probabilities, truth):
            writer.writerow([uid, prediction, CLASSES[prediction], *probability, label, CLASSES[label]])
    with (destination / 'bootstrap_scores.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['replicate', 'macro_f1'])
        writer.writerows(enumerate(distribution))
    return metrics


def run(root, device, threads):
    plan = read(root / 'plan.json')
    if (root / 'summary.json').exists():
        raise FileExistsError('Evaluation already complete; existing scores are immutable')
    if file_identity(plan['labels']['path']) != plan['labels']:
        raise ValueError('Labels changed after the shortlist was frozen')
    completed = []
    for system in plan['systems']:
        folder = root / system['id']
        folder.mkdir(exist_ok=True)
        directories, folds = [], []
        for fold in system['folds']:
            checkpoint = fold['checkpoint']
            if file_identity(checkpoint['path']) != checkpoint:
                raise ValueError('Checkpoint changed after the shortlist was frozen')
            destination = folder / f"fold_{fold['fold']}"
            if destination.exists():
                manifest = read(destination / 'manifest.json')
                if (manifest['checkpoint'] != checkpoint or manifest['labels'] != plan['labels']
                        or manifest['dataset_to_evaluate'] != system['test_features']
                        or manifest['recording_ids'] != plan['test_ids']):
                    raise ValueError('Existing fold evaluation differs from the frozen plan')
                metrics = read(destination / 'metrics.json')
            else:
                metrics = evaluate(evaluation_args([
                    '--checkpoint', checkpoint['path'], '--dataset_to_evaluate', system['test_features'],
                    '--labels_path', plan['labels']['path'], '--output_dir', str(destination),
                    '--device', device, '--threads', str(threads),
                    '--bootstrap_samples', str(plan['bootstrap_samples']), '--bootstrap_seed', str(plan['bootstrap_seed'])]))
            manifest = read(destination / 'manifest.json')
            if manifest['recording_ids'] != plan['test_ids'] or metrics['recordings'] != len(plan['test_ids']):
                raise ValueError('Evaluation does not cover the complete official test set')
            directories.append(destination)
            folds.append({'fold': fold['fold'], 'metrics': metrics, 'directory': str(destination)})
        ensemble_dir = folder / 'ensemble'
        metrics = (read(ensemble_dir / 'metrics.json') if ensemble_dir.exists() else
                   summarize_ensemble(directories, plan['test_ids'], ensemble_dir,
                                      samples=plan['bootstrap_samples'], seed=plan['bootstrap_seed']))
        scores = [fold['metrics']['macro_f1'] for fold in folds]
        completed.append({**system, 'fold_results': folds, 'ensemble_directory': str(ensemble_dir),
                          'ensemble_metrics': metrics, 'fold_macro_f1_mean': statistics.mean(scores),
                          'fold_macro_f1_sd': statistics.stdev(scores)})
        print(f"{system['name']}: ensemble test macro-F1={metrics['macro_f1']:.4f}", flush=True)
    write(root / 'summary.json', {'status': 'complete', 'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
                                  'plan': file_identity(root / 'plan.json'), 'systems': completed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root)
    else:
        run(root, args.device, args.threads)


if __name__ == '__main__':
    main()
