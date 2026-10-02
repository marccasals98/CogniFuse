"""
Compare cross-validation runs on the same recordings (e.g. the modality ablation).

For every pair of runs it reports:
    1. Confusion matrix of each run (pooled out-of-fold predictions).
    2. Paired patient-level bootstrap of the macro-F1 difference + exact McNemar test.
    3. Complementarity: on which recordings one run is right and the other wrong.

Usage:
    python utils/compare_cv_runs.py \
        both=~/logs/cognifuse/train/cross_validation/<run_both> \
        speech=~/logs/cognifuse/train/cross_validation/<run_speech> \
        text=~/logs/cognifuse/train/cross_validation/<run_text>
"""

import argparse
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np


def load_run(run_dir):
    """Pooled out-of-fold predictions of a cross-validation run, keyed by recording uid."""
    run_dir = Path(run_dir).expanduser()
    summary = json.loads((run_dir / 'summary.json').read_text())
    predictions = {}
    for fold_file in sorted(run_dir.glob('fold_*.json')):
        for p in json.loads(fold_file.read_text())['predictions']:
            predictions[p['uid']] = p
    return summary, predictions


def confusion_matrix(y_true, y_pred, n_classes):
    return np.bincount(y_true * n_classes + y_pred, minlength = n_classes ** 2).reshape(n_classes, n_classes)


def per_class_f1(cm):
    tp = np.diag(cm).astype(float)
    denominator = cm.sum(axis = 0) + cm.sum(axis = 1)
    return np.divide(2 * tp, denominator, out = np.zeros_like(tp), where = denominator > 0)


def macro_f1(y_true, y_pred, n_classes):
    return per_class_f1(confusion_matrix(y_true, y_pred, n_classes)).mean()


def mcnemar_exact(only_a, only_b):
    """Two-sided exact McNemar test on the discordant recordings."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_patient_bootstrap(y_true, pred_a, pred_b, patients, n_classes, n_boot, seed):
    """Resample patients (not recordings) with replacement; returns the bootstrap macro-F1 differences A - B."""
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patients)
    rows_of = {p: np.flatnonzero(patients == p) for p in unique_patients}
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(unique_patients, size = len(unique_patients), replace = True)
        idx = np.concatenate([rows_of[p] for p in sample])
        diffs[b] = macro_f1(y_true[idx], pred_a[idx], n_classes) - macro_f1(y_true[idx], pred_b[idx], n_classes)
    return diffs


def print_confusion(name, cm, classes, summary):
    width = max(len(c) for c in classes) + 2
    print(f"\n[{name}]  macro-F1 out-of-fold = {per_class_f1(cm).mean():.4f} "
          f"(summary.json: {summary['out_of_fold']['macro_f1']:.4f})")
    print(' ' * (width + 6) + 'predit ->')
    print('real'.ljust(width) + ''.join(c.rjust(width) for c in classes) + '   recall')
    for i, c in enumerate(classes):
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() else 0
        print(c.ljust(width) + ''.join(str(v).rjust(width) for v in cm[i]) + f'   {recall:.2f}')
    print('F1 per classe: ' + ', '.join(f'{c} {f:.2f}' for c, f in zip(classes, per_class_f1(cm))))


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs = '+', help = 'name=path_to_cross_validation_run (at least two)')
    parser.add_argument('--n_boot', type = int, default = 10000, help = 'Bootstrap resamples.')
    parser.add_argument('--seed', type = int, default = 1234, help = 'Bootstrap random seed.')
    args = parser.parse_args()

    runs = {}
    for item in args.runs:
        name, path = item.split('=', 1)
        runs[name] = load_run(path)
    if len(runs) < 2:
        parser.error('Give at least two runs.')

    names = list(runs)
    classes = runs[names[0]][0]['classes']
    n_classes = len(classes)
    uids = sorted(runs[names[0]][1])
    for name in names[1:]:
        if sorted(runs[name][1]) != uids:
            raise ValueError(f"Run '{name}' does not contain the same recordings as '{names[0]}'.")
        if runs[name][0]['classes'] != classes:
            raise ValueError(f"Run '{name}' uses different classes.")

    reference = runs[names[0]][1]
    y_true = np.array([reference[u]['label'] for u in uids])
    patients = np.array([reference[u]['patient_id'] for u in uids])
    preds = {name: np.array([runs[name][1][u]['prediction'] for u in uids]) for name in names}
    correct = {name: preds[name] == y_true for name in names}

    print(f"{len(uids)} gravacions, {len(np.unique(patients))} pacients, classes: {classes}")

    print('\n' + '=' * 70 + '\n1. MATRIUS DE CONFUSIÓ (files = classe real, columnes = classe predita)\n' + '=' * 70)
    for name in names:
        print_confusion(name, confusion_matrix(y_true, preds[name], n_classes), classes, runs[name][0])

    print('\n' + '=' * 70 + '\n2. COMPARACIÓ ESTADÍSTICA APARELLADA\n' + '=' * 70)
    print(f"Bootstrap per pacients ({args.n_boot} remostrejos): IC 95% de la diferència de macro-F1 (A - B).")
    print("McNemar exacte: compara encerts/errors gravació per gravació (no té en compte que un pacient pot tenir diverses gravacions).\n")
    print(f"{'A vs B':<20}{'dif. macro-F1':>14}{'IC 95%':>20}{'p bootstrap':>13}{'p McNemar':>11}")
    for a, b in itertools.combinations(names, 2):
        diff = macro_f1(y_true, preds[a], n_classes) - macro_f1(y_true, preds[b], n_classes)
        boot = paired_patient_bootstrap(y_true, preds[a], preds[b], patients, n_classes, args.n_boot, args.seed)
        low, high = np.percentile(boot, [2.5, 97.5])
        p_boot = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        only_a = int((correct[a] & ~correct[b]).sum())
        only_b = int((~correct[a] & correct[b]).sum())
        print(f"{a + ' vs ' + b:<20}{diff:>+14.3f}{f'[{low:+.3f}, {high:+.3f}]':>20}{p_boot:>13.3f}{mcnemar_exact(only_a, only_b):>11.3f}")

    print('\n' + '=' * 70 + '\n3. COMPLEMENTARIETAT (nombre de gravacions)\n' + '=' * 70)
    for a, b in itertools.combinations(names, 2):
        both_right = correct[a] & correct[b]
        only_a = correct[a] & ~correct[b]
        only_b = ~correct[a] & correct[b]
        both_wrong = ~correct[a] & ~correct[b]
        print(f"\n[{a} vs {b}]")
        print(f"{'classe':<10}{'tots dos encerten':>19}{'només ' + a:>16}{'només ' + b:>16}{'tots dos fallen':>17}")
        for i, c in enumerate(classes + ['TOTAL']):
            rows = (y_true == i) if c != 'TOTAL' else np.ones_like(y_true, dtype = bool)
            print(f"{c:<10}{(both_right & rows).sum():>19}{(only_a & rows).sum():>16}{(only_b & rows).sum():>16}{(both_wrong & rows).sum():>17}")
        oracle = (correct[a] | correct[b]).mean()
        agreement = (preds[a] == preds[b]).mean()
        print(f"Coincideixen en la predicció: {agreement:.0%}. "
              f"Encert si sempre triéssim el model que encerta (oracle): {oracle:.0%} "
              f"(vs {correct[a].mean():.0%} i {correct[b].mean():.0%}).")


if __name__ == '__main__':
    main()
