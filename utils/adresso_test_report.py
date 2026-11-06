"""Read completed official-test scores without loading models or changing CV ranks."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / 'outputs/adresso_test_evaluation_20261101'
TEST_HEADERS = [
    'Test status', 'Test policy', 'Test N', 'Test macro-F1', 'Test accuracy',
    'Test cn F1', 'Test ad F1', 'Test F1 95% CI low', 'Test F1 95% CI high',
    'Test fold F1 mean', 'Test fold F1 SD', 'Test results source',
]


def test_columns(results_directory, root=DEFAULT_ROOT):
    summary_path = Path(root) / 'summary.json'
    empty = ['Not included'] + [''] * (len(TEST_HEADERS) - 1)
    if not summary_path.exists():
        return empty
    summary = json.loads(summary_path.read_text())
    if summary.get('status') != 'complete':
        return empty
    source = Path(results_directory)
    source = source if source.is_absolute() else REPO / source
    for system in summary['systems']:
        if source.resolve() == Path(system['training_results']).resolve():
            metrics = system['ensemble_metrics']
            ci = metrics['macro_f1_bootstrap']
            return ['Complete', 'Mean probabilities of 5 validation-selected checkpoints',
                    metrics['recordings'], metrics['macro_f1'], metrics['accuracy'],
                    metrics['per_class_f1']['cn'], metrics['per_class_f1']['ad'],
                    ci['lower'], ci['upper'], system['fold_macro_f1_mean'],
                    system['fold_macro_f1_sd'], system['ensemble_directory']]
    return empty


def detailed_report(root=DEFAULT_ROOT):
    """Produce the separate test tab from completed, auditable evaluation outputs."""
    summary = json.loads((Path(root) / 'summary.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('Only complete test evaluations may be published')
    stamp = datetime.fromisoformat(summary['evaluated_at_utc']).astimezone(ZoneInfo('Europe/Madrid')).isoformat(timespec='seconds')
    headers = ['Experiment', 'Representation', 'Evaluation', 'Fold', 'Checkpoints', 'Test N',
               'Test macro-F1', 'Test accuracy', 'Test cn F1', 'Test ad F1',
               'F1 95% CI low', 'F1 95% CI high', 'Prediction policy', 'Model selection',
               'Evaluated (Europe/Madrid)', 'Results source']
    rows = []
    for system in summary['systems']:
        entries = [('Five-model ensemble', '0–4', 5, system['ensemble_metrics'], system['ensemble_directory'])]
        entries.extend(('Individual checkpoint', str(f['fold']), 1, f['metrics'], f['directory']) for f in system['fold_results'])
        for evaluation, fold, count, metrics, directory in entries:
            ci = metrics['macro_f1_bootstrap']
            rows.append([system['name'], system['representation'], evaluation, fold, count,
                         metrics['recordings'], metrics['macro_f1'], metrics['accuracy'],
                         metrics['per_class_f1']['cn'], metrics['per_class_f1']['ad'], ci['lower'], ci['upper'],
                         'Mean checkpoint probabilities, then argmax' if count == 5 else 'Argmax checkpoint probabilities',
                         'Original baseline' if system['id'] == 'baseline' else 'Highest five-fold validation macro-F1; no test selection',
                         stamp, directory])
    return {'title': 'Test evaluation', 'headers': headers, 'rows': rows}
