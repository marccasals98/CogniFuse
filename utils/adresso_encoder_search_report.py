"""Export the DeBERTa/WavLM tuning table without modifying historical results."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

from utils.tune_adresso import completed_trial, read_json, write_json


TITLE = 'Tuning - DeBERTa-large WavLM-large'
HEADERS = ['Experiment', 'Reference and changed setting', 'Completed folds',
           'Train macro-F1 mean', 'Train macro-F1 SD', 'Val macro-F1 mean', 'Val macro-F1 SD',
           'Train–val gap', 'Trainable parameters', 'Batch size', 'Epochs', 'Status',
           'Attention heads', 'Seq input dropout', 'Pool input dropout', 'Classifier dropout',
           'Additional hidden layers', 'Hidden width', 'Eval every (updates)', 'Checkpoint protocol',
           'Updated (Europe/Madrid)', 'Results source', 'Selected epoch by fold', 'Notes', 'Job ID']


def collect(root):
    manifest = read_json(root / 'manifest.json')
    counts = read_json(root / 'model_parameter_counts.json')
    jobs = read_json(root / 'jobs.json') if (root / 'jobs.json').exists() else {}
    fixed = manifest['fixed_parameters']
    rows = []
    stamp = datetime.now(ZoneInfo('Europe/Madrid')).isoformat(timespec='seconds')
    for trial in manifest['trials']:
        index, trial_id = trial['index'], trial['id']
        directory = root / 'trials' / trial_id
        result = directory / 'results'
        folds = [read_json(p) for p in sorted(result.glob('fold_*.json'))]
        if len(folds) > 5 or len({f['fold'] for f in folds}) != len(folds):
            raise ValueError('Invalid fold coverage')
        train, val = [f['training_macro_f1'] for f in folds], [f['metrics']['macro_f1'] for f in folds]
        mean = lambda values: statistics.mean(values) if values else ''
        sd = lambda values: statistics.stdev(values) if len(values) > 1 else ''
        if (result / 'summary.json').exists():
            verified = completed_trial(root, trial)
            if len(folds) != 5 or abs(mean(val) - verified['mean_macro_f1']) > 1e-12:
                raise ValueError('Completed summary differs from fold metrics')
            status = 'Complete'
        elif (directory / 'failed.json').exists():
            status = 'Failed'
        else:
            status = 'Running (provisional)' if directory.exists() else 'Queued'
        rows.append([
            f'T{index:02d} — {trial_id}',
            f'Repeat of CTC-word T{index:02d}; DeBERTa-v3-large + WavLM-large; feature widths 768 → 1024. Same architecture grid and training settings.',
            f'{len(folds)}/5', mean(train), sd(train), mean(val), sd(val),
            mean(train) - mean(val) if folds else '', counts[trial_id],
            fixed['training_batch_size'], fixed['max_epochs'], status,
            trial['seq_to_seq_heads_number'], trial['seq_to_seq_input_dropout'],
            trial['seq_to_one_input_dropout'], trial['classifier_layer_drop_out'],
            trial['classifier_hidden_layers'], fixed['classifier_hidden_layers_width'],
            fixed['eval_and_save_best_model_every'], 'best_validation_checkpoint', stamp, str(result),
            '; '.join(f'{f["fold"]}: {f["selected_epoch"]}' for f in folds),
            'English Wav2Vec2 CTC alignment, 0 extra context; mean WavLM frames/DeBERTa subwords; 200-token text chunks; paired word masks. '
            'Split/Python/NumPy seed=1234; Torch seed=3407. Official test excluded from tuning. '
            'Dropout varied jointly. Hidden-layer count excludes the initial hidden block. '
            + (f'Provisional metrics from {len(folds)} completed folds.' if folds and status != 'Complete' else ''),
            f'{jobs["array"]}_{index}' if jobs.get('array') else '',
        ])
    return {'title': TITLE, 'headers': HEADERS, 'rows': rows, 'snapshot': stamp,
            'completed_trials': sum(row[11] == 'Complete' for row in rows), 'expected_trials': 27}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-root', type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.sweep_root.resolve())
    write_json(args.sweep_root / 'spreadsheet_export.json', report)
    print(json.dumps({'completed_trials': report['completed_trials'], 'expected_trials': 27,
                      'export': str(args.sweep_root / 'spreadsheet_export.json')}))


if __name__ == '__main__':
    main()
