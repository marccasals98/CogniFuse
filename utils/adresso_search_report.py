"""Collect aggregate search results for separate original and word-feature tabs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from utils.adresso_test_report import TEST_HEADERS, test_columns
OLD = REPO / 'outputs/adresso_architecture_sweep_20261029'
NEW = REPO / 'outputs/adresso_architecture_sweep_words_20261029'
FEATURES = REPO / 'outputs/adresso_ctc_distilbert_words_20261029'
CONTROL = REPO / 'outputs/adresso_word_repeat_launch_20261029'


def read(path):
    return json.loads(Path(path).read_text())


def collect(root, template, new=False):
    stamp = datetime.now(ZoneInfo('Europe/Madrid')).isoformat(timespec='seconds')
    rows = json.loads(json.dumps(template['rows'][1:28] if new else template['rows']))
    grid = read(CONTROL / 'grid.json')
    preparation = read(FEATURES / 'summary.json') if (FEATURES / 'summary.json').exists() else {}
    for index, trial in enumerate(grid):
        row = rows[index if new else index + 1]
        directory = root / 'trials' / trial['id']
        result = directory / 'results'
        folds = [read(p) for p in sorted(result.glob('fold_*.json'))]
        train = [f['training_macro_f1'] for f in folds]
        val = [f['metrics']['macro_f1'] for f in folds]
        mean = lambda values: statistics.mean(values) if values else ''
        sd = lambda values: statistics.stdev(values) if len(values) > 1 else ''
        row[2:8] = [f'{len(folds)}/5', mean(train), sd(train), mean(val), sd(val),
                    mean(train) - mean(val) if folds else '']
        if (result / 'summary.json').exists():
            summary = read(result / 'summary.json')
            if len(folds) != 5 or abs(row[5] - summary['mean_macro_f1']) > 1e-12:
                raise ValueError('Inconsistent completed fold scores: ' + trial['id'])
            row[11] = 'Complete'
        elif directory.exists():
            row[11] = 'Running (provisional)'
        elif new and not preparation.get('training_ready'):
            row[11] = 'Preprocessing failed' if preparation.get('failed') else 'Awaiting preprocessing'
        else:
            row[11] = 'Queued'
        row[20] = stamp
        row[21] = str(result.relative_to(REPO))
        row[22] = '; '.join(f'{f["fold"]}: {f["selected_epoch"]}' for f in folds)
        if new:
            row[1] = f'Repeat of original T{index:02d}; changed inputs to English CTC word-level features. Same architecture and training settings.'
            row[23] = 'DistilBERT + Wav2Vec2-base-960h; mean subword/audio pooling, common word masks, 200-token text chunks. Original results retained in the old tab.'
        else:
            row[23] = 'Same features, saved folds, seeds, optimizer and budget as B0. Dropout varied jointly.'
        if row[11] != 'Complete':
            row[23] += f' Provisional metrics from {len(folds)} completed folds.' if folds else ' No completed-fold scores yet.'
    for row in rows:
        row[20] = stamp
        row.extend(test_columns(row[21]))
    return {'title': 'hyperparameter tuning new' if new else 'hyperparameter tuning',
            'sheetId': 14902701 if new else 0, 'headers': template['headers'] + TEST_HEADERS, 'rows': rows,
            'snapshot': stamp,
            'completed_trials': sum(r[11] == 'Complete' for r in (rows if new else rows[1:28])),
            'expected_trials': 27}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    template = read(CONTROL / 'original_export_template.json')
    report = {'spreadsheetId': '1rBZgU2UAZCfLE4riItdouDtae7mwiuUpZBqYDn4u1tw',
              'sheets': [collect(OLD, template), collect(NEW, template, new=True)]}
    (CONTROL / 'latest_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
