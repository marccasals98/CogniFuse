"""Compare saved CTC word boundaries with the original Whisper word CSV."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

from utils.ctc_word_windows import merge_word_windows
from utils.ctc_skipped_words import skip_metadata


def compare_word_windows(windows, whisper_rows, threshold_seconds=0.2):
    """Compare matching original words; disagreement is not ground-truth error.

    CTC boundaries use the original, unexpanded word interval. Empty CTC
    entries stay in the report with null comparisons and are excluded from
    aggregate difference statistics.
    """
    if not math.isfinite(threshold_seconds) or threshold_seconds < 0:
        raise ValueError('Disagreement threshold must be finite and nonnegative')
    if windows.get('format') != 'ctc_word_windows_v1':
        raise ValueError('Expected Step 5 ctc_word_windows_v1 output')
    words, word_windows = windows['words'], windows['word_windows']
    if not len(words) == len(word_windows) == len(whisper_rows) == len(windows['word_alignment_mask']):
        raise ValueError('Whisper and CTC original word counts differ')
    rows = []
    for word_id, (word, ctc, whisper) in enumerate(zip(words, word_windows, whisper_rows)):
        if ctc['word_id'] != word_id or ctc['word'] != word or whisper['word'] != word:
            raise ValueError(f'Exact original-word match failed at word {word_id}')
        start, end = float(whisper['start']), float(whisper['end'])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError(f'Invalid Whisper timestamps at word {word_id}')
        probability = whisper.get('probability')
        probability = float(probability) if probability not in (None, '') else None
        if probability is not None and not (math.isfinite(probability) and 0 <= probability <= 1):
            raise ValueError(f'Invalid Whisper probability at word {word_id}')
        row = {
            'word_id': word_id, 'word': word, 'status': ctc['status'],
            'whisper_start': start, 'whisper_end': end, 'whisper_probability': probability,
            'ctc_start': None, 'ctc_end': None, 'start_frame': None, 'end_frame_exclusive': None,
            'delta_start': None, 'delta_end': None,
            'signed_start_difference': None, 'signed_end_difference': None,
            'review_flag': None,
        }
        if ctc['status'] == 'aligned':
            if windows['word_alignment_mask'][word_id] is not True:
                raise ValueError(f'CTC validity mask disagrees at word {word_id}')
            ctc_start, ctc_end = ctc['start_time'], ctc['end_time']
            if not (math.isfinite(ctc_start) and math.isfinite(ctc_end) and 0 <= ctc_start <= ctc_end):
                raise ValueError(f'Invalid CTC timestamps at word {word_id}')
            delta_start, delta_end = abs(ctc_start - start), abs(ctc_end - end)
            row.update({
                'ctc_start': ctc_start, 'ctc_end': ctc_end,
                'start_frame': ctc['start_frame'], 'end_frame_exclusive': ctc['end_frame_exclusive'],
                'delta_start': delta_start, 'delta_end': delta_end,
                'signed_start_difference': ctc_start - start,
                'signed_end_difference': ctc_end - end,
                'review_flag': max(delta_start, delta_end) > threshold_seconds + 1e-9,
            })
        elif ctc['status'] != 'no_ctc_units' or windows['word_alignment_mask'][word_id] is not False:
            raise ValueError(f'Unexpected CTC alignment status at word {word_id}')
        elif any(ctc[key] is not None for key in ('start_time', 'end_time', 'start_frame', 'end_frame_exclusive')):
            raise ValueError(f'Unaligned word {word_id} must have null CTC intervals')
        rows.append(row)
    compared = [row for row in rows if row['ctc_start'] is not None]
    summary = {
        'total_words': len(rows), 'compared_words': len(compared),
        'words_without_ctc': len(rows) - len(compared),
        'review_flag_count': sum(row['review_flag'] for row in compared),
        'review_word_ids': [row['word_id'] for row in compared if row['review_flag']],
    }
    for boundary in ('start', 'end'):
        values = [row['delta_' + boundary] for row in compared]
        for name, function in [('mean', statistics.mean), ('median', statistics.median), ('max', max)]:
            summary[f'{name}_{boundary}_disagreement_seconds'] = function(values) if values else None
    return {
        **skip_metadata(windows),
        'format': 'whisper_ctc_comparison_v1', 'words': list(words), 'rows': rows, 'summary': summary,
        'threshold_seconds': threshold_seconds,
        'threshold_rule': 'either absolute boundary difference exceeds threshold (1e-9 s tolerance)',
        'ctc_boundaries': 'original_unexpanded', 'context_frames_in_source': windows['context_frames'],
        'frame_span_convention': windows['frame_span_convention'],
        'time_span_definition': windows['time_span_definition'],
        'signed_difference_definition': 'CTC minus Whisper, in seconds',
        'interpretation': 'Disagreement is a review signal, not proof either method is correct.',
        'audio_source': windows['audio_source'], 'word_source': windows['word_source'],
    }


def format_report(comparison):
    """Make a terminal table in original word order, including repeated words."""
    rows = comparison['rows']
    width = max([4] + [len(row['word']) for row in rows])
    lines = [f"RECORDING: {Path(comparison['audio_source']['path']).name}",
             'Times in seconds; CTC frames use [start, end). Boundaries exclude added context.',
             f"{'ID':>3}  {'Word':<{width}}  {'Whisper':>15}  {'CTC':>15}  {'Frames':>13}  {'Δ start':>8}  {'Δ end':>8}  Review"]
    lines.append('-' * len(lines[-1]))
    for row in rows:
        whisper = f"{row['whisper_start']:.3f}-{row['whisper_end']:.3f}"
        if row['ctc_start'] is None:
            ctc, frames, start, end, flag = '-', '-', '-', '-', 'no CTC units'
        else:
            ctc = f"{row['ctc_start']:.3f}-{row['ctc_end']:.3f}"
            frames = f"[{row['start_frame']},{row['end_frame_exclusive']})"
            start, end = f"{row['delta_start']:.3f}", f"{row['delta_end']:.3f}"
            flag = '*' if row['review_flag'] else ''
        lines.append(f"{row['word_id']:>3}  {row['word']:<{width}}  {whisper:>15}  {ctc:>15}  {frames:>13}  {start:>8}  {end:>8}  {flag}")
    summary = comparison['summary']
    lines.extend(['', f"Compared {summary['compared_words']}/{summary['total_words']} words; {summary['words_without_ctc']} without CTC intervals.",
                  f"Review (*): {summary['review_flag_count']} words exceed {comparison['threshold_seconds']:.3f} s at either boundary."])
    if summary['compared_words']:
        lines.append(f"Mean absolute disagreement: start {summary['mean_start_disagreement_seconds']:.3f} s; end {summary['mean_end_disagreement_seconds']:.3f} s.")
    lines.append(comparison['interpretation'])
    return '\n'.join(lines) + '\n'


def inspect_word_alignment(windows_dir, output_dir=None, threshold_seconds=0.2):
    """Validate source identity, compare, and optionally save new review files."""
    windows_dir = Path(windows_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError(f'Report already exists: {output_dir}. Choose a new directory.')
    path = windows_dir / 'word_windows.json'
    window_bytes = path.read_bytes()
    windows = json.loads(window_bytes)
    alignment_path = Path(windows['alignment_source']['path'])
    alignment_bytes = alignment_path.read_bytes()
    if hashlib.sha256(alignment_bytes).hexdigest() != windows['alignment_source']['sha256']:
        raise ValueError('Step 4 alignment changed since the word windows were generated')
    expected = merge_word_windows(json.loads(alignment_bytes), windows['context_frames'])
    if any(windows.get(key) != value for key, value in expected.items()):
        raise ValueError('Step 5 windows disagree with their source alignment')
    whisper_path = Path(windows['word_source']['path'])
    source_stat = whisper_path.stat()
    if (source_stat.st_size != windows['word_source']['size_bytes']
            or source_stat.st_mtime_ns != windows['word_source']['mtime_ns']):
        raise ValueError('Whisper CSV changed since CTC preparation; use the original CSV or regenerate the dependent exports')
    whisper_bytes = whisper_path.read_bytes()
    reader = csv.DictReader(io.StringIO(whisper_bytes.decode('utf-8'), newline=''))
    if not {'word', 'start', 'end'}.issubset(reader.fieldnames or []):
        raise ValueError('Whisper CSV requires word, start, and end columns')
    comparison = compare_word_windows(windows, list(reader), threshold_seconds)
    comparison['inputs'] = {
        'word_windows': {'path': str(path.resolve()), 'sha256': hashlib.sha256(window_bytes).hexdigest()},
        'whisper_csv': {'path': str(whisper_path.resolve()), 'sha256': hashlib.sha256(whisper_bytes).hexdigest()},
    }
    text = format_report(comparison)
    if output_dir is not None:
        serialized = json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        buffer = io.StringIO(newline='')
        if comparison['rows']:
            writer = csv.DictWriter(buffer, fieldnames=list(comparison['rows'][0]))
            writer.writeheader()
            writer.writerows(comparison['rows'])
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / 'comparison.json').write_text(serialized, encoding='utf-8')
        (output_dir / 'comparison.txt').write_text(text, encoding='utf-8')
        with (output_dir / 'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
            stream.write(buffer.getvalue())
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--word-windows-dir', type=Path, required=True, help='Existing Step 5 export.')
    parser.add_argument('--output-dir', type=Path, help='Optional new report directory; omit to print only.')
    parser.add_argument('--disagreement-threshold', type=float, default=0.2, help='Review threshold in seconds, not a quality pass/fail cutoff (default: 0.2).')
    args = parser.parse_args()
    comparison = inspect_word_alignment(args.word_windows_dir, args.output_dir, args.disagreement_threshold)
    print(format_report(comparison), end='')
    if args.output_dir is not None:
        print(f'Saved comparison to {args.output_dir}')


if __name__ == '__main__':
    main()
