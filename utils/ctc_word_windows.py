"""Merge cached CTC character spans into original-word windows (Step 5)."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

from utils.audio_frame_timing import frame_interval_seconds, frame_timing
from utils.transcript_normalization import normalize_ctc_words


INTERVAL_FIELDS = (
    'start_frame', 'end_frame_exclusive', 'start_time', 'end_time',
    'expanded_start_frame', 'expanded_end_frame_exclusive',
    'expanded_start_time', 'expanded_end_time',
)


def merge_word_windows(alignment, context_frames=0):
    """Return one row per original word without pooling or changing the words.

    Frames use half-open intervals. A word spans its first through last CTC
    character, including internal blank frames but excluding exterior blanks
    and delimiters. Context may overlap neighboring words, but never extends
    beyond the valid frames. Words with no CTC units have null intervals.
    """
    if alignment.get('unsupported_word_policy') == 'skip':
        from utils.ctc_skipped_words import merge_skipping_unsupported
        return merge_skipping_unsupported(alignment, context_frames, merge_word_windows)
    if type(context_frames) is not int or context_frames < 0:
        raise ValueError('context_frames must be a nonnegative integer')
    if alignment.get('format') != 'ctc_unit_alignment_v1':
        raise ValueError('Expected a Step 4 ctc_unit_alignment_v1 export')
    if alignment.get('frame_span_convention') != 'start_inclusive_end_exclusive':
        raise ValueError('Expected exclusive-end CTC frame spans')
    words, mappings, units = alignment['words'], alignment['word_mappings'], alignment['units']
    if len(words) != len(mappings):
        raise ValueError('Original word count differs from the word mapping')
    valid_count, timing = alignment['aligned_frames'], alignment['timing']
    if type(valid_count) is not int or valid_count <= 0:
        raise ValueError('Expected a positive valid-frame count')
    canonical = frame_timing(
        SimpleNamespace(conv_kernel=timing['conv_kernel'], conv_stride=timing['conv_stride']),
        timing['num_samples'], timing['sample_rate'], valid_count,
    )
    if any(timing.get(key) != value for key, value in canonical.items()):
        raise ValueError('Timing does not match the valid CTC frame sequence')
    previous_end = 0
    for index, unit in enumerate(units):
        start, end, owner = unit['start_frame'], unit['end_frame_exclusive'], unit['word_id']
        if unit['target_index'] != index:
            raise ValueError('CTC target indices must be complete and ordered')
        if type(owner) is not int or not -1 <= owner < len(words):
            raise ValueError('CTC unit has an invalid original word ID')
        if unit['token_id'] == alignment['blank_id']:
            raise ValueError('Blank frames must not appear as CTC target units')
        if not isinstance(unit['token'], str) or len(unit['token']) != 1:
            raise ValueError('Expected character-level CTC units')
        if (type(start) is not int or type(end) is not int
                or not previous_end <= start < end <= valid_count):
            raise ValueError('CTC units must be monotonic and inside valid frames')
        if (not math.isclose(unit['start_time'], frame_interval_seconds(start, timing)[0], abs_tol=1e-9)
                or not math.isclose(unit['end_time'], frame_interval_seconds(end - 1, timing)[1], abs_tol=1e-9)):
            raise ValueError('CTC unit timestamps disagree with their frame spans')
        previous_end = end

    # Revalidate the original normalization using the characters present in this
    # export. The full model vocabulary is unnecessary for a merge operation.
    vocabulary = {'<blank>': alignment['blank_id']}
    delimiters = {unit['token'] for unit in units if unit['word_id'] == -1}
    if len(delimiters) > 1:
        raise ValueError('Expected a single CTC word-delimiter symbol')
    for unit in units:
        if (type(unit['token_id']) is not int or unit['token_id'] < 0
                or vocabulary.get(unit['token'], unit['token_id']) != unit['token_id']):
            raise ValueError('CTC token IDs are inconsistent')
        vocabulary[unit['token']] = unit['token_id']
    delimiter = next(iter(delimiters), '|')
    if delimiter not in vocabulary:
        vocabulary[delimiter] = max(vocabulary.values()) + 1
    normalized = normalize_ctc_words(words, vocabulary, alignment['blank_id'], delimiter)
    if (normalized['word_mappings'] != mappings
            or normalized['tokens'] != [unit['token'] for unit in units]
            or normalized['token_word_ids'] != [unit['word_id'] for unit in units]):
        raise ValueError('CTC units and mapping do not match the original normalized words')

    windows, claimed, previous_token_end = [], set(), 0
    previous_word_end, previous_expanded_start, previous_expanded_end = 0, 0, 0
    for word_id, (word, mapping) in enumerate(zip(words, mappings)):
        if not isinstance(word, str) or mapping['word_id'] != word_id or mapping['word'] != word:
            raise ValueError('Word mapping no longer matches the exact original word list')
        start, end = mapping['token_start'], mapping['token_end']
        if (type(start) is not int or type(end) is not int
                or not previous_token_end <= start <= end <= len(units)):
            raise ValueError(f'Invalid target-unit range for word {word_id}')
        previous_token_end = end
        word_units = units[start:end]
        if any(unit['word_id'] != word_id for unit in word_units):
            raise ValueError(f'CTC character ownership disagrees for word {word_id}')
        if ''.join(unit['token'] for unit in word_units) != mapping['ctc_form']:
            raise ValueError(f'CTC characters do not reproduce word {word_id}')
        expected_status = 'ready' if word_units else 'no_ctc_units'
        if mapping['status'] != expected_status:
            raise ValueError(f'Word {word_id} has inconsistent alignment status')
        claimed.update(range(start, end))
        row = {
            'word_id': word_id, 'word': word, 'ctc_form': mapping['ctc_form'],
            'status': 'aligned' if word_units else 'no_ctc_units',
            'token_start': start, 'token_end_exclusive': end,
            'ctc_unit_count': len(word_units),
            **{field: None for field in INTERVAL_FIELDS},
        }
        if word_units:
            first, last = word_units[0]['start_frame'], word_units[-1]['end_frame_exclusive']
            expanded_start = max(0, first - context_frames)
            expanded_end = min(valid_count, last + context_frames)
            if (first < previous_word_end or expanded_start < previous_expanded_start
                    or expanded_end < previous_expanded_end):
                raise ValueError('Word windows are not monotonic in original word order')
            previous_word_end = last
            previous_expanded_start, previous_expanded_end = expanded_start, expanded_end
            row.update({
                'start_frame': first, 'end_frame_exclusive': last,
                'start_time': frame_interval_seconds(first, timing)[0],
                'end_time': frame_interval_seconds(last - 1, timing)[1],
                'expanded_start_frame': expanded_start, 'expanded_end_frame_exclusive': expanded_end,
                'expanded_start_time': frame_interval_seconds(expanded_start, timing)[0],
                'expanded_end_time': frame_interval_seconds(expanded_end - 1, timing)[1],
            })
        windows.append(row)
    for index, unit in enumerate(units):
        if index not in claimed and unit['word_id'] != -1:
            raise ValueError('A CTC character is not covered by its original word mapping')
    return {
        'format': 'ctc_word_windows_v1',
        'ctc_model': alignment['ctc_model'], 'model_revision': alignment.get('model_revision'),
        'audio_source': alignment['audio_source'], 'word_source': alignment['word_source'],
        'timing': timing, 'context_frames': context_frames,
        'frame_span_convention': 'start_inclusive_end_exclusive',
        'time_span_definition': 'union_of_convolution_receptive_fields',
        'frame_reference': 'ctc_encoder',
        'words': list(words), 'word_windows': windows,
        'word_alignment_mask': [row['status'] == 'aligned' for row in windows],
        'aligned_word_count': sum(row['status'] == 'aligned' for row in windows),
        'word_intervals_computed': True,
        'embeddings_computed': False,
    }


def export_word_windows(alignment_dir, output_dir, context_frames=0):
    """Save a separate JSON and readable CSV; never overwrite existing data."""
    alignment_dir, output_dir = Path(alignment_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Export already exists: {output_dir}. Choose a new directory.')
    path = alignment_dir / 'alignment.json'
    source = path.read_bytes()
    result = merge_word_windows(json.loads(source), context_frames)
    result['alignment_source'] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(source).hexdigest()}
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    csv_stream = io.StringIO(newline='')
    columns = ['word_id', 'word', 'ctc_form', 'status', 'token_start', 'token_end_exclusive',
               'ctc_unit_count', *INTERVAL_FIELDS]
    writer = csv.DictWriter(csv_stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(result['word_windows'])
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'word_windows.json').write_text(serialized, encoding='utf-8')
    with (output_dir / 'word_windows.csv').open('w', encoding='utf-8', newline='') as stream:
        stream.write(csv_stream.getvalue())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment-dir', type=Path, required=True, help='Existing Step 4 export.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing exports are refused.')
    parser.add_argument('--ctc-context-frames', type=int, default=0, help='Frames to add on each side, clipped to valid frames (default: 0).')
    args = parser.parse_args()
    if args.ctc_context_frames < 0:
        parser.error('--ctc-context-frames must be nonnegative')
    result = export_word_windows(args.alignment_dir, args.output_dir, args.ctc_context_frames)
    print(f"Merged {result['aligned_word_count']} aligned words; preserved {len(result['words'])} original entries")
    print(f"Context: {result['context_frames']} CTC frames per side; saved to {args.output_dir}")
    missing = [row['word_id'] for row in result['word_windows'] if row['status'] != 'aligned']
    if missing:
        print(f'Entries without CTC units have null intervals: {missing}')
    print('Step 6 compares these word boundaries with Whisper; no embeddings have been pooled yet.')


if __name__ == '__main__':
    main()
