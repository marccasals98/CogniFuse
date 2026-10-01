"""Mean-pool cached acoustic frames into original-word embeddings (Step 7)."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.audio_frame_timing import frame_interval_seconds, frame_timing
from utils.ctc_word_windows import merge_word_windows


def validate_timing(timing, count):
    expected = frame_timing(
        SimpleNamespace(conv_kernel=timing['conv_kernel'], conv_stride=timing['conv_stride']),
        timing['num_samples'], timing['sample_rate'], count,
    )
    if any(timing.get(key) != value for key, value in expected.items()):
        raise ValueError('Timing metadata disagrees with the valid frame grid')


def map_frame_window(start, end, ctc_timing, audio_timing):
    """Map a half-open CTC span to valid acoustic frames.

    Identical grids preserve frame indices exactly. Otherwise use acoustic
    frame centers inside the CTC convolution-support interval. If a short
    interval contains no center, use the valid frame nearest its midpoint.
    This fallback is explicitly recorded, never confused with an exact match.
    """
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= ctc_timing['valid_frame_count']):
        raise ValueError('CTC window is outside valid frames')
    grid_keys = ('sample_rate', 'num_samples', 'frame_stride_samples',
                 'receptive_field_samples', 'frame_count')
    if all(ctc_timing[key] == audio_timing[key] for key in grid_keys):
        return start, end, 'identical_grid'
    begin = frame_interval_seconds(start, ctc_timing)[0]
    finish = frame_interval_seconds(end - 1, ctc_timing)[1]
    origin = audio_timing['first_frame_center_seconds']
    stride = audio_timing['frame_stride_seconds']
    count = audio_timing['valid_frame_count']
    # Tolerance is measured in frames and avoids float round-off at boundaries.
    first = max(0, min(count, math.ceil((begin - origin) / stride - 1e-9)))
    last = max(0, min(count, math.ceil((finish - origin) / stride - 1e-9)))
    if first < last:
        return first, last, 'frame_centers_in_interval'
    nearest = math.floor(((begin + finish) / 2 - origin) / stride + 0.5)
    nearest = max(0, min(count - 1, nearest))
    return nearest, nearest + 1, 'nearest_center_fallback'


def pool_audio_words(features, valid_mask, audio_metadata, windows):
    """Return [N_words, D_audio], [N_words] mask, and explicit pooling metadata."""
    if audio_metadata.get('format') != 'acoustic_frames_v1' or windows.get('format') != 'ctc_word_windows_v1':
        raise ValueError('Expected Step 2 acoustic frames and Step 5 word windows')
    if audio_metadata.get('augmentation') is not None or audio_metadata.get('pooled') is not False:
        raise ValueError('Expected unaugmented, unpooled acoustic features')
    if audio_metadata['source'] != windows['audio_source']:
        raise ValueError('Acoustic features and CTC windows refer to different source audio')
    if features.ndim != 2 or features.shape[1] == 0 or not features.is_floating_point():
        raise ValueError('Expected a floating acoustic tensor [T, D]')
    if list(features.shape) != audio_metadata['feature_shape']:
        raise ValueError('Acoustic tensor shape differs from its metadata')
    if valid_mask.dtype != torch.bool or valid_mask.shape != (len(features),):
        raise ValueError('Expected a boolean acoustic valid-frame mask [T]')
    count = int(valid_mask.sum())
    if count == 0 or not torch.equal(valid_mask, torch.arange(len(features), device=valid_mask.device) < count):
        raise ValueError('Valid acoustic frames must form a nonempty contiguous prefix')
    audio_timing, ctc_timing = audio_metadata['timing'], windows['timing']
    validate_timing(audio_timing, count)
    validate_timing(ctc_timing, ctc_timing['valid_frame_count'])
    tolerance = max(1 / audio_timing['sample_rate'], 1 / ctc_timing['sample_rate'])
    if abs(audio_timing['audio_duration_seconds'] - ctc_timing['audio_duration_seconds']) > tolerance:
        raise ValueError('CTC and acoustic recording durations differ')
    if not torch.isfinite(features[:count]).all():
        raise ValueError('Valid acoustic frames contain nonfinite features')
    words, source_windows = windows['words'], windows['word_windows']
    if not len(words) == len(source_windows) == len(windows['word_alignment_mask']):
        raise ValueError('Original words, windows and alignment mask have different lengths')
    margin = windows['context_frames']
    if type(margin) is not int or margin < 0:
        raise ValueError('Invalid CTC context margin')
    embeddings = torch.zeros(len(words), features.shape[1], dtype=torch.float32)
    word_mask = torch.zeros(len(words), dtype=torch.bool)
    rows, previous_start, previous_end = [], 0, 0
    for word_id, (word, window) in enumerate(zip(words, source_windows)):
        if window['word_id'] != word_id or window['word'] != word:
            raise ValueError('Word windows do not preserve the original word mapping')
        row = {
            'word_id': word_id, 'word': word, 'status': window['status'],
            'ctc_start_frame': None, 'ctc_end_frame_exclusive': None,
            'ctc_pool_start_time': None, 'ctc_pool_end_time': None,
            'audio_start_frame': None, 'audio_end_frame_exclusive': None,
            'audio_start_time': None, 'audio_end_time': None,
            'audio_frame_count': 0, 'mapping_method': None,
        }
        if window['status'] == 'no_ctc_units':
            if windows['word_alignment_mask'][word_id] is not False:
                raise ValueError('Unaligned word disagrees with its validity mask')
            if any(window[key] is not None for key in ('start_frame', 'end_frame_exclusive', 'expanded_start_frame', 'expanded_end_frame_exclusive')):
                raise ValueError('Unaligned words must have null frame intervals')
        elif window['status'] == 'aligned' and windows['word_alignment_mask'][word_id] is True:
            raw_start, raw_end = window['start_frame'], window['end_frame_exclusive']
            if (type(raw_start) is not int or type(raw_end) is not int
                    or not 0 <= raw_start < raw_end <= ctc_timing['valid_frame_count']):
                raise ValueError('Word interval is outside valid CTC frames')
            start, end = window['expanded_start_frame'], window['expanded_end_frame_exclusive']
            if start != max(0, raw_start - margin) or end != min(ctc_timing['valid_frame_count'], raw_end + margin):
                raise ValueError('Expanded window disagrees with its context margin')
            first, last, method = map_frame_window(start, end, ctc_timing, audio_timing)
            if not 0 <= first < last <= count or first < previous_start or last < previous_end:
                raise ValueError('Mapped acoustic windows are invalid or nonmonotonic')
            previous_start, previous_end = first, last
            embeddings[word_id] = features[first:last].detach().float().mean(dim=0).cpu()
            word_mask[word_id] = True
            row.update({
                'ctc_start_frame': start, 'ctc_end_frame_exclusive': end,
                'ctc_pool_start_time': frame_interval_seconds(start, ctc_timing)[0],
                'ctc_pool_end_time': frame_interval_seconds(end - 1, ctc_timing)[1],
                'audio_start_frame': first, 'audio_end_frame_exclusive': last,
                'audio_start_time': frame_interval_seconds(first, audio_timing)[0],
                'audio_end_time': frame_interval_seconds(last - 1, audio_timing)[1],
                'audio_frame_count': last - first, 'mapping_method': method,
            })
        else:
            raise ValueError('Word alignment status and mask disagree')
        rows.append(row)
    if not torch.isfinite(embeddings).all():
        raise ValueError('Mean pooling produced nonfinite embeddings')
    metadata = {
        'format': 'audio_word_embeddings_v1', 'pooling': 'mean',
        'words': list(words), 'word_windows': rows,
        'feature_shape': list(embeddings.shape), 'feature_dtype': str(embeddings.dtype),
        'valid_word_count': int(word_mask.sum()), 'context_frames': margin,
        'audio_model': audio_metadata['audio_model'], 'ctc_model': windows['ctc_model'],
        'audio_source': audio_metadata['source'], 'word_source': windows['word_source'],
        'audio_timing': audio_timing, 'ctc_timing': ctc_timing,
        'frame_span_convention': 'start_inclusive_end_exclusive',
        'missing_word_policy': 'zero vector with false word mask; original indices preserved',
        'embeddings_file': 'audio_word_embeddings.pt', 'word_mask_file': 'word_mask.pt',
        'text_embeddings_computed': False,
    }
    return embeddings, word_mask, metadata


def export_audio_words(audio_frames_dir, word_windows_dir, output_dir):
    """Pool saved inputs and publish a new export, refusing existing outputs."""
    audio_frames_dir, word_windows_dir, output_dir = map(Path, (audio_frames_dir, word_windows_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f'Export already exists: {output_dir}. Choose a new directory.')
    audio_path, words_path = audio_frames_dir / 'metadata.json', word_windows_dir / 'word_windows.json'
    audio_metadata, windows = json.loads(audio_path.read_bytes()), json.loads(words_path.read_bytes())
    alignment_path = Path(windows['alignment_source']['path'])
    source_bytes = alignment_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != windows['alignment_source']['sha256']:
        raise ValueError('Source character alignment changed since Step 5')
    expected = merge_word_windows(json.loads(source_bytes), windows['context_frames'])
    if any(windows.get(key) != value for key, value in expected.items()):
        raise ValueError('Word windows disagree with their source alignment')
    features_path, mask_path = audio_frames_dir / 'frames.pt', audio_frames_dir / 'valid_mask.pt'
    features = torch.load(features_path, map_location='cpu', weights_only=True)
    mask = torch.load(mask_path, map_location='cpu', weights_only=True)
    embeddings, word_mask, metadata = pool_audio_words(features, mask, audio_metadata, windows)
    metadata['inputs'] = {
        name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in [('audio_metadata', audio_path), ('frames', features_path),
                           ('valid_frame_mask', mask_path), ('word_windows', words_path)]
    }
    serialized = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(embeddings, output_dir / 'audio_word_embeddings.pt')
    torch.save(word_mask, output_dir / 'word_mask.pt')
    (output_dir / 'metadata.json').write_text(serialized, encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-frames-dir', type=Path, required=True, help='Existing Step 2 acoustic export.')
    parser.add_argument('--word-windows-dir', type=Path, required=True, help='Existing Step 5 word windows.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New output directory; never overwrite existing files.')
    args = parser.parse_args()
    metadata = export_audio_words(args.audio_frames_dir, args.word_windows_dir, args.output_dir)
    print(f"Saved acoustic word embeddings {metadata['feature_shape']} to {args.output_dir}")
    print(f"Valid words: {metadata['valid_word_count']}/{len(metadata['words'])}; context: {metadata['context_frames']} CTC frames per side")
    fallback = sum(row['mapping_method'] == 'nearest_center_fallback' for row in metadata['word_windows'])
    if fallback:
        print(f'Nearest-frame fallback used for {fallback} short word intervals; see metadata.')
    print('Text word embeddings follow in Step 8; existing training outputs are unchanged.')


if __name__ == '__main__':
    main()
