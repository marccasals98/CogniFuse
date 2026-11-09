"""Audit one existing ADReSSo word export before adding trainable pooling.

No encoders run and no source files are changed. An optional new JSON report
records provenance and structural checks, not clinical or alignment accuracy.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from utils.batch_word_embeddings import validate_package
from utils.compare_word_alignment import inspect_word_alignment
from utils.word_audio_pool import pool_audio_words


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    return {'available': len(values), 'min': min(values) if values else None,
            'median': statistics.median(values) if values else None,
            'max': max(values) if values else None}


def audit_recording(feature_root, uid, partition='train'):
    """Follow the published package's exact lineage and validate saved inputs."""
    require(uid and Path(uid).name == uid and uid not in ('.', '..'), 'Invalid UID')
    require(partition in ('train', 'test'), 'Expected train or test partition')
    root = Path(feature_root).resolve()
    config = read_json(root / 'config.json')
    summary = read_json(root / 'summary.json')
    key = partition + '/' + uid
    item = summary['records'][key]
    require(item['status'] == 'complete', 'Recording export is not complete')
    checked_files = {}
    checks = []

    def digest(path):
        path = Path(path).resolve()
        if str(path) not in checked_files:
            checksum = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    checksum.update(chunk)
            checked_files[str(path)] = checksum.hexdigest()
        return checked_files[str(path)]

    def verify_source(source):
        require(digest(source['path']) == source['sha256'],
                f'Source hash mismatch: {source["path"]}')

    for source in item['source_sha256'].values():
        verify_source(source)
    package_dir = root / partition / 'embeddings'
    for name, checksum in item['file_sha256'].items():
        require(Path(name).name == name, 'Invalid published filename')
        verify_source({'path': package_dir / name, 'sha256': checksum})
    checks.append('Original audio, word CSV, and published files match summary hashes')

    record = {'uid': uid, **{name: source['path']
                           for name, source in item['source_sha256'].items()}}
    package, windows_dir = validate_package(package_dir, record, config)
    checks.append('Published vectors and masks match their original Step 7/8 inputs')

    audio_dir = Path(package['inputs']['audio']['metadata.json']['path']).parent
    text_dir = Path(package['inputs']['text']['metadata.json']['path']).parent
    audio_meta = read_json(audio_dir / 'metadata.json')
    text_meta = read_json(text_dir / 'metadata.json')
    frames_dir = Path(audio_meta['inputs']['audio_metadata']['path']).parent
    frame_meta = read_json(frames_dir / 'metadata.json')
    windows = read_json(windows_dir / 'word_windows.json')
    alignment_path = Path(windows['alignment_source']['path'])
    alignment = read_json(alignment_path)
    ctc_dir = Path(alignment['ctc_export'])
    ctc_meta = read_json(ctc_dir / 'metadata.json')
    comparison_path = Path(item['comparison_report'])
    comparison = read_json(comparison_path)

    # Select the exact staged package by its published metadata hash, not by
    # attempt ordering. Other stages are followed through explicit provenance.
    candidates = list((root / 'recordings' / partition / uid / 'step9_package').glob(
        '*/' + uid + '_word_metadata.json'))
    candidates = [p.parent for p in candidates
                  if digest(p) == digest(package_dir / (uid + '_word_metadata.json'))]
    require(len(candidates) == 1, 'Expected one staged package matching published metadata')
    stages = {'frames': frames_dir, 'ctc': ctc_dir, 'alignment': alignment_path.parent,
              'windows': windows_dir, 'comparison': comparison_path.parent,
              'audio_words': audio_dir, 'text_words': text_dir, 'package': candidates[0]}
    signature = {'settings': config, 'sources': item['source_sha256']}
    for name, directory in stages.items():
        marker_path = directory / '_batch_complete.json'
        marker = read_json(marker_path)
        require(marker['signature'] == signature, f'Stage signature differs: {name}')
        actual = {p.name: digest(p) for p in sorted(directory.iterdir())
                  if p.is_file() and p.name != '_batch_complete.json'}
        require(actual == marker['files'], f'Stage file manifest differs: {name}')
        digest(marker_path)
    digest(root / 'config.json')
    digest(root / 'summary.json')
    checks.append('All eight stage completion manifests and extraction signatures match')

    require(frame_meta.get('model_revision') is not None,
            'Acoustic frame metadata lacks a checkpoint revision; '
            'this strict audit requires explicit revision provenance')
    require(frame_meta['audio_model'] == config['audio_model']
            and frame_meta['model_revision'] == config['audio_revision'],
            'Acoustic encoder identity differs')
    require(text_meta['text_model'] == config['text_model']
            and text_meta['model_revision'] == config['text_revision'],
            'Text encoder identity differs')
    ctc_revision = config.get('ctc_revision', config['audio_revision'])
    require(ctc_meta['ctc_model'] == config['ctc_model']
            and ctc_meta['model_revision'] == ctc_revision
            and alignment['model_revision'] == ctc_revision, 'CTC identity differs')
    require(frame_meta['source'] == package['audio_source'], 'Acoustic source differs')
    checks.append('Acoustic, text, and CTC checkpoint IDs/revisions match configuration')

    def load(directory, filename):
        return torch.load(directory / filename, map_location='cpu', weights_only=True)

    frames = load(frames_dir, 'frames.pt')
    frame_mask = load(frames_dir, 'valid_mask.pt')
    pooled, audio_mask, rebuilt = pool_audio_words(frames, frame_mask, frame_meta, windows)
    saved_audio = load(package_dir, uid + '_word_audio.pt')
    require(torch.equal(pooled, saved_audio), 'Recomputed acoustic means differ')
    # inputs is added during export, rather than by the pure pooling function.
    require(rebuilt == {k: v for k, v in audio_meta.items() if k != 'inputs'},
            'Recomputed pooling metadata differs')
    checks.append('Timing-aware frame mapping and mean pooling reproduce saved word vectors exactly')

    text = load(text_dir, 'text_word_embeddings.pt')
    text_mask = load(text_dir, 'text_word_mask.pt')
    paired = audio_mask & text_mask
    words = package['words']
    require(package['word_ids'] == list(range(len(words))), 'Original word IDs differ')
    require(text_meta['words'] == words == alignment['words'] == windows['words'],
            'Original word lists differ')
    for i, row in enumerate(text_meta['word_mappings']):
        require(row['word_id'] == i and row['word'] == words[i], 'Text word mapping differs')
        require(bool(text_mask[i]) == (row['status'] == 'encoded'), 'Text status differs')
    require(len(text_meta['word_mappings']) == len(words), 'Text mapping count differs')
    expected_comparison = inspect_word_alignment(windows_dir, threshold_seconds=comparison['threshold_seconds'])
    require(comparison == expected_comparison, 'Recomputed Whisper/CTC comparison differs')
    checks.append('Original word IDs, repeated words, masks, and confidence/boundary rows agree')

    confidence = []
    for i in range(len(words)):
        units = [u for u in alignment['units'] if u['word_id'] == i]
        counts = [u['end_frame_exclusive'] - u['start_frame'] for u in units]
        for unit, count in zip(units, counts):
            require(count > 0 and math.isfinite(unit['mean_log_emission'])
                    and unit['mean_log_emission'] <= 1e-6, 'Invalid CTC character score')
            require(math.isclose(math.exp(unit['mean_log_emission']),
                                 unit['emission_confidence'], rel_tol=1e-6, abs_tol=1e-9),
                    'CTC score/confidence disagree')
        value = math.exp(sum(n * u['mean_log_emission'] for n, u in zip(counts, units))
                         / sum(counts)) if counts else None
        require(bool(counts) == bool(audio_mask[i]), 'CTC unit availability differs')
        confidence.append(value)
    rows = comparison['rows']
    asr = [r['whisper_probability'] for r in rows if r['whisper_probability'] is not None]
    counts = [r['audio_frame_count'] for r in rebuilt['word_windows'] if r['audio_frame_count']]
    methods = Counter(r['mapping_method'] for r in rebuilt['word_windows'] if r['mapping_method'])
    checks.append('Per-word CTC emission confidence is derivable from finite character scores')

    return {
        'format': 'fusion_input_audit_v1', 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed', 'scope': 'One recording; provenance and structural checks only',
        'feature_root': str(root), 'recording_key': key,
        'models': {'audio': {'id': config['audio_model'], 'revision': config['audio_revision']},
                   'text': {'id': config['text_model'], 'revision': config['text_revision']},
                   'ctc': {'id': config['ctc_model'], 'revision': ctc_revision}},
        'audio': {'duration_seconds': frame_meta['timing']['audio_duration_seconds'],
                  'frames_shape': list(frames.shape), 'dtype': str(frames.dtype),
                  'frame_stride_seconds': frame_meta['timing']['frame_stride_seconds'],
                  'receptive_field_seconds': frame_meta['timing']['receptive_field_seconds'],
                  'valid_frames': int(frame_mask.sum()), 'tensor_bytes': frames.numel() * frames.element_size()},
        'words': {'count': len(words), 'audio_valid': int(audio_mask.sum()),
                  'text_valid': int(text_mask.sum()), 'paired_valid': int(paired.sum()),
                  'repeated_word_types': sum(n > 1 for n in Counter(words).values()),
                  'audio_shape': list(saved_audio.shape), 'text_shape': list(text.shape),
                  'context_frames': windows['context_frames'], 'frames_per_word': describe(counts),
                  'frame_mapping_methods': dict(methods), 'text_chunks': len(text_meta['chunks']),
                  'text_chunk_size': text_meta['chunk_size'],
                  'text_truncated_tokens': text_meta['truncated_tokens'],
                  'mean_pool_max_abs_error': float((pooled - saved_audio).abs().max())},
        'reliability': {'asr_confidence': describe(asr),
                        'ctc_emission_confidence': describe([c for c in confidence if c is not None]),
                        'boundary_comparison': comparison['summary'],
                        'boundary_review_threshold_seconds': comparison['threshold_seconds'],
                        'boundary_entropy': 'not exported; no boundary posterior defined',
                        'recording_quality': 'not exported; waveform available for future extraction'},
        'checks': checks, 'stages': {k: str(v) for k, v in stages.items()},
        'verified_file_count': len(checked_files), 'verified_sha256': dict(sorted(checked_files.items())),
        'next_step': 'Package cached frames, mapped word intervals, text vectors, masks, and quality sources',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', type=Path, required=True)
    parser.add_argument('--uid', required=True)
    parser.add_argument('--partition', choices=('train', 'test'), default='train')
    parser.add_argument('--output', type=Path, help='Optional new JSON report; existing files are refused.')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('Report already exists; choose a new output path')
    torch.set_num_threads(1)
    result = audit_recording(args.feature_root, args.uid, args.partition)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
        print(f'Passed {result["recording_key"]}: {result["words"]["count"]} words, '
              f'{result["audio"]["valid_frames"]} frames; report: {args.output}')
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
