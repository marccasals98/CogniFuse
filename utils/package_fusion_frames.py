"""Package audited frozen frames and text words for future trainable pooling.

This exporter runs no encoders. The standalone reader uses only package files;
absolute paths in provenance are never needed to load a completed package.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from utils.audit_fusion_inputs import audit_recording, read_json, require
from utils.word_audio_pool import map_frame_window, pool_audio_words, validate_timing


FORMAT = 'fusion_frame_inputs_v1'
TENSOR_DTYPES = {
    'frames.pt': torch.float32,
    'frame_mask.pt': torch.bool,
    'word_ids.pt': torch.int64,
    'word_frame_intervals.pt': torch.int64,
    'audio_word_mask.pt': torch.bool,
    'text_words.pt': torch.float32,
    'text_word_mask.pt': torch.bool,
    'paired_word_mask.pt': torch.bool,
    'word_quality.pt': torch.float32,
    'word_quality_observed.pt': torch.bool,
    'recording_quality.pt': torch.float32,
    'recording_quality_observed.pt': torch.bool,
}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def schema_fingerprint(schema):
    encoded = json.dumps(schema, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def load_tensor(directory, filename):
    return torch.load(Path(directory) / filename, map_location='cpu', weights_only=True)


def mean_word_vectors(tensors):
    """Reconstruct raw acoustic means for compatibility verification, not training."""
    frames = tensors['frames.pt']
    result = frames.new_zeros(len(tensors['word_ids.pt']), frames.shape[1])
    for i in tensors['audio_word_mask.pt'].nonzero().flatten().tolist():
        start, end = tensors['word_frame_intervals.pt'][i].tolist()
        result[i] = frames[start:end].mean(dim=0)
    return result


def read_fusion_package(directory):
    """Validate hashes, schema, masks, intervals, and shapes without source access."""
    directory = Path(directory)
    metadata = read_json(directory / 'metadata.json')
    require(metadata['format'] == FORMAT, 'Expected a fusion_frame_inputs_v1 package')
    schema = metadata['schema']
    require(metadata['schema_sha256'] == schema_fingerprint(schema), 'Schema fingerprint differs')
    require(schema['format'] == FORMAT, 'Schema format differs')
    require(set(metadata['tensors']) == set(TENSOR_DTYPES), 'Incomplete tensor manifest')
    tensors = {}
    for name, dtype in TENSOR_DTYPES.items():
        spec = metadata['tensors'][name]
        require(digest(directory / name) == spec['sha256'], f'Tensor hash differs: {name}')
        tensor = load_tensor(directory, name)
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == dtype
                and str(tensor.dtype) == spec['dtype'] and list(tensor.shape) == spec['shape'],
                f'Tensor shape or dtype differs: {name}')
        require(not tensor.requires_grad, f'Saved encoder features must be detached: {name}')
        if tensor.is_floating_point():
            require(bool(torch.isfinite(tensor).all()), f'Nonfinite tensor: {name}')
        tensors[name] = tensor

    dims = metadata['dimensions']
    require(set(dims) == {'frames', 'words', 'audio', 'text'}
            and all(type(v) is int and v > 0 for v in dims.values()), 'Invalid dimensions')
    t, n, a, e = (dims[k] for k in ('frames', 'words', 'audio', 'text'))
    require(schema['audio_width'] == a and schema['text_width'] == e, 'Encoder widths differ')
    quality = schema['quality']
    require(quality == {'status': 'pending', 'word_features': [], 'recording_features': []},
            'Step 2 supports only the pending, empty quality schema')
    shapes = {'frames.pt': (t, a), 'frame_mask.pt': (t,), 'word_ids.pt': (n,),
              'word_frame_intervals.pt': (n, 2), 'audio_word_mask.pt': (n,),
              'text_words.pt': (n, e), 'text_word_mask.pt': (n,), 'paired_word_mask.pt': (n,),
              'word_quality.pt': (n, 0), 'word_quality_observed.pt': (n, 0),
              'recording_quality.pt': (0,), 'recording_quality_observed.pt': (0,)}
    for name, shape in shapes.items():
        require(tuple(tensors[name].shape) == shape, f'Invalid package dimensions: {name}')
    require(metadata['partition'] in ('train', 'test') and metadata['uid'] not in ('', '.', '..')
            and Path(metadata['uid']).name == metadata['uid'], 'Invalid recording identity')
    require(len(metadata['words']) == n and all(isinstance(w, str) for w in metadata['words']),
            'Invalid original words')
    require(torch.equal(tensors['word_ids.pt'], torch.arange(n)), 'Original word IDs differ')
    audio_mask, text_mask = tensors['audio_word_mask.pt'], tensors['text_word_mask.pt']
    paired = audio_mask & text_mask
    require(torch.equal(tensors['paired_word_mask.pt'], paired) and bool(paired.any()),
            'Invalid paired word mask')
    require(bool((tensors['text_words.pt'][~text_mask] == 0).all()), 'Invalid text must be zero')
    frame_mask = tensors['frame_mask.pt']
    valid_count = int(frame_mask.sum())
    require(valid_count > 0 and torch.equal(frame_mask, torch.arange(t) < valid_count),
            'Valid frames must form a nonempty contiguous prefix')
    validate_timing(metadata['timing']['audio'], valid_count)
    validate_timing(metadata['timing']['ctc'], metadata['timing']['ctc']['valid_frame_count'])
    require(len(metadata['word_windows']) == len(metadata['ctc_word_windows'])
            == len(metadata['text_word_mappings']) == n, 'Word-window or text mapping count differs')
    for i, row in enumerate(metadata['word_windows']):
        ctc = metadata['ctc_word_windows'][i]
        text_mapping = metadata['text_word_mappings'][i]
        require(all(r['word_id'] == i and r['word'] == metadata['words'][i]
                    for r in (row, ctc, text_mapping)), 'Word-window or text mapping IDs differ')
        require(ctc['status'] == row['status'], 'CTC/acoustic statuses differ')
        start, end = tensors['word_frame_intervals.pt'][i].tolist()
        if audio_mask[i]:
            require(0 <= start < end <= valid_count, f'Invalid acoustic interval at word {i}')
            first, last, method = map_frame_window(
                ctc['expanded_start_frame'], ctc['expanded_end_frame_exclusive'],
                metadata['timing']['ctc'], metadata['timing']['audio'])
            require((start, end) == (first, last) and method == row['mapping_method'],
                    f'Acoustic interval disagrees with timing-based mapping at word {i}')
            require(start == row['audio_start_frame'] and end == row['audio_end_frame_exclusive']
                    and row['audio_frame_count'] == end - start and row['status'] == 'aligned',
                    f'Acoustic interval metadata differs at word {i}')
        else:
            require((start, end) == (-1, -1) and row['audio_start_frame'] is None
                    and row['audio_end_frame_exclusive'] is None
                    and row['audio_frame_count'] == 0 and row['status'] == 'no_ctc_units',
                    f'Invalid acoustic sentinel at word {i}')
    expected_counts = {'audio_words': int(audio_mask.sum()), 'text_words': int(text_mask.sum()),
                       'paired_words': int(paired.sum()), 'frames': valid_count}
    require(metadata['valid_counts'] == expected_counts, 'Validity counts differ')
    return metadata, tensors


def build_package(audit):
    """Materialize a new interface from the exact lineage validated by the audit."""
    stages = {name: Path(path) for name, path in audit['stages'].items()}
    config = read_json(Path(audit['feature_root']) / 'config.json')
    frame_meta = read_json(stages['frames'] / 'metadata.json')
    audio_meta = read_json(stages['audio_words'] / 'metadata.json')
    text_meta = read_json(stages['text_words'] / 'metadata.json')
    windows = read_json(stages['windows'] / 'word_windows.json')
    alignment = read_json(stages['alignment'] / 'alignment.json')
    comparison = read_json(stages['comparison'] / 'comparison.json')
    frames = load_tensor(stages['frames'], 'frames.pt').detach().float().contiguous()
    frame_mask = load_tensor(stages['frames'], 'valid_mask.pt')
    pooled, audio_mask, mapping = pool_audio_words(frames, frame_mask, frame_meta, windows)
    reference = load_tensor(stages['audio_words'], 'audio_word_embeddings.pt')
    require(torch.equal(pooled, reference), 'Frame means do not reproduce source word vectors')
    text = load_tensor(stages['text_words'], 'text_word_embeddings.pt').detach().float().contiguous()
    text_mask = load_tensor(stages['text_words'], 'text_word_mask.pt')
    words = audio_meta['words']
    require(text_meta['words'] == words == windows['words'] == alignment['words']
            == comparison['words'], 'Original word lists differ')
    n = len(words)
    require(text.ndim == 2 and len(text) == n and text_mask.shape == (n,), 'Invalid text dimensions')
    require(len(comparison['rows']) == n and all(r['word_id'] == i and r['word'] == words[i]
            for i, r in enumerate(comparison['rows'])), 'Quality-source word IDs differ')
    intervals = torch.full((n, 2), -1, dtype=torch.int64)
    for i, row in enumerate(mapping['word_windows']):
        if audio_mask[i]:
            intervals[i] = torch.tensor([row['audio_start_frame'], row['audio_end_frame_exclusive']])
    # Invalid source padding may contain nonfinite placeholders; no invalid
    # acoustic frame participates in word pooling or survives into the package.
    frames = frames.masked_fill(~frame_mask[:, None], 0)
    tensors = {'frames.pt': frames, 'frame_mask.pt': frame_mask,
               'word_ids.pt': torch.arange(n), 'word_frame_intervals.pt': intervals,
               'audio_word_mask.pt': audio_mask, 'text_words.pt': text,
               'text_word_mask.pt': text_mask, 'paired_word_mask.pt': audio_mask & text_mask,
               'word_quality.pt': torch.empty(n, 0),
               'word_quality_observed.pt': torch.empty(n, 0, dtype=torch.bool),
               'recording_quality.pt': torch.empty(0),
               'recording_quality_observed.pt': torch.empty(0, dtype=torch.bool)}
    schema = {'format': FORMAT, 'models': audit['models'],
              'audio_width': frames.shape[1], 'text_width': text.shape[1],
              'acoustic': {'layer': config.get('acoustic_layer', 'last_hidden_state'),
                           'context': config.get('acoustic_context', 'whole_recording'),
                           'precision': 'float32'},
              'text_encoding': {'pooling': text_meta['pooling'], 'chunk_size': text_meta['chunk_size'],
                                'tokenizer_sha256': text_meta['tokenizer_sha256']},
              'windows': {'context_frames': windows['context_frames'],
                          'interval_convention': 'start_inclusive_end_exclusive'},
              'quality': {'status': 'pending', 'word_features': [], 'recording_features': []}}
    partition, uid = audit['recording_key'].split('/')
    metadata = {'format': FORMAT, 'uid': uid, 'partition': partition, 'words': words,
                'schema': schema, 'schema_sha256': schema_fingerprint(schema),
                'dimensions': {'frames': len(frames), 'words': n, 'audio': frames.shape[1],
                               'text': text.shape[1]},
                'valid_counts': {'frames': int(frame_mask.sum()), 'audio_words': int(audio_mask.sum()),
                                 'text_words': int(text_mask.sum()),
                                 'paired_words': int((audio_mask & text_mask).sum())},
                'mask_policy': 'audio AND text; all original word rows retained',
                'timing': {'audio': frame_meta['timing'], 'ctc': windows['timing']},
                'word_windows': mapping['word_windows'], 'ctc_word_windows': windows['word_windows'],
                'text_word_mappings': text_meta['word_mappings'], 'text_chunks': text_meta['chunks'],
                'quality_sources': {
                    'status': 'raw_evidence_only; not normalized gate inputs',
                    'comparison_rows': comparison['rows'],
                    'ctc_units': alignment['units'],
                    'ctc_confidence_definition': alignment['confidence_definition']},
                'provenance': audit,
                'compatibility': {'mean_pool_max_abs_error': float((pooled - reference).abs().max()),
                                  'text_vectors': 'unchanged; float32', 'encoders_rerun': False}}
    require(torch.equal(mean_word_vectors(tensors), reference), 'Packaged means differ from source')
    return metadata, tensors


def package_fusion_frames(feature_root, uid, output_dir, partition='train'):
    """Publish a validated independent package, refusing every existing target."""
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Export already exists: {output}. Choose a new directory.')
    audit = audit_recording(feature_root, uid, partition)
    metadata, tensors = build_package(audit)
    # Validate all sources before creating any export directories, and detect
    # input changes between the audit and materialization.
    for path, checksum in audit['verified_sha256'].items():
        require(digest(path) == checksum, f'Audited source changed while packaging: {path}')
    json.dumps(metadata, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fusion-package-', dir=output.parent) as temporary:
        staging = Path(temporary)
        metadata['tensors'] = {}
        for name, tensor in tensors.items():
            torch.save(tensor, staging / name)
            metadata['tensors'][name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                         'sha256': digest(staging / name)}
        (staging / 'metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        read_fusion_package(staging)
        output.mkdir(exist_ok=False)
        # Link newly serialized staging files, never original source files.
        # Publishing metadata last marks completion. An interrupted publication
        # has no metadata and cannot be mistaken for a complete package.
        for name in [*TENSOR_DTYPES, 'metadata.json']:
            os.link(staging / name, output / name)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', type=Path, required=True)
    parser.add_argument('--uid', required=True)
    parser.add_argument('--partition', choices=('train', 'test'), default='train')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for one recording.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = package_fusion_frames(args.feature_root, args.uid, args.output_dir, args.partition)
    print(f'Packaged {metadata["partition"]}/{metadata["uid"]}: {metadata["dimensions"]}')
    print(f'Mean-pool reconstruction error: {metadata["compatibility"]["mean_pool_max_abs_error"]}')
    print(f'Quality status: pending; package: {args.output_dir}')


if __name__ == '__main__':
    main()
