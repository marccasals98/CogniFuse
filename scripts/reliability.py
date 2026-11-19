"""Training-split-only quality statistics and masked, reproducible transforms."""

import math
from pathlib import Path

import torch

from utils.fusion_quality import WORD_FEATURES, RECORDING_FEATURES
from utils.package_fusion_frames import digest, read_fusion_package, schema_fingerprint

OUTPUT_FEATURES = [*WORD_FEATURES, *RECORDING_FEATURES,
                   *['observed/' + name for name in WORD_FEATURES + RECORDING_FEATURES]]


def validate_quality_state(state, schema_sha256):
    if (state['format'] != 'frame_quality_standardization_v1'
            or state['fit_split'] != 'train'
            or state['schema_sha256'] != schema_sha256
            or state['word_features'] != WORD_FEATURES or state['recording_features'] != RECORDING_FEATURES
            or state['output_features'] != OUTPUT_FEATURES
            or len(state['fit_uids']) != len(set(state['fit_uids']))
            or set(state['fit_uids']) != set(state['fit_metadata_sha256'])):
        raise ValueError('Quality normalization state/schema differs')
    for group, names in (('word', WORD_FEATURES), ('recording', RECORDING_FEATURES)):
        stats = state[group]
        if any(len(stats[name]) != len(names) for name in ('mean', 'scale', 'observations')):
            raise ValueError('Invalid quality normalization dimensions')
        if (any(not math.isfinite(x) for x in stats['mean'])
                or any(not math.isfinite(x) or x <= 0 for x in stats['scale'])
                or any(type(x) is not int or x <= 0 for x in stats['observations'])):
            raise ValueError('Invalid quality normalization statistics')


def fit_quality_normalization(dataset):
    """Inspect only records selected by the existing training-fold dataset."""
    if dataset.split != 'train' or dataset.schema['quality']['status'] != 'ready':
        raise ValueError('Quality normalization can only be fitted on ready training data')
    moments = {group: [torch.zeros(len(names), dtype=torch.float64) for _ in range(3)]
               for group, names in (('word', WORD_FEATURES), ('recording', RECORDING_FEATURES))}
    receipts = {}
    for record in dataset.segments:
        uid, path = record['uid'], Path(record['package_path'])
        if uid in receipts:
            raise ValueError('Duplicate training UID during normalization')
        metadata, tensors = read_fusion_package(path)
        if (metadata['partition'] != 'train' or metadata['uid'] != uid
                or metadata['schema_sha256'] != dataset.schema_sha256):
            raise ValueError('Normalization training package identity/schema differs')
        receipts[uid] = digest(path / 'metadata.json')
        for group in ('word', 'recording'):
            values = tensors[group + '_quality.pt'].double()
            observed = tensors[group + '_quality_observed.pt']
            if group == 'word':
                observed = observed & tensors['paired_word_mask.pt'][:, None]
            else:
                values, observed = values[None], observed[None]
            clean = values.masked_fill(~observed, 0)
            count, total, squared = moments[group]
            count.add_(observed.sum(dim=0))
            total.add_(clean.sum(dim=0))
            squared.add_(clean.square().sum(dim=0))
    state = {'format': 'frame_quality_standardization_v1', 'schema_sha256': dataset.schema_sha256,
             'fit_split': 'train',
             'word_features': list(WORD_FEATURES), 'recording_features': list(RECORDING_FEATURES),
             'output_features': list(OUTPUT_FEATURES),
             'fit_uids': sorted(receipts), 'fit_metadata_sha256': receipts,
             'imputation': 'training mean, equivalent to normalized zero',
             'constant_scale': 'population std below 1e-6 replaced by 1'}
    for group, names in (('word', WORD_FEATURES), ('recording', RECORDING_FEATURES)):
        count, total, squared = moments[group]
        if bool((count == 0).any()):
            missing = [name for name, c in zip(names, count) if c == 0]
            raise ValueError(f'No observed training values for {group} quality: {missing}')
        mean = total / count
        scale = (squared / count - mean.square()).clamp_min(0).sqrt()
        scale[scale < 1e-6] = 1
        state[group] = {'mean': mean.tolist(), 'scale': scale.tolist(),
                        'observations': count.long().tolist()}
    validate_quality_state(state, dataset.schema_sha256)
    return state


def validate_training_quality_state(dataset, state):
    validate_quality_state(state, dataset.schema_sha256)
    if set(state['fit_uids']) != {record['uid'] for record in dataset.segments}:
        raise ValueError('Quality normalization was fitted on a different training split')
    for record in dataset.segments:
        if digest(Path(record['package_path']) / 'metadata.json') != state['fit_metadata_sha256'][record['uid']]:
            raise ValueError('Training package changed since quality normalization was fitted')


def transform_quality(tensors, state, schema_sha256):
    validate_quality_state(state, schema_sha256)
    valid = tensors['paired_word_mask.pt']
    normalized, indicators = [], []
    for group in ('word', 'recording'):
        values = tensors[group + '_quality.pt']
        observed = tensors[group + '_quality_observed.pt']
        mean = values.new_tensor(state[group]['mean'])
        scale = values.new_tensor(state[group]['scale'])
        z = ((values - mean) / scale).masked_fill(~observed, 0)
        if group == 'recording':
            z, observed = z.expand(len(valid), -1), observed.expand(len(valid), -1)
        normalized.append(z)
        indicators.append(observed.to(values.dtype))
    # [word z (4), recording z (3), word flags (4), recording flags (3)].
    result = torch.cat([*normalized, *indicators], dim=-1).masked_fill(~valid[:, None], 0)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Nonfinite normalized reliability features')
    return result


def quality_state_fingerprint(state):
    return schema_fingerprint(state)
