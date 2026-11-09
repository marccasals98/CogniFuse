"""Frame-package batching and acoustic mean/attention word pooling."""

from pathlib import Path
import json

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from utils.package_fusion_frames import FORMAT, read_fusion_package, schema_fingerprint


def package_header(directory, uid=None, partition=None, schema=None):
    with (Path(directory) / 'metadata.json').open(encoding='utf-8') as stream:
        metadata = json.load(stream)
    if metadata['format'] != FORMAT or metadata['schema_sha256'] != schema_fingerprint(metadata['schema']):
        raise ValueError('Invalid frame package format/schema')
    if uid is not None and metadata['uid'] != uid:
        raise ValueError('Frame package UID differs from dataset index')
    if partition is not None and metadata['partition'] != partition:
        raise ValueError('Frame package partition differs from dataset split')
    if schema is not None and metadata['schema_sha256'] != schema:
        raise ValueError('Frame package schema differs from training/checkpoint schema')
    return metadata


def frame_item(directory, label, uid=None, partition=None, schema=None):
    package_header(directory, uid, partition, schema)
    metadata, tensors = read_fusion_package(directory)
    return {**{name[:-3]: value for name, value in tensors.items()},
            'uid': metadata['uid'], 'schema_sha256': metadata['schema_sha256'],
            'label': torch.tensor(label, dtype=torch.long)}


def collate_frame_items(items):
    if not items or len({item['schema_sha256'] for item in items}) != 1:
        raise ValueError('Frame batches require a single common package schema')
    batch = {'uids': [item['uid'] for item in items],
             'schema_sha256': items[0]['schema_sha256'],
             'labels': torch.stack([item['label'] for item in items])}
    for key in ('frames', 'frame_mask', 'word_ids', 'word_frame_intervals',
                'audio_word_mask', 'text_words', 'text_word_mask', 'paired_word_mask',
                'word_quality', 'word_quality_observed'):
        batch[key] = pad_sequence([item[key] for item in items], batch_first=True,
                                  padding_value=-1 if key in ('word_ids', 'word_frame_intervals') else 0)
    for key in ('recording_quality', 'recording_quality_observed'):
        batch[key] = torch.stack([item[key] for item in items])
    return batch


def frame_batch_to_device(batch, device):
    """Keep IDs/labels out of model inputs and preserve mask/index dtypes."""
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items() if key not in ('uids', 'labels')}


def valid_word_windows(frames, intervals, audio_mask, frame_mask):
    """Yield valid acoustic windows, skipping sentinel/padded word rows."""
    if (frames.ndim != 3 or intervals.ndim != 3 or intervals.shape[-1] != 2
            or tuple(audio_mask.shape) != tuple(intervals.shape[:2])
            or tuple(frame_mask.shape) != tuple(frames.shape[:2])
            or frames.shape[0] != intervals.shape[0]
            or intervals.dtype != torch.int64 or audio_mask.dtype != torch.bool
            or frame_mask.dtype != torch.bool):
        raise ValueError('Invalid frame pooling shapes/dtypes')
    indices = intervals.detach().cpu().tolist()
    valid_words = audio_mask.detach().cpu().tolist()
    for b, mask in enumerate(valid_words):
        for i, valid in enumerate(mask):
            if not valid:
                continue
            start, end = indices[b][i]
            if not 0 <= start < end <= frames.shape[1] or not bool(frame_mask[b, start:end].all()):
                raise ValueError('Word interval includes invalid or out-of-range frames')
            yield b, i, start, end


class MeanAcousticWordPooling(nn.Module):
    """Average raw frames inside valid intervals before normalization/adapters."""

    def forward(self, frames, intervals, audio_mask, frame_mask):
        output = frames.new_zeros(*audio_mask.shape, frames.shape[-1])
        # Preserve the old per-word reduction order for exact baseline parity.
        for b, i, start, end in valid_word_windows(frames, intervals, audio_mask, frame_mask):
            output[b, i] = frames[b, start:end].mean(dim=0)
        return output


class AttentionAcousticWordPooling(nn.Module):
    """Learn acoustic frame weights within each fixed, aligned word window."""

    def __init__(self, audio_dim, hidden_dim=64):
        super().__init__()
        if type(audio_dim) is not int or audio_dim <= 0 or type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError('Acoustic attention dimensions must be positive integers')
        self.audio_dim = audio_dim
        # The output bias would cancel in the within-word softmax.
        self.scorer = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, 1, bias=False))

    def forward(self, frames, intervals, audio_mask, frame_mask, return_attention=False):
        if frames.ndim != 3 or frames.shape[-1] != self.audio_dim:
            raise ValueError('Acoustic frame width differs from attention configuration')
        output = frames.new_zeros(*audio_mask.shape, frames.shape[-1])
        diagnostics = []
        for b, i, start, end in valid_word_windows(frames, intervals, audio_mask, frame_mask):
            window = frames[b, start:end]
            scores = self.scorer(window).squeeze(-1)
            # Normalize in float32 for reduced-precision inputs; preserve double
            # precision for numerical checks. Values remain raw encoder vectors.
            weights = torch.softmax(scores.float() if scores.dtype in (torch.float16, torch.bfloat16)
                                    else scores, dim=0)
            output[b, i] = (weights[:, None] * window).sum(dim=0).to(frames.dtype)
            if return_attention:
                diagnostics.append({'batch_index': b, 'word_index': i,
                                    'start_frame': start, 'end_frame_exclusive': end,
                                    'weights': weights.detach()})
        return (output, diagnostics) if return_attention else output
