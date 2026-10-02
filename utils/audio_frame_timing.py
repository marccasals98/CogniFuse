"""Timing for unpadded, whole-recording Wav2Vec2 feature sequences."""

import math


def frame_timing(config, num_samples, sample_rate, frame_count):
    """Derive the convolution grid and verify the observed sequence length.

    Intervals describe convolution receptive fields, not the complete context
    used by the transformer. Ends are exclusive; timestamps are in seconds
    from the beginning of the original recording.
    """
    if getattr(config, 'add_adapter', False):
        raise ValueError('Frame timing does not yet support temporal adapters')
    kernels, strides = list(config.conv_kernel), list(config.conv_stride)
    if not kernels or len(kernels) != len(strides):
        raise ValueError('Expected matching convolution kernels and strides')
    if min(num_samples, sample_rate, frame_count) <= 0:
        raise ValueError('Audio and frame sequence must be nonempty')
    hop, receptive_field, expected = 1, 1, num_samples
    for kernel, stride in zip(kernels, strides):
        if kernel <= 0 or stride <= 0:
            raise ValueError('Convolution kernels and strides must be positive')
        receptive_field += (kernel - 1) * hop
        hop *= stride
        expected = (expected - kernel) // stride + 1
    if expected != frame_count:
        raise ValueError(f'Expected {expected} unpadded frames, received {frame_count}')
    return {
        'sample_rate': sample_rate,
        'num_samples': num_samples,
        'audio_duration_seconds': num_samples / sample_rate,
        'frame_count': frame_count,
        'frame_stride_samples': hop,
        'frame_stride_seconds': hop / sample_rate,
        'receptive_field_samples': receptive_field,
        'receptive_field_seconds': receptive_field / sample_rate,
        'first_frame_start_seconds': 0.0,
        'first_frame_center_seconds': receptive_field / (2 * sample_rate),
        'timestamp_reference': 'original_recording_seconds',
        'interval_convention': 'start_inclusive_end_exclusive',
        'padding': 'none',
        'valid_frame_count': frame_count,
        'conv_kernel': kernels,
        'conv_stride': strides,
    }


def frame_interval_seconds(frame_index, timing):
    """Return the convolution interval [start, end) for a valid frame."""
    if not isinstance(frame_index, int) or not 0 <= frame_index < timing['frame_count']:
        raise ValueError('Frame index is outside the valid sequence')
    start_samples = frame_index * timing['frame_stride_samples']
    return (start_samples / timing['sample_rate'],
            (start_samples + timing['receptive_field_samples']) / timing['sample_rate'])


def time_to_frame_index(seconds, timing):
    """Find the nearest frame center, clipping recording edges to valid frames.

    Times outside the recording are rejected. Exact ties choose the later
    frame. Use frame_interval_seconds when convolution support is needed.
    """
    if not math.isfinite(seconds) or not 0 <= seconds <= timing['audio_duration_seconds']:
        raise ValueError('Time is outside the recording')
    index = math.floor(
        (seconds * timing['sample_rate'] - timing['receptive_field_samples'] / 2)
        / timing['frame_stride_samples'] + 0.5
    )
    return max(0, min(timing['frame_count'] - 1, index))
