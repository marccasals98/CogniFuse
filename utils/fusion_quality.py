"""Versioned measurement-quality definitions; no labels or neural inference."""

import math

import torch


WORD_FEATURES = ['asr_confidence', 'ctc_emission_confidence',
                 'boundary_start_disagreement_seconds', 'boundary_end_disagreement_seconds']
RECORDING_FEATURES = ['snr_proxy_db', 'clipping_ratio', 'spectral_flatness_noise_proxy']


def quality_schema():
    return {'status': 'ready', 'version': 'fusion_quality_v1',
            'word_features': list(WORD_FEATURES), 'recording_features': list(RECORDING_FEATURES),
            'word_definition': 'Whisper probability; exp(frame-weighted character log emission); '
                               'absolute unexpanded CTC/Whisper start and end differences',
            'recording_estimator': {'version': 'energy_quiet_spectrum_v1', 'window_seconds': 0.02,
                'minimum_frames': 10, 'quiet_fraction': 0.1, 'minimum_quiet_frames': 2,
                'power_floor': 1e-12, 'snr_max_db': 60.0, 'clipping_threshold': 0.999,
                'snr_definition': '10log10(p90/p10) of DC-centered mono frame power, clipped to [0,60]',
                'noise_definition': 'spectral flatness of mean quiet-frame Hann-window power spectrum, excluding DC',
                'clipping_definition': 'fraction of original channel samples with abs(sample)>=0.999 before mixing or gain changes'}}


def quality_dimensions(schema):
    if schema == {'status': 'pending', 'word_features': [], 'recording_features': []}:
        return 0, 0
    if schema != quality_schema():
        raise ValueError('Unsupported quality schema, feature order, or estimator version')
    return len(WORD_FEATURES), len(RECORDING_FEATURES)


def word_quality(metadata):
    """Join evidence by original ID; missing values use zero with observed=False."""
    words = metadata['words']
    rows = metadata['quality_sources']['comparison_rows']
    if len(rows) != len(words):
        raise ValueError('Quality-source word count differs')
    groups = [[] for _ in words]
    frame_count = metadata['timing']['ctc']['valid_frame_count']
    for unit in metadata['quality_sources']['ctc_units']:
        owner = unit['word_id']
        if type(owner) is not int or not -1 <= owner < len(words):
            raise ValueError('Invalid CTC quality word ID')
        if owner == -1:
            continue
        start, end, score = unit['start_frame'], unit['end_frame_exclusive'], unit['mean_log_emission']
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= frame_count
                or not math.isfinite(score) or score > 1e-6):
            raise ValueError('Invalid CTC confidence evidence')
        groups[owner].append((end - start, score))
    values = torch.zeros(len(words), len(WORD_FEATURES), dtype=torch.float32)
    observed = torch.zeros_like(values, dtype=torch.bool)
    for i, (word, row, units) in enumerate(zip(words, rows, groups)):
        if row['word_id'] != i or row['word'] != word:
            raise ValueError('Quality-source original word IDs differ')
        probability = row.get('whisper_probability')
        if probability is not None:
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid Whisper confidence')
            values[i, 0], observed[i, 0] = probability, True
        if units:
            values[i, 1] = math.exp(sum(n * score for n, score in units) / sum(n for n, _ in units))
            observed[i, 1] = True
        aligned = metadata['word_windows'][i]['status'] == 'aligned'
        if bool(units) != aligned:
            raise ValueError('CTC quality evidence disagrees with word alignment')
        for column, name in ((2, 'delta_start'), (3, 'delta_end')):
            value = row.get(name)
            if value is not None:
                if not aligned or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid boundary disagreement evidence')
                values[i, column], observed[i, column] = value, True
    return values, observed


def recording_quality(waveform, sample_rate):
    """Transparent energy/noise proxies, not calibrated SNR or reverberation."""
    if (waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1
            or not waveform.is_floating_point() or not bool(torch.isfinite(waveform).all())
            or type(sample_rate) is not int or sample_rate < 100):
        raise ValueError('Expected finite, nonempty [channels,samples] audio and sample rate')
    config = quality_schema()['recording_estimator']
    raw = waveform.detach().double().cpu()
    values = torch.zeros(len(RECORDING_FEATURES), dtype=torch.float32)
    observed = torch.zeros(len(RECORDING_FEATURES), dtype=torch.bool)
    values[1] = (raw.abs() >= config['clipping_threshold']).double().mean()
    observed[1] = True
    mono = raw.mean(dim=0)
    mono = mono - mono.mean()
    width = round(sample_rate * config['window_seconds'])
    count = len(mono) // width
    diagnostics = {'sample_rate': sample_rate, 'channels': len(raw), 'samples': raw.shape[1],
                   'complete_frames': count, 'window_samples': width}
    if count < config['minimum_frames']:
        diagnostics['missing_reason'] = 'too_few_complete_frames_for_energy_or_noise'
        return values, observed, diagnostics
    frames = mono[:count * width].reshape(count, width)
    power = frames.square().mean(dim=1)
    low, high = torch.quantile(power, torch.tensor([0.1, 0.9], dtype=torch.float64))
    floor = config['power_floor']
    if high > floor:
        values[0] = (10 * torch.log10(high.clamp_min(floor) / low.clamp_min(floor))).clamp(
            0, config['snr_max_db'])
        observed[0] = True
    quiet_count = max(config['minimum_quiet_frames'], math.ceil(count * config['quiet_fraction']))
    quiet = frames[power.argsort()[:quiet_count]]
    if quiet.square().mean() > floor:
        window = torch.hann_window(width, periodic=False, dtype=torch.float64)
        spectrum = torch.fft.rfft(quiet * window).abs().square().mean(dim=0)[1:]
        spectrum = spectrum + floor
        values[2] = (spectrum.log().mean().exp() / spectrum.mean()).clamp(0, 1)
        observed[2] = True
    diagnostics.update(quiet_frames=quiet_count, p10_power=float(low), p90_power=float(high))
    return values, observed, diagnostics


def validate_quality(metadata, tensors):
    q, r = quality_dimensions(metadata['schema']['quality'])
    for name, flag in (('word_quality.pt', 'word_quality_observed.pt'),
                       ('recording_quality.pt', 'recording_quality_observed.pt')):
        if bool((tensors[name][~tensors[flag]] != 0).any()):
            raise ValueError('Missing quality values must use zero placeholders')
    if not q:
        return
    values, observed = word_quality(metadata)
    if (not torch.equal(observed, tensors['word_quality_observed.pt'])
            or not torch.allclose(values, tensors['word_quality.pt'], rtol=1e-6, atol=1e-7)):
        raise ValueError('Word quality tensors disagree with original evidence')
    w = tensors['word_quality.pt']
    rec = tensors['recording_quality.pt']
    if (bool((w[:, :2] < 0).any()) or bool((w[:, :2] > 1).any()) or bool((w[:, 2:] < 0).any())
            or rec.shape != (r,) or bool((rec < 0).any()) or rec[0] > 60 or bool((rec[1:] > 1).any())):
        raise ValueError('Quality values outside declared ranges')
