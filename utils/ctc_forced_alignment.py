"""Force-align cached CTC emissions to their original transcript targets."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio

from utils.audio_frame_timing import frame_interval_seconds, frame_timing
from utils.transcript_normalization import normalize_ctc_words


def align_ctc_units(log_probs, valid_mask, transcript, metadata):
    """Return a best CTC path and one span per target, including delimiters.

    Frame spans are [start_frame, end_frame_exclusive). Time spans cover their
    convolution receptive fields; adjacent time spans may overlap slightly.
    Blank frames are retained in the path, but are not interpreted as pauses.
    """
    if metadata.get('format') != 'ctc_emissions_v1':
        raise ValueError('Expected Step 3 ctc_emissions_v1 metadata')
    if log_probs.ndim != 3 or log_probs.shape[0] != 1:
        raise ValueError('Expected CTC log probabilities shaped [1, T, V]')
    if list(log_probs.shape) != metadata['logits_shape']:
        raise ValueError('CTC tensor shape differs from metadata')
    if valid_mask.dtype != torch.bool or valid_mask.shape != log_probs.shape[:2]:
        raise ValueError('Expected a boolean valid-frame mask shaped [1, T]')
    valid_count = int(valid_mask.sum())
    expected_mask = torch.arange(log_probs.shape[1], device=valid_mask.device)[None] < valid_count
    if valid_count == 0 or not torch.equal(valid_mask, expected_mask):
        raise ValueError('Valid frames must form a nonempty contiguous prefix')
    timing = metadata['timing']
    canonical_timing = frame_timing(
        SimpleNamespace(conv_kernel=timing['conv_kernel'], conv_stride=timing['conv_stride']),
        timing['num_samples'], timing['sample_rate'], valid_count,
    )
    if any(timing.get(key) != value for key, value in canonical_timing.items()):
        raise ValueError('Timing metadata does not match the valid convolution grid')
    blank, vocabulary = metadata['blank_id'], metadata['vocabulary']
    if set(vocabulary.values()) != set(range(log_probs.shape[-1])):
        raise ValueError('CTC vocabulary differs from the emission dimension')
    if not isinstance(blank, int) or not 0 <= blank < log_probs.shape[-1]:
        raise ValueError('Invalid CTC blank index')
    normalized = normalize_ctc_words(transcript['words'], vocabulary, blank,
                                     transcript['word_delimiter'])
    if any(transcript.get(key) != value for key, value in normalized.items()):
        raise ValueError('Saved transcript targets or word mapping are inconsistent')
    targets = transcript['target_ids']
    if not targets:
        raise ValueError('Cannot align a transcript with no CTC target units')
    minimum_frames = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
    if minimum_frames > valid_count:
        raise ValueError('Insufficient valid frames, including repeated-character blanks')
    emissions = log_probs[:, :valid_count].detach().float().cpu().contiguous()
    if not torch.isfinite(emissions).all():
        raise ValueError('Valid CTC frames contain nonfinite log probabilities')
    if not torch.allclose(emissions.logsumexp(-1), torch.zeros(1, valid_count), atol=1e-4, rtol=0):
        raise ValueError('Expected normalized log probabilities, not raw logits')
    path, scores = torchaudio.functional.forced_align(
        emissions, torch.tensor([targets], dtype=torch.long), blank=blank,
    )
    spans = torchaudio.functional.merge_tokens(path[0], scores[0], blank=blank)
    if [span.token for span in spans] != targets:
        raise RuntimeError('CTC path does not collapse to the exact transcript targets')
    if not torch.isfinite(scores).all():
        raise RuntimeError('Forced alignment returned nonfinite scores')
    units, previous_end = [], 0
    for index, span in enumerate(spans):
        if not previous_end <= span.start < span.end <= valid_count:
            raise RuntimeError('CTC token spans are not monotonic valid-frame intervals')
        previous_end = span.end
        units.append({
            'target_index': index, 'token_id': int(span.token),
            'token': transcript['tokens'][index],
            'word_id': transcript['token_word_ids'][index],
            'start_frame': span.start, 'end_frame_exclusive': span.end,
            'start_time': frame_interval_seconds(span.start, timing)[0],
            'end_time': frame_interval_seconds(span.end - 1, timing)[1],
            'mean_log_emission': span.score,
            'emission_confidence': math.exp(span.score),
        })
    alignment = {
        'format': 'ctc_unit_alignment_v1',
        'backend': 'torchaudio.functional.forced_align',
        'torchaudio_version': torchaudio.__version__,
        'ctc_model': metadata['ctc_model'],
        'model_revision': metadata.get('model_revision'),
        'audio_source': metadata['audio_source'],
        'word_source': metadata['word_source'],
        'timing': timing, 'blank_id': blank,
        'words': transcript['words'], 'word_mappings': transcript['word_mappings'],
        'units': units, 'aligned_frames': valid_count,
        'path_log_score': scores.double().sum().item(),
        'frame_span_convention': 'start_inclusive_end_exclusive',
        'time_span_definition': 'union_of_convolution_receptive_fields',
        'confidence_definition': 'exp(mean log emission over nonblank token frames); not a boundary posterior',
        'word_intervals_computed': False,
    }
    return alignment, path, scores


def export_alignment(ctc_dir, output_dir):
    """Read an existing Step 3 export and write a new, separate alignment."""
    ctc_dir, output_dir = Path(ctc_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Export already exists: {output_dir}. Choose a new directory.')
    metadata = json.loads((ctc_dir / 'metadata.json').read_text(encoding='utf-8'))
    transcript = json.loads((ctc_dir / 'transcript.json').read_text(encoding='utf-8'))
    log_probs = torch.load(ctc_dir / 'ctc_log_probs.pt', map_location='cpu', weights_only=True)
    valid_mask = torch.load(ctc_dir / 'valid_frame_mask.pt', map_location='cpu', weights_only=True)
    alignment, path, scores = align_ctc_units(log_probs, valid_mask, transcript, metadata)
    alignment['ctc_export'] = str(ctc_dir.resolve())
    alignment['input_sha256'] = {
        name: hashlib.sha256((ctc_dir / name).read_bytes()).hexdigest()
        for name in ('metadata.json', 'transcript.json', 'ctc_log_probs.pt', 'valid_frame_mask.pt')
    }
    serialized = json.dumps(alignment, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(path, output_dir / 'path.pt')
    torch.save(scores, output_dir / 'path_log_probs.pt')
    (output_dir / 'alignment.json').write_text(serialized, encoding='utf-8')
    return alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ctc-dir', type=Path, required=True, help='Existing Step 3 export directory.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; never overwrite existing exports.')
    args = parser.parse_args()
    alignment = export_alignment(args.ctc_dir, args.output_dir)
    print(f"Aligned {len(alignment['units'])} CTC units across {alignment['aligned_frames']} valid frames")
    print(f"Preserved {len(alignment['words'])} original word indices; saved to {args.output_dir}")
    print('Word interval merging is Step 5; alignment quality still needs inspection.')


if __name__ == '__main__':
    main()
