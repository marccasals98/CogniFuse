"""Prepare Spanish CTC emissions and word-mapped targets for one recording."""

import argparse
import csv
import json
from pathlib import Path

import torch
import torchaudio
import transformers

from scripts.ctc_encoder import DEFAULT_CTC_MODEL, SpanishCTCEncoder
from utils.ctc_skipped_words import skip_metadata


def read_words(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if 'word' not in (reader.fieldnames or []):
            raise ValueError('Expected the existing Whisper word CSV with a word column')
        words = [row['word'] for row in reader]
    if not words:
        raise ValueError('Word CSV is empty; CTC alignment needs a transcript')
    return words


def source_identity(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def export_ctc(audio_path, words_path, output_dir, encoder):
    """Create emissions and targets only; do not infer word boundaries yet."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Export already exists: {output_dir}. Choose a new directory.')
    audio_source, word_source = source_identity(audio_path), source_identity(words_path)
    transcript = encoder.normalize_words(read_words(words_path))
    if not transcript['target_ids']:
        raise ValueError('Transcript has no alignable CTC characters')
    waveform, source_rate = torchaudio.load(audio_path)
    source_samples = waveform.shape[-1]
    waveform = torchaudio.functional.resample(waveform.mean(dim=0), source_rate, 16000)
    emissions = encoder(waveform)
    targets = transcript['target_ids']
    minimum_frames = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
    if minimum_frames > emissions['timing']['frame_count']:
        raise ValueError('Transcript needs more CTC frames than the audio provides')
    metadata = {
        **skip_metadata(transcript),
        'format': 'ctc_emissions_v1', 'ctc_model': encoder.model_id,
        'model_revision': getattr(encoder.model.config, '_commit_hash', None),
        'model_config': encoder.model.config.to_dict(),
        'processor_config': encoder.processor.feature_extractor.to_dict(),
        'versions': {'torch': torch.__version__, 'torchaudio': torchaudio.__version__,
                     'transformers': transformers.__version__},
        'audio_source': audio_source, 'word_source': word_source,
        'source_sample_rate': source_rate, 'source_num_samples': source_samples,
        'timing': emissions['timing'],
        'logits_shape': list(emissions['ctc_logits'].shape),
        'vocabulary': encoder.vocabulary, 'blank_id': encoder.blank_id,
        'minimum_target_frames': minimum_frames,
        'alignment_computed': False,
        'ctc_logits_file': 'ctc_logits.pt',
        'ctc_log_probs_file': 'ctc_log_probs.pt',
        'valid_frame_mask_file': 'valid_frame_mask.pt',
        'transcript_file': 'transcript.json',
    }
    encoded_metadata = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    encoded_transcript = json.dumps(transcript, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('ctc_logits', 'ctc_log_probs', 'valid_frame_mask'):
        torch.save(emissions[key], output_dir / (key + '.pt'))
    (output_dir / 'transcript.json').write_text(encoded_transcript, encoding='utf-8')
    (output_dir / 'metadata.json').write_text(encoded_metadata, encoding='utf-8')
    return metadata, transcript


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-path', type=Path, required=True)
    parser.add_argument('--words-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing exports are never overwritten.')
    parser.add_argument('--ctc-model', default=DEFAULT_CTC_MODEL)
    parser.add_argument('--revision', help='Optional checkpoint revision/commit.')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--skip-unsupported-words', action='store_true',
                        help='Omit unsupported words from CTC targets; preserve original IDs as invalid rows.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Export already exists: {args.output_dir}. Choose a new directory.')
    if not args.audio_path.is_file():
        raise FileNotFoundError(args.audio_path)
    read_words(args.words_csv)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    encoder = SpanishCTCEncoder.from_pretrained(
        args.ctc_model, device=device, local_files_only=args.local_files_only, revision=args.revision,
    )
    encoder.skip_unsupported_words = args.skip_unsupported_words
    metadata, transcript = export_ctc(args.audio_path, args.words_csv, args.output_dir, encoder)
    print(f"Saved CTC logits {metadata['logits_shape']} to {args.output_dir}")
    print(f"Preserved {len(transcript['words'])} original words; {len(transcript['target_ids'])} CTC target units")
    unalignable = [word['word_id'] for word in transcript['word_mappings'] if word['status'] != 'ready']
    if unalignable:
        print(f'Words without CTC units (retained in metadata): {unalignable}')
    print('Forced alignment has not been run; it is Step 4.')


if __name__ == '__main__':
    main()
