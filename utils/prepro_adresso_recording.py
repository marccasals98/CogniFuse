"""Validate and export one ADReSSo recording using the existing encoders.

Original transcripts, timestamps, and saved embeddings remain unchanged.
New full-transcript exports carry masks, source hashes, and alignment reports.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

import soundfile as sf
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from scripts.adresso import ADReSSoIndex
from scripts.data import PrecomputedADDataset
from utils.embedding_chunks import encode_full_transcript
from utils.embedding_masks import embedding_masks
from utils.prepro_embeddings import align_audio, build_audio_encoder, extract_audio_features
from utils.word_alignment import frame_bounds, token_audio_intervals

TEXT_MODEL = 'distilbert-base-uncased'
AUDIO_MODEL = 'facebook/wav2vec2-base-960h'


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest()}


def match_words(transcription, rows, duration):
    """Restore case for matching only; never drop words or change timestamps."""
    words, audit, cursor, previous_start = [], [], 0, -math.inf
    for word_id, row in enumerate(rows):
        word = unicodedata.normalize('NFC', row['word']).strip()
        start, end = float(row['start']), float(row['end'])
        if not word or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end <= duration + 0.05:
            raise ValueError(f'Invalid word/timestamp at word ID {word_id}')
        if start < previous_start:
            raise ValueError(f'Nonmonotonic word start at word ID {word_id}; source is not repaired automatically')
        match = re.compile(re.escape(word), re.IGNORECASE).search(transcription, cursor)
        if match is None or any(c.isalnum() for c in transcription[cursor:match.start()]):
            raise ValueError(f'Word ID {word_id} does not match the supplied transcript in order')
        matched = match.group()
        words.append((matched, start, end))
        audit.append({'word_id': word_id, 'original_word': word, 'matched_word': matched,
                      'case_changed': word != matched, 'char_start': match.start(),
                      'char_end': match.end(), 'start': start, 'end': end})
        cursor, previous_start = match.end(), start
    if not words or any(c.isalnum() for c in transcription[cursor:]):
        raise ValueError('Empty word list or transcript words missing from timestamps')
    # Reuse the existing strict correspondence check before loading encoders.
    token_audio_intervals(transcription, words, [], audio_duration=duration)
    return words, audit


def prepare_recording(dataset_root, uid, split='train', pauses=False):
    index = ADReSSoIndex(dataset_root)
    records = index.training if split == 'train' else index.test
    selected = [r for r in records if r['uid'] == uid]
    if len(selected) != 1:
        raise ValueError(f'UID {uid!r} is not present in the requested {split} partition')
    record = selected[0]
    partition = index.root / record['partition']
    transcript_path = partition / 'text/text_transcriptions.csv'
    with transcript_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['uid'] == uid]
    if len(rows) != 1:
        raise ValueError('Expected exactly one transcript row for the selected UID')
    if split == 'train' and rows[0]['diagno'] != record['diagnosis']:
        raise ValueError('Transcript diagnosis disagrees with the audio directory')
    column = 'transcription_pause' if pauses else 'transcription'
    transcription = unicodedata.normalize('NFC', rows[0][column])
    if not transcription.strip():
        raise ValueError('Empty transcript')
    words_path = partition / 'text'
    if split == 'train':
        words_path /= record['diagnosis']
    words_path /= uid + '.csv'
    with words_path.open(encoding='utf-8-sig', newline='') as stream:
        word_rows = list(csv.DictReader(stream))
    info = sf.info(record['audio_path'])
    words, word_audit = match_words(transcription, word_rows, info.duration)
    return {'record': record, 'transcription': transcription, 'transcript_column': column,
            'words': words, 'word_audit': word_audit,
            'audio_info': {'source_sample_rate': info.samplerate, 'source_channels': info.channels,
                           'duration_seconds': info.duration},
            'sources': {'audio': identity(record['audio_path']), 'transcripts': identity(transcript_path),
                        'words': identity(words_path)}}


def export_recording(prepared, output_dir, tokenizer, text_model, audio_encoder,
                     device='cpu', chunk_size=200):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Output already exists: {output_dir}; choose a new directory')
    transcription, words = prepared['transcription'], prepared['words']
    text, offsets, chunks = encode_full_transcript(tokenizer, text_model, transcription, device, chunk_size)
    audio_mask, text_mask = embedding_masks(torch.ones(len(offsets), dtype=torch.bool), offsets)
    frames, frame_rate = extract_audio_features(prepared['record']['audio_path'], 'wav2vec2', audio_encoder, device)
    audio, content_tokens = align_audio(transcription, words, offsets, frames, frame_rate)
    intervals = token_audio_intervals(transcription, words, offsets, prepared['audio_info']['duration_seconds'])
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(transcription, truncation=False, verbose=False)['input_ids'])
    token_audit = []
    for position, (token, offset, interval) in enumerate(zip(tokens, offsets, intervals)):
        pooled = frame_bounds(*interval, frame_rate, len(frames)) if interval is not None else None
        token_audit.append({'position': position, 'token': token, 'char_offsets': offset,
                            'audio_interval': interval, 'frame_range': pooled,
                            'audio_valid': bool(audio_mask[position]), 'text_valid': bool(text_mask[position])})
    if len(tokens) != len(offsets) or audio.shape[0] != text.shape[0]:
        raise ValueError('Audio/text/token positions disagree')
    if not torch.isfinite(audio).all() or not torch.isfinite(text).all():
        raise ValueError('Nonfinite exported vectors')
    if (audio[~audio_mask] != 0).any() or not audio_mask.any() or not text_mask.any():
        raise ValueError('Invalid audio layout or empty masks')
    for source in prepared['sources'].values():
        if identity(source['path']) != source:
            raise ValueError('Source changed during extraction')
    stem = prepared['record']['uid'] + ('distil_pauses' if prepared['transcript_column'] == 'transcription_pause' else 'distil')
    metadata = {'format': 'adresso_full_transcript_v1', 'uid': prepared['record']['uid'],
                'partition': prepared['record']['partition'], 'label': prepared['record']['label'],
                'sources': prepared['sources'], 'audio_info': prepared['audio_info'],
                'text_model': TEXT_MODEL, 'audio_model': AUDIO_MODEL,
                'text_model_revision': getattr(text_model.config, '_commit_hash', None),
                'audio_model_revision': getattr(audio_encoder[1].config, '_commit_hash', None),
                'transcript_column': prepared['transcript_column'], 'chunk_size': chunk_size,
                'chunks': chunks, 'content_tokens': content_tokens, 'truncated_tokens': 0,
                'audio_shape': list(audio.shape), 'text_shape': list(text.shape),
                'audio_valid_positions': int(audio_mask.sum()), 'text_valid_positions': int(text_mask.sum()),
                'audio_frame_rate': frame_rate, 'audio_frame_count': len(frames),
                'alignment': 'Existing ASR word times; legacy nominal frame-rate pooling and short-window expansion',
                'asr_generation_provenance': 'Historical generation settings unavailable; existing text and word CSV reused',
                'case_only_matches': sum(r['case_changed'] for r in prepared['word_audit']),
                'zero_duration_words': sum(start == end for _, start, end in words),
                'word_alignment': prepared['word_audit'], 'token_alignment': token_audit,
                'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
                'scope': 'Single-recording loader compatibility; complete-cohort training readiness is not established'}
    json.dumps(metadata, ensure_ascii=False, allow_nan=False)  # Validate metadata before writing.
    output_dir.mkdir(parents=True, exist_ok=False)
    for suffix, value in (('.pt', text), ('_mask.pt', text_mask), ('_audio.pt', audio), ('_audio_mask.pt', audio_mask)):
        torch.save(value, output_dir / (stem + suffix))
    # Check the saved files through the actual classifier loader's validators.
    loaded_text = PrecomputedADDataset._load_features(output_dir / (stem + '.pt'))
    loaded_audio = PrecomputedADDataset._load_features(output_dir / (stem + '_audio.pt'))
    loaded_tm = PrecomputedADDataset._load_mask(output_dir / (stem + '_mask.pt'), len(loaded_text))
    loaded_am = PrecomputedADDataset._load_mask(output_dir / (stem + '_audio_mask.pt'), len(loaded_audio))
    torch.testing.assert_close(loaded_text, text, rtol=0, atol=0)
    torch.testing.assert_close(loaded_audio, audio, rtol=0, atol=0)
    if not torch.equal(loaded_tm, text_mask) or not torch.equal(loaded_am, audio_mask):
        raise ValueError('Saved masks differ from validated masks')
    metadata['loader_validation'] = 'passed'
    (output_dir / (stem + '_extraction.json')).write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--uid', required=True)
    parser.add_argument('--split', choices=('train', 'test'), default='train')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pauses', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=200)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.dry_run and (args.output_dir is None or args.output_dir.exists()):
        parser.error('Extraction requires a new --output-dir')
    prepared = prepare_recording(args.dataset_root, args.uid, args.split, args.pauses)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True, local_files_only=args.local_files_only)
    encoded = tokenizer(prepared['transcription'], return_offsets_mapping=True, truncation=False, verbose=False)
    if not tokenizer.num_special_tokens_to_add(pair=False) < args.chunk_size <= tokenizer.model_max_length:
        parser.error('--chunk-size must leave room for content and fit the tokenizer context')
    token_audio_intervals(prepared['transcription'], prepared['words'], encoded['offset_mapping'], prepared['audio_info']['duration_seconds'])
    print(f"Validated {args.uid}: {len(prepared['words'])} words, {len(encoded['input_ids'])} text positions", flush=True)
    if args.dry_run:
        print('Dry run: no encoders loaded and no outputs written', flush=True)
        return
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(4)
    print('Loading English encoders', flush=True)
    text_model = AutoModel.from_pretrained(TEXT_MODEL, local_files_only=args.local_files_only).to(device).eval()
    audio_encoder = build_audio_encoder('wav2vec2', device, args.local_files_only)
    print('Extracting complete recording and transcript', flush=True)
    metadata = export_recording(prepared, args.output_dir, tokenizer, text_model, audio_encoder, device, args.chunk_size)
    print(f"Saved {metadata['audio_shape']} audio and {metadata['text_shape']} text; loader validation passed: {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
