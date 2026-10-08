"""Extract paired embeddings for complete recordings using bounded text chunks."""

import argparse
import json
from pathlib import Path
import unicodedata

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor

if __package__:
    from .embedding_chunks import encode_full_transcript
    from .embedding_masks import embedding_masks
    from .word_alignment import frame_bounds, token_audio_intervals
else:
    from embedding_chunks import encode_full_transcript
    from embedding_masks import embedding_masks
    from word_alignment import frame_bounds, token_audio_intervals


TEXT_MODELS = {
    'bert': ('bert-base-uncased', ''),
    'distilbert': ('distilbert-base-uncased', 'distil'),
    'roberta': ('roberta-base', 'roberta'),
    'stella': ('NovaSearch/stella_en_1.5B_v5', 'stella'),
    'mistral': ('mistralai/Mistral-7B-v0.1', 'mistral'),
    'qwen': ('Qwen/Qwen2.5-7B', 'qwen'),
}


def build_audio_encoder(audio_model, device, local_files_only):
    if audio_model == 'wav2vec2':
        processor = Wav2Vec2Processor.from_pretrained(
            'facebook/wav2vec2-base-960h', local_files_only=local_files_only,
        )
        model = Wav2Vec2Model.from_pretrained(
            'facebook/wav2vec2-base-960h', local_files_only=local_files_only,
        ).to(device).eval()
        return processor, model
    if audio_model == 'egemaps':
        import opensmile
        return opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return None


def extract_audio_features(path, audio_model, encoder, device):
    if audio_model == 'wav2vec2':
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform).squeeze(0)
        processor, model = encoder
        inputs = processor(waveform, sampling_rate=16000, return_tensors='pt').to(device)
        with torch.no_grad():
            features = model(**inputs).last_hidden_state[0].cpu()
        frame_rate = 50
    else:
        y, sr = librosa.load(path)
        if audio_model == 'egemaps':
            size = int(0.1 * sr)
            frames = librosa.util.frame(y, frame_length=size, hop_length=size).T
            features = torch.tensor(np.vstack([encoder.process_signal(frame, sr) for frame in frames])).float()
            frame_rate = 10
        else:
            size = int(0.02 * sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=size, hop_length=size, n_mels=80)
            features = torch.tensor(mel).float().T
            frame_rate = 50
    if torch.isnan(features).any():
        features = torch.nan_to_num(features, nan=0.0)
    if not len(features) or not torch.isfinite(features).all():
        raise ValueError(f'Invalid audio features: {path}')
    return features, frame_rate


def align_audio(transcription, words, offsets, features, frame_rate):
    intervals = token_audio_intervals(
        transcription, words, offsets, audio_duration=len(features) / frame_rate,
    )
    aligned = torch.zeros(len(offsets), features.shape[1])
    aligned[0] = features.mean(dim=0)
    for i, interval in enumerate(intervals):
        if interval is not None:
            first, last = frame_bounds(*interval, frame_rate, len(features))
            aligned[i] = features[first:last].mean(dim=0).clamp(-1e3, 1e3)
    expected = sum(start != end for start, end in offsets)
    count = sum(interval is not None for interval in intervals)
    if count != expected or not torch.isfinite(aligned).all():
        raise ValueError('Invalid aligned audio embeddings')
    return aligned, count


def preprocess_text(args):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    root_path = args.dataset_path / 'WAB_samples'
    preprocessing_path = args.dataset_path / 'preprocessing'
    embeddings_dir = args.embeddings_dir or preprocessing_path / 'embeddings_full'
    df = pd.read_csv(preprocessing_path / 'transcriptions.csv', dtype={'uid': str}, keep_default_na=False)
    if df.uid.duplicated().any():
        raise ValueError('Transcript UIDs must be unique')
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('limit must be positive')
        df = df.iloc[:args.limit]
    model_id, text_prefix = TEXT_MODELS[args.text_model]
    text_prefix += '_pauses' if args.pauses else ''
    audio_suffix = {'wav2vec2': '_audio', 'egemaps': '_egemaps', 'mel': '_mel'}[args.audio_model]
    row_column = 'transcription_pause' if args.pauses else 'transcription'

    # Check all inputs and output conflicts before loading models or writing.
    audio_paths = {}
    for path in sorted(root_path.iterdir()):
        if path.is_file() and path.suffix.lower() in ('.wav', '.mp3', '.alac'):
            if path.stem in audio_paths:
                raise ValueError(f'Multiple audio files share UID {path.stem!r}')
            audio_paths[path.stem] = path
    for uid in df.uid:
        if not uid or Path(uid).name != uid:
            raise ValueError('UIDs must be nonempty filenames without directories')
        if uid not in audio_paths:
            raise FileNotFoundError(f'No recording for UID {uid!r}')
        if not (preprocessing_path / 'words' / (uid + '.csv')).is_file():
            raise FileNotFoundError(f'Missing word timestamps for UID {uid!r}')
        if not args.overwrite:
            for suffix in ('.pt', '_mask.pt', audio_suffix + '.pt', audio_suffix + '_mask.pt', '_extraction.json'):
                path = embeddings_dir / (uid + text_prefix + suffix)
                if path.exists():
                    raise FileExistsError(f'Output already exists: {path}. Choose a new directory or explicitly use --overwrite.')

    options = {'local_files_only': args.local_files_only}
    if args.text_model == 'stella':
        options['trust_remote_code'] = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **options)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer requires a pad token or an EOS token for chunk padding')
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_id, **options).to(device).eval()
    audio_encoder = build_audio_encoder(args.audio_model, device, args.local_files_only)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(df.itertuples(index=False), 1):
        transcription = unicodedata.normalize('NFC', str(getattr(row, row_column)))
        print(f'Processing {row.uid}', flush=True)
        text, offsets, n_chunks = encode_full_transcript(
            tokenizer, model, transcription, device, args.chunk_size,
        )
        audio_mask, text_mask = embedding_masks(torch.ones(len(offsets), dtype=torch.bool), offsets)
        words_df = pd.read_csv(preprocessing_path / 'words' / (row.uid + '.csv'), dtype={'word': str}, keep_default_na=False)
        words = list(words_df[['word', 'start', 'end']].itertuples(index=False, name=None))
        features, frame_rate = extract_audio_features(audio_paths[row.uid], args.audio_model, audio_encoder, device)
        audio, n_tokens = align_audio(transcription, words, offsets, features, frame_rate)
        stem = row.uid + text_prefix
        # Save only after full token coverage, alignment and finite-value checks.
        for suffix, tensor in (('.pt', text), ('_mask.pt', text_mask),
                               (audio_suffix + '.pt', audio), (audio_suffix + '_mask.pt', audio_mask)):
            torch.save(tensor, embeddings_dir / (stem + suffix))
        metadata = {
            'format': 'full_transcript_v1', 'text_model': model_id,
            'audio_model': args.audio_model, 'chunk_size': args.chunk_size,
            'chunks': n_chunks, 'content_tokens': n_tokens, 'saved_positions': len(offsets),
            'transcript_column': row_column, 'truncated_tokens': 0,
        }
        (embeddings_dir / (stem + '_extraction.json')).write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Full transcript: {n_tokens} content tokens, {n_chunks} chunks, {len(offsets)} saved positions', flush=True)
        print(f'Completed audios: {index}/{len(df)}', flush=True)
    print(f'Saved complete recording embeddings to {embeddings_dir}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-path', type=Path, default=Path('/home/usuaris/veussd/marc.casals/datasets/WAB_samples'))
    parser.add_argument('--embeddings-dir', type=Path, help='Defaults to preprocessing/embeddings_full; preserves the old embeddings directory.')
    parser.add_argument('--text-model', choices=TEXT_MODELS, default='distilbert')
    parser.add_argument('--audio-model', choices=('wav2vec2', 'egemaps', 'mel'), default='wav2vec2')
    parser.add_argument('--chunk-size', type=int, default=200, help='Encoder input length per chunk, including special tokens; never truncates the recording.')
    parser.add_argument('--pauses', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, help='Process only the first N recordings for a smoke test.')
    return parser.parse_args()


if __name__ == '__main__':
    preprocess_text(parse_args())
