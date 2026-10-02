"""Export unpooled Wav2Vec2 frames without regenerating existing embeddings."""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
import transformers
from transformers import Wav2Vec2Model, Wav2Vec2Processor

if __package__:
    from .audio_frame_timing import frame_timing
else:
    from audio_frame_timing import frame_timing


# Match the current offline acoustic encoder; CTC alignment is a later step.
MODEL_ID = 'facebook/wav2vec2-base-960h'
SAMPLE_RATE = 16000


def extract_recording_frames(audio_path, processor, model, device):
    """Return [T, D] hidden states, a [T] validity mask and exact grid metadata."""
    waveform, source_rate = torchaudio.load(audio_path)
    source_channels, source_samples = waveform.shape
    if source_samples == 0 or not torch.isfinite(waveform).all():
        raise ValueError('Expected nonempty, finite audio')
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = torchaudio.transforms.Resample(source_rate, SAMPLE_RATE)(waveform).squeeze(0)
    inputs = processor(waveform, sampling_rate=SAMPLE_RATE,
                       return_tensors='pt', padding=False).to(device)
    if inputs['input_values'].shape != (1, waveform.numel()):
        raise ValueError('The processor changed the unpadded waveform length')
    if 'attention_mask' in inputs and not inputs['attention_mask'].bool().all():
        raise ValueError('Unexpected padding in single-recording input')
    # Validate the convolution geometry before a potentially expensive forward.
    expected = waveform.numel()
    for kernel, stride in zip(model.config.conv_kernel, model.config.conv_stride):
        expected = (expected - kernel) // stride + 1
    timing = frame_timing(model.config, waveform.numel(), SAMPLE_RATE, expected)
    model.eval()
    with torch.no_grad():
        features = model(**inputs).last_hidden_state[0].detach().cpu()
    if features.ndim != 2 or features.shape[0] != expected or features.shape[1] == 0:
        raise ValueError('Encoder output does not match the expected frame sequence')
    if not torch.isfinite(features).all():
        raise ValueError('Encoder produced nonfinite acoustic features')
    # Single recordings have no batch padding, repetition or speed augmentation.
    valid_mask = torch.ones(len(features), dtype=torch.bool)
    timing.update({
        'source_sample_rate': source_rate,
        'source_num_samples': source_samples,
        'source_channels': source_channels,
        'source_duration_seconds': source_samples / source_rate,
    })
    return features, valid_mask, timing


def export_recording_frames(audio_path, output_dir, processor, model, device):
    """Save a separate export directory; refuse to replace any existing export."""
    audio_path, output_dir = Path(audio_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Export already exists: {output_dir}. Choose a new output directory.')
    source_stat = audio_path.stat()
    features, valid_mask, timing = extract_recording_frames(audio_path, processor, model, device)
    metadata = {
        'format': 'acoustic_frames_v1',
        'source': {
            'path': str(audio_path.resolve()),
            'size_bytes': source_stat.st_size,
            'mtime_ns': source_stat.st_mtime_ns,
        },
        'audio_model': MODEL_ID,
        'model_config': model.config.to_dict(),
        'processor_config': processor.feature_extractor.to_dict(),
        'versions': {'torch': torch.__version__, 'torchaudio': torchaudio.__version__,
                     'transformers': transformers.__version__},
        'feature_shape': list(features.shape),
        'feature_dtype': str(features.dtype),
        'features_file': 'frames.pt',
        'valid_mask_file': 'valid_mask.pt',
        'timing': timing,
        'augmentation': None,
        'pooled': False,
    }
    # Serialize metadata before creating outputs. A failed export is never
    # silently overwritten on a later run; choose a fresh destination to retry.
    serialized = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(features, output_dir / 'frames.pt')
    torch.save(valid_mask, output_dir / 'valid_mask.pt')
    (output_dir / 'metadata.json').write_text(serialized, encoding='utf-8')
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-path', type=Path, required=True, help='One complete recording.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for this recording; existing directories are refused.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--local-files-only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.audio_path.is_file():
        raise FileNotFoundError(args.audio_path)
    if args.output_dir.exists():
        raise FileExistsError(f'Export already exists: {args.output_dir}. Choose a new output directory.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, local_files_only=args.local_files_only)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID, local_files_only=args.local_files_only).to(device).eval()
    metadata = export_recording_frames(args.audio_path, args.output_dir, processor, model, device)
    print(f"Saved {metadata['feature_shape']} acoustic frames to {args.output_dir}")
    print(f"Frame stride: {metadata['timing']['frame_stride_seconds']:.6f} seconds")


if __name__ == '__main__':
    main()
