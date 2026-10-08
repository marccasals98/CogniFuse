"""Package a Step 7/8 recording for the existing PrecomputedADDataset."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from utils.prepro_word_text import read_audio_words
from utils.ctc_skipped_words import skip_metadata


AUDIO_SUFFIX = '_word_audio.pt'
TEXT_SUFFIX = '_word_text.pt'


def file_identity(path):
    path = Path(path)
    return {'path': str(path.resolve()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def load_mask(path, count):
    mask = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (count,):
        raise ValueError(f'Expected a boolean word mask of shape [{count}]: {path}')
    return mask


def package_word_embeddings(audio_words_dir, text_words_dir, output_dir):
    """Copy validated vectors into a new directory, using paired validity.

    No encoders run and no rows are removed. Both training masks use the
    intersection so this experiment includes only words valid in both modalities.
    Original modality masks are retained in the per-recording metadata.
    """
    audio_dir, text_dir, output_dir = map(Path, (audio_words_dir, text_words_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f'Export already exists: {output_dir}. Choose a new directory.')
    audio_meta, audio, audio_mask = read_audio_words(audio_dir)
    text_meta = json.loads((text_dir / 'metadata.json').read_bytes())
    if text_meta.get('format') != 'text_word_embeddings_v1':
        raise ValueError('Expected a Step 8 text_word_embeddings_v1 export')
    words = audio_meta['words']
    if skip_metadata(audio_meta) != skip_metadata(text_meta):
        raise ValueError('Audio/text skipped-word metadata differs')
    if text_meta['words'] != words:
        raise ValueError('Audio/text original word order differs')
    rows = text_meta['word_mappings']
    if len(rows) != len(words) or any(
        row['word_id'] != i or row['word'] != word
        for i, (row, word) in enumerate(zip(rows, words))
    ):
        raise ValueError('Text word IDs do not match the acoustic rows')
    for name in ('metadata.json', 'audio_word_embeddings.pt', 'word_mask.pt'):
        if file_identity(audio_dir / name)['sha256'] != text_meta['inputs'][name]['sha256']:
            raise ValueError(f'Step 8 was computed from different acoustic inputs: {name}')
    if any(text_meta[key] != audio_meta[key] for key in ('audio_source', 'word_source')):
        raise ValueError('Audio/text source identities differ')

    text = torch.load(text_dir / 'text_word_embeddings.pt', map_location='cpu', weights_only=True)
    if (not isinstance(text, torch.Tensor) or text.ndim != 2
            or text.shape[0] != len(words) or text.shape[1] < 1
            or not text.is_floating_point() or not torch.isfinite(text).all()
            or list(text.shape) != text_meta['feature_shape']):
        raise ValueError('Invalid text word embeddings or feature shape')
    if list(audio.shape) != text_meta['audio_feature_shape']:
        raise ValueError('Step 8 acoustic feature shape differs')
    text_mask = load_mask(text_dir / 'text_word_mask.pt', len(words))
    paired = load_mask(text_dir / 'paired_word_mask.pt', len(words))
    if not torch.equal(paired, audio_mask & text_mask):
        raise ValueError('Paired mask differs from the audio/text intersection')
    if not paired.any():
        raise ValueError('At least one valid audio/text word pair is required')
    for mask, field in ((audio_mask, 'valid_audio_words'), (text_mask, 'valid_text_words'),
                        (paired, 'valid_paired_words')):
        if int(mask.sum()) != text_meta[field]:
            raise ValueError(f'Incorrect Step 8 count: {field}')
    for i, row in enumerate(rows):
        expected_status = 'encoded' if text_mask[i] else 'no_text_tokens'
        if row['status'] != expected_status:
            raise ValueError('Text word status disagrees with its mask')
    if torch.any(text[~text_mask] != 0):
        raise ValueError('Invalid text rows must be zero placeholders')

    uid = Path(audio_meta['audio_source']['path']).stem
    if not uid or uid in ('.', '..'):
        raise ValueError('Audio source must provide a recording UID')
    audio_name, text_name = uid + AUDIO_SUFFIX, uid + TEXT_SUFFIX
    tensors = {audio_name: audio.float(), text_name: text.float(),
               Path(audio_name).stem + '_mask.pt': paired,
               Path(text_name).stem + '_mask.pt': paired}
    metadata = {
        **skip_metadata(audio_meta),
        'format': 'packaged_word_embeddings_v1', 'uid': uid, 'words': words,
        'word_ids': list(range(len(words))), 'audio_feature_shape': list(audio.shape),
        'text_feature_shape': list(text.shape), 'text_model': text_meta['text_model'],
        'text_model_revision': text_meta['model_revision'],
        'audio_source': audio_meta['audio_source'], 'word_source': audio_meta['word_source'],
        'audio_suffix': AUDIO_SUFFIX, 'text_suffix': TEXT_SUFFIX,
        'training_mask_policy': 'audio AND text for both modalities; all original rows retained',
        'original_audio_mask': audio_mask.tolist(), 'original_text_mask': text_mask.tolist(),
        'paired_mask': paired.tolist(), 'valid_paired_words': int(paired.sum()),
        'inputs': {
            'audio': {name: file_identity(audio_dir / name) for name in
                      ('metadata.json', 'audio_word_embeddings.pt', 'word_mask.pt')},
            'text': {name: file_identity(text_dir / name) for name in
                     ('metadata.json', 'text_word_embeddings.pt', 'text_word_mask.pt', 'paired_word_mask.pt')},
        },
    }
    # All source validation finishes before any output is created.
    json.dumps(metadata, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, tensor in tensors.items():
        torch.save(tensor, output_dir / name)
    metadata['output_sha256'] = {
        name: file_identity(output_dir / name)['sha256'] for name in tensors
    }
    (output_dir / (uid + '_word_metadata.json')).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8',
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-words-dir', type=Path, required=True)
    parser.add_argument('--text-words-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; never overwrite exports.')
    args = parser.parse_args()
    metadata = package_word_embeddings(args.audio_words_dir, args.text_words_dir, args.output_dir)
    print(f"Packaged {metadata['uid']} into {args.output_dir}")
    print(f"Audio {metadata['audio_feature_shape']}; text {metadata['text_feature_shape']}; "
          f"valid pairs {metadata['valid_paired_words']}/{len(metadata['words'])}")
    print(f'Loader suffixes: --precomputed_audio_suffix {AUDIO_SUFFIX} --precomputed_text_suffix {TEXT_SUFFIX}')
    print('Packaging only. Classifier training has not been run.')


if __name__ == '__main__':
    main()
