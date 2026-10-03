"""Create word-level text embeddings paired with the Step 7 acoustic export."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

from scripts.word_text_encoder import encode_word_text


DEFAULT_TEXT_MODEL = 'dccuchile/bert-base-spanish-wwm-uncased'


def read_audio_words(directory):
    """Read and validate the exact word order and masks used for acoustic rows."""
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_bytes())
    audio = torch.load(directory / 'audio_word_embeddings.pt', map_location='cpu', weights_only=True)
    mask = torch.load(directory / 'word_mask.pt', map_location='cpu', weights_only=True)
    if metadata.get('format') != 'audio_word_embeddings_v1':
        raise ValueError('Expected a Step 7 audio_word_embeddings_v1 export')
    words, windows = metadata['words'], metadata['word_windows']
    if audio.ndim != 2 or list(audio.shape) != metadata['feature_shape'] or len(audio) != len(words) or len(windows) != len(words):
        raise ValueError('Acoustic rows and original word count disagree')
    if not audio.is_floating_point() or not torch.isfinite(audio).all():
        raise ValueError('Invalid acoustic word embeddings')
    if mask.dtype != torch.bool or mask.shape != (len(words),) or int(mask.sum()) != metadata['valid_word_count']:
        raise ValueError('Invalid acoustic word mask')
    for word_id, (word, window) in enumerate(zip(words, windows)):
        if window['word_id'] != word_id or window['word'] != word:
            raise ValueError('Acoustic word IDs do not match original word order')
        if window['status'] not in ('aligned', 'no_ctc_units') or bool(mask[word_id]) != (window['status'] == 'aligned'):
            raise ValueError('Acoustic word status disagrees with its mask')
    if torch.any(audio[~mask] != 0):
        raise ValueError('Invalid acoustic words must use zero placeholder rows')
    # Verify the word list against the Step 5 export recorded by Step 7.
    source = metadata['inputs']['word_windows']
    source_bytes = Path(source['path']).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source['sha256']:
        raise ValueError('Step 5 word windows changed since acoustic pooling')
    original = json.loads(source_bytes)
    if original['words'] != words or original['word_alignment_mask'] != mask.tolist():
        raise ValueError('Acoustic words or mask differ from the source windows')
    return metadata, audio, mask


def export_text_words(audio_words_dir, output_dir, tokenizer, model,
                      model_id=DEFAULT_TEXT_MODEL, device='cpu', chunk_size=512):
    """Save new text vectors, masks and shared word IDs; preserve audio files."""
    audio_words_dir, output_dir = Path(audio_words_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Export already exists: {output_dir}. Choose a new directory.')
    audio_metadata, audio, audio_mask = read_audio_words(audio_words_dir)
    words = audio_metadata['words']
    text, text_mask, mapping = encode_word_text(words, tokenizer, model, device, chunk_size)
    if len(audio) != len(text) or len(text) != len(words) or mapping['words'] != words:
        raise ValueError('Audio and text embeddings do not preserve one-to-one word correspondence')
    paired_mask = audio_mask & text_mask
    metadata = {
        'format': 'text_word_embeddings_v1', **mapping,
        'text_model': model_id, 'model_revision': getattr(model.config, '_commit_hash', None),
        'model_config': model.config.to_dict(), 'tokenizer_class': type(tokenizer).__name__,
        'tokenizer_sha256': hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode('utf-8')).hexdigest(),
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
        'feature_shape': list(text.shape), 'feature_dtype': str(text.dtype),
        'audio_feature_shape': list(audio.shape), 'valid_text_words': int(text_mask.sum()),
        'valid_audio_words': int(audio_mask.sum()), 'valid_paired_words': int(paired_mask.sum()),
        'missing_word_policy': 'zero row with false text mask; original indices preserved',
        'audio_source': audio_metadata['audio_source'], 'word_source': audio_metadata['word_source'],
        'embeddings_file': 'text_word_embeddings.pt', 'text_mask_file': 'text_word_mask.pt',
        'paired_mask_file': 'paired_word_mask.pt',
        'audio_export': str(audio_words_dir.resolve()),
        'inputs': {
            name: {'path': str((audio_words_dir / name).resolve()),
                   'sha256': hashlib.sha256((audio_words_dir / name).read_bytes()).hexdigest()}
            for name in ('metadata.json', 'audio_word_embeddings.pt', 'word_mask.pt')
        },
    }
    serialized = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(text, output_dir / 'text_word_embeddings.pt')
    torch.save(text_mask, output_dir / 'text_word_mask.pt')
    torch.save(paired_mask, output_dir / 'paired_word_mask.pt')
    (output_dir / 'metadata.json').write_text(serialized, encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-words-dir', type=Path, required=True, help='Existing Step 7 export.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing exports are refused.')
    parser.add_argument('--text-model', default=DEFAULT_TEXT_MODEL)
    parser.add_argument('--chunk-size', type=int, default=512, help='Maximum tokens per input including special tokens; never truncate words.')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--revision', help='Optional checkpoint revision/commit.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Export already exists: {args.output_dir}. Choose a new directory.')
    read_audio_words(args.audio_words_dir)
    options = {'local_files_only': args.local_files_only}
    if args.revision is not None:
        options['revision'] = args.revision
    config = AutoConfig.from_pretrained(args.text_model, **options)
    if getattr(config, '_commit_hash', None):
        options['revision'] = config._commit_hash
    # Word vectors use token hidden states, so BERT's sequence pooler is unused.
    model_options = {'add_pooling_layer': False} if config.model_type == 'bert' else {}
    model, loading = AutoModel.from_pretrained(
        args.text_model, config=config, output_loading_info=True, **options, **model_options,
    )
    missing = [key for key in loading.get('missing_keys', []) if not (key.startswith('pooler.') or '.pooler.' in key)]
    if missing or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError(f'Checkpoint lacks complete pretrained text encoder weights: {loading}')
    revision = getattr(model.config, '_commit_hash', None)
    if revision:
        options['revision'] = revision
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True, **options)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    metadata = export_text_words(args.audio_words_dir, args.output_dir, tokenizer, model,
                                 args.text_model, device, args.chunk_size)
    print(f"Saved text word embeddings {metadata['feature_shape']} to {args.output_dir}")
    print(f"Audio {metadata['audio_feature_shape']}; text {metadata['feature_shape']}; original words: {len(metadata['words'])}")
    print(f"Valid audio/text pairs: {metadata['valid_paired_words']}/{len(metadata['words'])}; chunks: {len(metadata['chunks'])}; no truncated tokens")
    print('Step 8 prepares matched representations; classifier training is Step 9.')


if __name__ == '__main__':
    main()
