# Shared interval, timing and corruption methods copied from the frozen paper_20261021 runner.
"""Matched word features and reproducible transcription corruption for the paper.

Writes only to a new experiment directory. Existing preprocessing is read-only.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.word_text_encoder import encode_word_text

TEXT_MODELS = {'distilbert': 'distilbert-base-uncased', 'deberta_v3_large': 'microsoft/deberta-v3-large'}
CORRUPTION_SEEDS = (1701, 1702, 1703)
RATES = (5, 10, 20, 30)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_tensor(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def stable_seed(uid, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}:{uid}'.encode()).digest()[:8], 'little')


def corrupt_words(words, rate, uid, seed, vocabulary):
    """Edit exactly round(rate*N/100) distinct sites; cycle S/D/I equally.

    Return the new word list and the original audio slot for each text word.
    Insertions use -1. No error indicator is supplied to the model/gate.
    """
    if not 0 <= rate <= 100 or not words or not vocabulary:
        raise ValueError('Need words, a label-independent vocabulary and a rate in [0, 100]')
    rng = np.random.default_rng(stable_seed(uid, seed))
    order = rng.permutation(len(words))
    operations = {int(site): ('substitute', 'delete', 'insert')[j % 3]
                  for j, site in enumerate(order[:int(math.floor(len(words) * rate / 100 + .5))])}
    result, owners, edits = [], [], []
    for i, word in enumerate(words):
        operation = operations.get(i)
        if operation == 'delete':
            edits.append({'word_id': i, 'operation': operation})
            continue
        replacement = word
        if operation in ('substitute', 'insert'):
            # Reject identical substitutions, including vocabularies of size 1.
            candidates = [w for w in vocabulary if w.casefold() != word.casefold()]
            if not candidates:
                raise ValueError('Corruption vocabulary needs at least two distinct words')
            replacement = candidates[int(rng.integers(len(candidates)))]
        result.append(replacement if operation == 'substitute' else word)
        owners.append(i)
        if operation == 'insert':
            result.append(replacement)
            owners.append(-1)
        if operation:
            edits.append({'word_id': i, 'operation': operation})
    if not result:
        raise ValueError('Corruption removed every word')
    return result, owners, edits


def pool_intervals(frames, timing, starts, ends, context_ms):
    """Mean valid frame centers inside each expanded interval, nearest fallback."""
    origin, stride = timing['first_frame_center_seconds'], timing['frame_stride_seconds']
    n = len(frames)
    output = torch.zeros(len(starts), frames.shape[1])
    valid = torch.zeros(len(starts), dtype=torch.bool)
    for i, (start, end) in enumerate(zip(starts, ends)):
        if start is None or end is None:
            continue
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            continue
        begin = max(0., start - context_ms / 1000)
        finish = min(timing['audio_duration_seconds'], end + context_ms / 1000)
        if begin >= finish:
            continue
        first = max(0, min(n, math.ceil((begin - origin) / stride - 1e-9)))
        last = max(0, min(n, math.ceil((finish - origin) / stride - 1e-9)))
        if first >= last:
            first = max(0, min(n - 1, int(math.floor(((begin + finish) / 2 - origin) / stride + .5))))
            last = first + 1
        output[i] = frames[first:last].mean(0)
        valid[i] = True
    return output, valid


def temporal_features(starts, ends):
    """Duration, preceding nonnegative gap, and +/-2-word local words/second."""
    features = torch.zeros(len(starts), 3)
    for i, (start, end) in enumerate(zip(starts, ends)):
        if start is None or end is None or end <= start:
            continue
        features[i, 0] = end - start
        if i and ends[i - 1] is not None:
            features[i, 1] = max(0., start - ends[i - 1])
        first, last = max(0, i - 2), min(len(starts), i + 3)
        ids = [j for j in range(first, last) if starts[j] is not None and ends[j] is not None and ends[j] > starts[j]]
        if ids:
            span = max(ends[j] for j in ids) - min(starts[j] for j in ids)
            features[i, 2] = len(ids) / max(.02, span)
    return torch.log1p(features)


def load_text_model(model_id, device, revision=None, cache_dir=None):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    options = dict(local_files_only=True)
    if cache_dir:
        options['cache_dir'] = str(cache_dir)
    if revision:
        options['revision'] = revision
    config = AutoConfig.from_pretrained(model_id, **options)
    options['revision'] = config._commit_hash
    extra = {'add_pooling_layer': False} if config.model_type == 'bert' else {}
    model, loading = AutoModel.from_pretrained(model_id, config=config, output_loading_info=True, **options, **extra)
    missing = [k for k in loading.get('missing_keys', []) if 'pooler.' not in k]
    if missing or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError(f'Incomplete pretrained encoder: {model_id}: {loading}')
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **options)
    return tokenizer, model.to(device).eval()


def prepare(root, device='cuda', corruption=True, limit=None):
    """Read only the 166 development recordings; preserve the selected baseline."""
    root = Path(root)
    study = json.loads((root / 'study.json').read_text())
    source = Path(study['features_root']) / 'train/embeddings'
    records = study['records'][:limit] if limit else study['records']
    manifests = []
    for record in records:
        uid = record['uid']
        p = source / f'{uid}_word_metadata.json'
        meta = json.loads(p.read_text())
        target = root / 'features' / f'{uid}.pt'
        manifests.append({**record, 'features': str(target), 'source_metadata': str(p)})
        if target.exists():
            cached = torch.load(target, weights_only=True)
            if 'alignment_mask' in cached:
                continue
        words = meta['words']
        audio_meta = json.loads(Path(meta['inputs']['audio']['metadata.json']['path']).read_text())
        windows = json.loads(Path(audio_meta['inputs']['word_windows']['path']).read_text())
        frame_meta = json.loads(Path(audio_meta['inputs']['audio_metadata']['path']).read_text())
        frames = torch.load(audio_meta['inputs']['frames']['path'], weights_only=True, map_location='cpu')
        frame_mask = torch.load(audio_meta['inputs']['valid_frame_mask']['path'], weights_only=True, map_location='cpu')
        assert frame_mask.all() and torch.isfinite(frames).all()
        assert frame_meta['audio_model'] == study['audio_model']
        assert meta['text_model'] == study['text_model'] and meta['text_model_revision'] == study['text_revision']
        assert windows['words'] == words and windows['context_frames'] == 0
        with Path(meta['word_source']['path']).open() as stream:
            whisper = list(csv.DictReader(stream))
        assert [r['word'] for r in whisper] == words
        starts = [r.get('start_time') for r in windows['word_windows']]
        ends = [r.get('end_time') for r in windows['word_windows']]
        text = torch.load(source / f'{uid}_word_text.pt', weights_only=True)
        clean_audio = torch.load(source / f'{uid}_word_audio.pt', weights_only=True)
        common = torch.tensor(meta['paired_mask'], dtype=torch.bool)
        alignment_mask = common.clone()
        data = {'words': words, 'audio': {}, 'text': {study.get('encoder', 'distilbert'): text},
                'timing': temporal_features(starts, ends),
                'confidence': torch.tensor([float(r['probability']) for r in whisper])}
        assert torch.isfinite(data['confidence']).all() and ((data['confidence'] >= 0) & (data['confidence'] <= 1)).all()
        for alignment, begin, end in [('ctc', starts, ends),
                ('whisper', [float(r['start']) for r in whisper], [float(r['end']) for r in whisper])]:
            for context in (0, 100, 200):
                values, mask = pool_intervals(frames, frame_meta['timing'], begin, end, context)
                alignment_mask &= mask
                data['audio'][f'{alignment}_{context}'] = values
        # The baseline is the exact saved tensor, not a repooling approximation.
        assert torch.allclose(data['audio']['ctc_0'][common], clean_audio[common], atol=1e-6, rtol=1e-5)
        data['audio']['ctc_0'] = clean_audio
        data['mask'] = common
        data['alignment_mask'] = alignment_mask
        assert common.any() and alignment_mask.any()
        atomic_tensor(target, data)
        print(f'Prepared acoustic variants: {uid}', flush=True)
    vocabulary = ['the', 'a', 'an', 'of', 'in', 'with', 'for', 'to', 'house', 'person',
                  'table', 'water', 'tree', 'child', 'woman', 'man', 'eat', 'be', 'have', 'and']
    if corruption:
        tokenizer, model = load_text_model(study['text_model'], device, study['text_revision'], study.get('model_cache'))
        assert model.config._commit_hash == study['text_revision']
        for record in manifests:
            data = torch.load(record['features'], weights_only=True)
            for seed in CORRUPTION_SEEDS:
                for rate in RATES:
                    target = root / 'corruption' / f'{rate}_{seed}' / f"{record['uid']}.pt"
                    if target.exists():
                        continue
                    words, owners, edits = corrupt_words(data['words'], rate, record['uid'], seed, vocabulary)
                    values, mask, _ = encode_word_text(words, tokenizer, model, device, chunk_size=study['text_chunk_size'])
                    atomic_tensor(target, {'words': words, 'owners': owners, 'edits': edits, 'text': values, 'mask': mask})
            print(f"Prepared corruption: {record['uid']}", flush=True)
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest = {'format': 'adresso_paper_features_v1', 'records': manifests,
        'text_models': {study.get('encoder', 'distilbert'): study['text_model']},
        'revisions': {study.get('encoder', 'distilbert'): study['text_revision']},
        'audio_model': study['audio_model'], 'corruption_seeds': CORRUPTION_SEEDS,
        'corruption_rates': RATES, 'corruption_vocabulary': vocabulary,
        'paired_mask_policy': 'selected baseline mask; alignment family alone uses intersection of six variants with matched CTC control',
        'complete': len(manifests) == 166 and limit is None}
    atomic_json(root / ('features_manifest.json' if limit is None else 'pilot_features_manifest.json'), manifest)
    if corruption:
        atomic_json(root / ('corruption_manifest.json' if limit is None else 'pilot_corruption_manifest.json'), manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-corruption', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare(args.root, args.device, not args.no_corruption, args.limit)


