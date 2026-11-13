"""Integrated Gradients word attributions for a saved CogniFuse classifier.

The model is explained on the precomputed word embeddings it was trained on:
audio (wav2vec2) and text (DistilBERT), aligned one row per word. Both
modalities are attributed jointly along the straight path from an all-zero
baseline to the real input, so every word gets an audio and a text score.

Explained output: by default the logit difference ``logit[ad] - logit[cn]``
(positive attribution pushes towards AD, negative towards CN). The masks are
kept fixed; padding never appears because recordings are explained one by one.

Completeness check (IG axiom): the sum of all attributions must equal
``f(input) - f(baseline)``. The relative error is reported per recording.

Training code is reused unchanged: the checkpoint is loaded with
``scripts.evaluate.load_model`` and recordings come from a cross-validation
``fold_<k>.json`` (its validation predictions), so only unseen recordings are
explained. The saved predictions are also used to verify the checkpoint loads
correctly with this code version.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluate import file_identity, load_model


def gauss_legendre(steps):
    """Integration points in (0, 1) and weights summing to 1."""
    x, w = np.polynomial.legendre.leggauss(steps)
    return torch.tensor((x + 1) / 2, dtype=torch.float64), torch.tensor(w / 2, dtype=torch.float64)


def explained_output(logits, target, class_ids):
    if target == 'ad_minus_cn':
        return logits[:, class_ids['ad']] - logits[:, class_ids['cn']]
    return logits[:, class_ids[target]]


def split_input_norms(model):
    """Move the two input LayerNorms out of the loaded model instance.

    LayerNorm is scale invariant (LN(a*x) == LN(x) unless a is ~0), so on the
    straight path 0 -> x the output is flat and then jumps right next to the
    zero baseline; numerical IG misses that jump. Integrating after the norms
    keeps the same endpoints, LN(0) -> LN(x), hence the same f(x) - f(0), on a
    smooth path. Only this in-memory instance changes, never the class code.
    """
    norms = model.speech_feature_extractor_norm_layer, model.text_feature_extractor_norm_layer
    model.speech_feature_extractor_norm_layer = torch.nn.Identity()
    model.text_feature_extractor_norm_layer = torch.nn.Identity()
    return norms


def integrated_gradients(model, audio, text, masks, target, class_ids, steps, chunk):
    """Joint IG over audio and text with the all-zero raw-input baseline.

    audio/text: normalised embeddings LN(x) [tokens, dims]; the baselines are
    LN(0). masks: [2, tokens]. Returns per-element attributions for both
    modalities plus f(input) and f(baseline).
    """
    audio, audio_base = audio
    text, text_base = text
    alphas, weights = gauss_legendre(steps)
    alphas, weights = alphas.to(audio.device, audio.dtype), weights.to(audio.device, audio.dtype)
    audio_grad = torch.zeros_like(audio)
    text_grad = torch.zeros_like(text)
    for start in range(0, steps, chunk):
        a = alphas[start:start + chunk].view(-1, 1, 1)
        w = weights[start:start + chunk]
        scaled_audio = (audio_base + a * (audio - audio_base)).requires_grad_(True)
        scaled_text = (text_base + a * (text - text_base)).requires_grad_(True)
        batch_masks = masks.unsqueeze(0).expand(len(a), -1, -1)
        output = explained_output(model(scaled_audio, scaled_text, batch_masks), target, class_ids)
        grads = torch.autograd.grad((output * w).sum(), (scaled_audio, scaled_text))
        audio_grad += grads[0].sum(0)
        text_grad += grads[1].sum(0)
    with torch.no_grad():
        both = torch.stack((audio, audio_base))
        both_text = torch.stack((text, text_base))
        ends = explained_output(model(both, both_text, masks.unsqueeze(0).expand(2, -1, -1)),
                                target, class_ids)
    return ((audio - audio_base) * audio_grad, (text - text_base) * text_grad,
            ends[0].item(), ends[1].item())


def read_words(features_dir, uid, tokens):
    metadata_path = Path(features_dir) / f'{uid}_word_metadata.json'
    if not metadata_path.is_file():
        return [''] * tokens
    words = json.loads(metadata_path.read_text())['words']
    if len(words) != tokens:
        raise ValueError(f'{uid}: {len(words)} words but {tokens} feature rows')
    return words


def load_tensor(path, kind):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f'Expected a tensor: {path}')
    return value.to(torch.float32) if kind == 'features' else value.bool()


def main(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}; choose a new directory')
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved = checkpoint['settings']
    parameters = copy.deepcopy(SimpleNamespace(**saved) if isinstance(saved, dict) else saved)
    if getattr(parameters, 'dataset', None) != 'adresso':
        raise ValueError('This first version explains ADReSSo (cn/ad) checkpoints only')
    class_ids = {'cn': 0, 'ad': 1}
    model = load_model(parameters, checkpoint['model'], device)  # eval(): no dropout
    audio_norm, text_norm = split_input_norms(model)

    fold = json.loads(Path(args.fold_results).read_text())
    if Path(fold['checkpoint']).resolve() != Path(args.checkpoint).resolve():
        raise ValueError('fold_results belongs to a different checkpoint')
    recordings = fold['predictions'][:args.limit] if args.limit else fold['predictions']
    features_dir = Path(args.features_dir or parameters.precomputed_features_dir)
    audio_suffix, text_suffix = parameters.precomputed_audio_suffix, parameters.precomputed_text_suffix

    output.mkdir(parents=True)
    word_rows, recording_rows, mismatches = [], [], 0
    for index, record in enumerate(recordings):
        uid = record['uid']
        audio = load_tensor(features_dir / f'{uid}{audio_suffix}', 'features')
        text = load_tensor(features_dir / f'{uid}{text_suffix}', 'features')
        audio_mask = load_tensor(features_dir / f'{uid}{Path(audio_suffix).stem}_mask.pt', 'mask')
        text_mask = load_tensor(features_dir / f'{uid}{Path(text_suffix).stem}_mask.pt', 'mask')
        masks = torch.stack((audio_mask, text_mask)).to(device)
        with torch.no_grad():
            audio = (audio_norm(audio.to(device)), audio_norm(torch.zeros_like(audio, device=device)))
            text = (text_norm(text.to(device)), text_norm(torch.zeros_like(text, device=device)))
            logits = model(audio[0].unsqueeze(0), text[0].unsqueeze(0), masks.unsqueeze(0))[0]
        prediction = int(logits.argmax())
        mismatches += prediction != record['prediction']
        attr_audio, attr_text, f_input, f_baseline = integrated_gradients(
            model, audio, text, masks, args.target, class_ids, args.steps, args.chunk)
        word_audio, word_text = attr_audio.sum(-1).cpu(), attr_text.sum(-1).cpu()
        total = (word_audio.sum() + word_text.sum()).item()
        gap = f_input - f_baseline
        error = abs(total - gap) / max(abs(gap), 1e-8)

        words = read_words(features_dir, uid, audio_mask.shape[0])
        valid = (audio_mask & text_mask).tolist()
        for position, word in enumerate(words):
            word_rows.append({
                'uid': uid, 'label': record['label'], 'prediction': prediction,
                'position': position, 'word': word, 'valid': int(valid[position]),
                'attr_audio': round(word_audio[position].item(), 6),
                'attr_text': round(word_text[position].item(), 6),
                'attr_total': round(word_audio[position].item() + word_text[position].item(), 6),
            })
        recording_rows.append({
            'uid': uid, 'label': record['label'], 'prediction': prediction,
            'saved_prediction': record['prediction'],
            'prob_ad': round(logits.softmax(-1)[class_ids['ad']].item(), 6),
            'words': len(words), 'f_input': round(f_input, 6), 'f_baseline': round(f_baseline, 6),
            'sum_audio': round(word_audio.sum().item(), 6), 'sum_text': round(word_text.sum().item(), 6),
            'completeness_rel_error': round(error, 6),
        })
        print(f'[{index + 1}/{len(recordings)}] {uid}: label={record["label"]} pred={prediction} '
              f'f(x)-f(0)={gap:.4f} sum_attr={total:.4f} rel_error={error:.2e}', flush=True)

    for name, rows in (('words.csv', word_rows), ('recordings.csv', recording_rows)):
        with open(output / name, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    errors = [row['completeness_rel_error'] for row in recording_rows]
    summary = {
        'method': 'integrated_gradients',
        'baseline': 'zeros (audio and text raw inputs), integrated after the input LayerNorms',
        'target': args.target, 'steps': args.steps, 'integration': 'gauss_legendre',
        'checkpoint': file_identity(args.checkpoint), 'fold_results': str(Path(args.fold_results).resolve()),
        'features_dir': str(features_dir), 'device': device, 'recordings': len(recording_rows),
        'prediction_mismatches_vs_saved': int(mismatches),
        'completeness_rel_error_max': max(errors), 'completeness_rel_error_mean': sum(errors) / len(errors),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if mismatches:
        print(f'WARNING: {mismatches} predictions differ from the saved fold results', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='.chkpt saved by scripts/train.py')
    parser.add_argument('--fold_results', required=True, help='fold_<k>.json of the same checkpoint')
    parser.add_argument('--output_dir', required=True, help='New directory for words.csv, recordings.csv, summary.json')
    parser.add_argument('--features_dir', default=None, help='Defaults to the checkpoint precomputed_features_dir')
    parser.add_argument('--target', default='ad_minus_cn', choices=['ad_minus_cn', 'ad', 'cn'])
    parser.add_argument('--steps', type=int, default=64, help='IG integration steps')
    parser.add_argument('--chunk', type=int, default=16, help='Steps per forward/backward pass')
    parser.add_argument('--limit', type=int, default=0, help='Explain only the first N recordings (0 = all)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--threads', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
