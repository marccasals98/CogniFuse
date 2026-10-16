"""Evaluate a saved CogniFuse classifier on a prepared feature directory.

Training code is reused unchanged. Checkpoint selection is always explicit;
labels are separate from features, and bootstrap never reruns the model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.data import PrecomputedADDataset
from scripts.settings import LABELS_TO_IDS


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def class_names(parameters):
    count = int(parameters.number_classes)
    names = getattr(parameters, 'label_names', None) or getattr(parameters, 'target_classes', None)
    if names is None:
        dataset = getattr(parameters, 'dataset', 'ppa')
        if dataset == 'adresso' and count == 2:
            names = ['cn', 'ad']
        elif dataset == 'ppa' and count == 3:
            names = list(LABELS_TO_IDS)
        else:
            names = [f'class_{index}' for index in range(count)]
    if count < 2 or len(names) != count or len(set(names)) != count:
        raise ValueError('Checkpoint has an invalid class count or class-name mapping')
    return list(names)


class EvaluationDataset(Dataset):
    """Index the entire requested directory, without applying training folds."""

    collate_fn = staticmethod(PrecomputedADDataset.collate_fn)

    def __init__(self, directory, parameters):
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise FileNotFoundError(f'Prepared feature directory not found: {self.directory}')
        audio_suffix = getattr(parameters, 'precomputed_audio_suffix', 'distil_audio.pt')
        text_suffix = getattr(parameters, 'precomputed_text_suffix', 'distil.pt')
        for suffix in (audio_suffix, text_suffix):
            if not suffix or Path(suffix).name != suffix or not suffix.endswith('.pt'):
                raise ValueError('Checkpoint feature suffixes must be filenames ending in .pt')
        if audio_suffix == text_suffix:
            raise ValueError('Audio and text feature suffixes must differ')

        def index(suffix):
            result = {}
            for path in sorted(self.directory.rglob('*' + suffix)):
                uid = path.name[:-len(suffix)]
                if not uid or uid in result:
                    raise ValueError(f'Empty or duplicate evaluation recording ID: {uid!r}')
                result[uid] = path
            return result

        audio, text = index(audio_suffix), index(text_suffix)
        if not audio or set(audio) != set(text):
            raise ValueError(f'Evaluation needs complete audio/text pairs: '
                             f'missing_audio={sorted(set(text) - set(audio))}, '
                             f'missing_text={sorted(set(audio) - set(text))}')
        self.records = []
        self.input_files = {}
        for uid in sorted(audio):
            record = {'uid': uid, 'audio': audio[uid], 'text': text[uid]}
            for modality in ('audio', 'text'):
                path = record[modality]
                mask = path.with_name(path.stem + '_mask.pt')
                if not mask.is_file():
                    raise FileNotFoundError(f'Missing {modality} mask for {uid}: {mask}')
                record[modality + '_mask'] = mask
            self.records.append(record)
        self.expected_dimensions = (int(parameters.speech_feature_extractor_output_vectors_dimension),
                                    int(parameters.text_feature_extractor_output_vectors_dimension))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        for key in ('audio', 'text', 'audio_mask', 'text_mask'):
            self.input_files[str(record[key])] = file_identity(record[key])
        audio = PrecomputedADDataset._load_features(record['audio'])
        text = PrecomputedADDataset._load_features(record['text'])
        if audio.shape[0] != text.shape[0] or (audio.shape[1], text.shape[1]) != self.expected_dimensions:
            raise ValueError(f'Feature dimensions do not match checkpoint for {record["uid"]}')
        am = PrecomputedADDataset._load_mask(record['audio_mask'], len(audio))
        tm = PrecomputedADDataset._load_mask(record['text_mask'], len(text))
        return audio, torch.tensor(-1), text, am, tm

    def verify_inputs(self):
        for source in self.input_files.values():
            if file_identity(source['path']) != source:
                raise ValueError(f'Evaluation input changed during inference: {source["path"]}')


def normalize_uid(value):
    value = str(value).strip()
    name = Path(value).name
    if Path(name).suffix.lower() in {'.wav', '.mp3', '.alac', '.flac', '.m4a'}:
        name = Path(name).stem
    if not name:
        raise ValueError('Empty recording ID in labels')
    return name


def read_labels(path, uids, names, dataset_name, id_column=None, label_column=None, group_column=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames or [], list(reader)

    def column(explicit, choices):
        if explicit is not None:
            if explicit not in fields:
                raise ValueError(f'Label CSV has no column {explicit!r}')
            return explicit
        normalized = {name.strip().lower(): name for name in fields}
        for choice in choices:
            if choice.lower() in normalized:
                return normalized[choice.lower()]
        raise ValueError(f'Cannot identify label CSV column; expected one of {choices}')

    id_column = column(id_column, ('uid', 'ID', 'filename', 'audio_path', 'file'))
    label_column = column(label_column, ('label', 'diagnosis', 'Dx', 'DX_Pilar', 'target'))
    if group_column is not None and group_column not in fields:
        raise ValueError(f'Label CSV has no group column {group_column!r}')
    mapping = {str(name).strip().casefold(): index for index, name in enumerate(names)}
    if dataset_name == 'adresso' and names == ['cn', 'ad']:
        mapping.update({'control': 0, 'probablead': 1})
    labels, groups = {}, {}
    for row in rows:
        uid = normalize_uid(row[id_column])
        if uid in labels:
            raise ValueError(f'Duplicate label ID: {uid}')
        value = row[label_column].strip()
        if value.casefold() in mapping:
            label = mapping[value.casefold()]
        elif value in {str(index) for index in range(len(names))}:
            label = int(value)
        else:
            raise ValueError(f'Unknown or missing label {value!r} for {uid}')
        labels[uid] = label
        if group_column is not None:
            group = row[group_column].strip()
            if not group:
                raise ValueError(f'Missing bootstrap group for {uid}')
            groups[uid] = group
    expected = set(uids)
    if set(labels) != expected:
        raise ValueError(f'Labels must match evaluation IDs exactly: '
                         f'missing={sorted(expected - set(labels))}, extra={sorted(set(labels) - expected)}')
    return [labels[uid] for uid in uids], ([groups[uid] for uid in uids] if group_column else None)


def bootstrap_macro_f1(y_true, y_pred, number_classes, samples=10000, seed=1234,
                       confidence_level=0.95, groups=None):
    """Paired recording/group bootstrap conditional on already-fixed predictions."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if (y_true.ndim != 1 or y_pred.shape != y_true.shape or not len(y_true)
            or number_classes < 2 or samples < 2 or not 0 < confidence_level < 1
            or np.any(y_true < 0) or np.any(y_true >= number_classes)
            or np.any(y_pred < 0) or np.any(y_pred >= number_classes)):
        raise ValueError('Invalid bootstrap labels, dimensions, class count, or settings')
    if groups is not None and (len(groups) != len(y_true) or any(not str(g) for g in groups)):
        raise ValueError('Bootstrap group IDs must cover every recording')
    rng = np.random.default_rng(seed)
    distribution = np.empty(samples, dtype=float)

    def score(indices):
        truth, prediction = y_true[indices], y_pred[indices]
        values = np.zeros(truth.shape[0], dtype=float)
        for label in range(number_classes):
            actual, predicted = truth == label, prediction == label
            denominator = actual.sum(axis=1) + predicted.sum(axis=1)
            numerator = 2 * (actual & predicted).sum(axis=1)
            values += np.divide(numerator, denominator, out=np.zeros(len(values)), where=denominator != 0)
        return values / number_classes

    if groups is None:
        for start in range(0, samples, 256):
            size = min(256, samples - start)
            indices = rng.integers(0, len(y_true), size=(size, len(y_true)))
            distribution[start:start + size] = score(indices)
        units = len(y_true)
    else:
        members = {}
        for index, group in enumerate(groups):
            members.setdefault(str(group), []).append(index)
        members = [np.asarray(indices) for indices in members.values()]
        units = len(members)
        for replicate in range(samples):
            selected = rng.integers(0, units, size=units)
            indices = np.concatenate([members[index] for index in selected])
            distribution[replicate] = score(indices[None, :])[0]
    alpha = (1 - confidence_level) / 2
    low, high = np.quantile(distribution, [alpha, 1 - alpha])
    result = {'method': 'paired_percentile', 'confidence_level': confidence_level,
              'lower': float(low), 'upper': float(high), 'standard_error': float(distribution.std(ddof=1)),
              'resamples': samples, 'seed': seed, 'unit': 'group' if groups is not None else 'recording',
              'resampling_units': units, 'recordings': len(y_true),
              'scope': 'evaluation-sample uncertainty conditional on the fixed checkpoint; excludes training variability',
              'class_ids': list(range(number_classes)), 'zero_division': 0}
    return result, distribution


def load_model(parameters, weights, device):
    # The existing model uses historical sibling imports. Keep them unchanged.
    scripts_dir = str(Path(__file__).resolve().parent)
    added = scripts_dir not in sys.path
    if added:
        sys.path.insert(0, scripts_dir)
    try:
        from model import Classifier
        model = Classifier(parameters, device)
    finally:
        if added:
            sys.path.remove(scripts_dir)
    model.load_state_dict(weights, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize evaluation metadata: {type(value).__name__}')


def evaluate(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Evaluation output already exists: {output}; choose a new directory')
    if args.batch_size < 1 or args.threads < 1 or args.bootstrap_samples < 2 or not 0 < args.confidence_level < 1:
        raise ValueError('Batch size, threads, and bootstrap settings must be positive and valid')
    if args.group_column and not args.labels_path:
        raise ValueError('--group_column requires --labels_path')
    torch.set_num_threads(args.threads)
    checkpoint_identity = file_identity(args.checkpoint)
    # Locally trained checkpoints contain argparse.Namespace settings as well
    # as tensors, so tensor-only loading cannot reconstruct this saved format.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not {'model', 'settings'} <= checkpoint.keys():
        raise ValueError('Expected a checkpoint saved by scripts/train.py')
    saved = checkpoint['settings']
    parameters = copy.deepcopy(SimpleNamespace(**saved) if isinstance(saved, dict) else saved)
    original_settings = copy.deepcopy(vars(parameters))
    if not getattr(parameters, 'precomputed_features_dir', None):
        raise ValueError('This evaluator supports checkpoints trained on precomputed features')
    names = class_names(parameters)
    parameters.precomputed_features_dir = str(Path(args.dataset_to_evaluate).resolve())
    dataset = EvaluationDataset(args.dataset_to_evaluate, parameters)
    uids = [r['uid'] for r in dataset.records]
    labels = groups = labels_identity = None
    if args.labels_path:
        labels_identity = file_identity(args.labels_path)
        labels, groups = read_labels(args.labels_path, uids, names, getattr(parameters, 'dataset', 'ppa'),
                                     args.id_column, args.label_column, args.group_column)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    model = load_model(parameters, checkpoint['model'], device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=dataset.collate_fn)
    probabilities = []
    print(f'Evaluating explicit checkpoint: {checkpoint_identity["path"]}', flush=True)
    print(f'{len(dataset)} recordings; labels {"available" if labels is not None else "not supplied"}; device={device}', flush=True)
    with torch.inference_mode():
        for audio, _, text, masks in loader:
            logits = model(audio.to(device), text.to(device), masks.to(device))
            if logits.shape != (audio.shape[0], len(names)) or not torch.isfinite(logits).all():
                raise ValueError('Classifier produced invalid logits')
            probabilities.extend(logits.softmax(-1).cpu().tolist())
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(axis=1).tolist()
    metrics = {'status': 'inference_only', 'recordings': len(dataset), 'class_names': names,
               'label_map': {name: index for index, name in enumerate(names)}}
    distribution = None
    if labels is not None:
        metric_labels = list(range(len(names)))
        bootstrap, distribution = bootstrap_macro_f1(labels, predictions, len(names), args.bootstrap_samples,
                                                       args.bootstrap_seed, args.confidence_level, groups)
        metrics.update({'status': 'evaluated', 'macro_f1': float(f1_score(labels, predictions, labels=metric_labels, average='macro', zero_division=0)),
                        'accuracy': float(accuracy_score(labels, predictions)),
                        'per_class_f1': dict(zip(names, f1_score(labels, predictions, labels=metric_labels, average=None, zero_division=0).tolist())),
                        'confusion_matrix': confusion_matrix(labels, predictions, labels=metric_labels).tolist(),
                        'confusion_matrix_axes': 'rows=true, columns=predicted; class_names order',
                        'macro_f1_bootstrap': bootstrap})
    dataset.verify_inputs()
    if file_identity(args.checkpoint) != checkpoint_identity:
        raise ValueError('Checkpoint file changed during evaluation')
    if labels_identity and file_identity(args.labels_path) != labels_identity:
        raise ValueError('Label file changed during evaluation')
    manifest = {'checkpoint': checkpoint_identity, 'saved_model_name': getattr(parameters, 'model_name', None),
                'checkpoint_settings': original_settings,
                'checkpoint_epoch': checkpoint.get('training_variables', {}).get('epoch'),
                'selection_protocol': 'fixed_epochs_final_model' if getattr(parameters, 'fold_results_path', None) else 'validation_selected_checkpoint',
                'dataset_to_evaluate': str(dataset.directory), 'recording_ids': uids,
                'input_files': list(dataset.input_files.values()), 'labels': labels_identity,
                'class_names': names, 'device': device, 'batch_size': args.batch_size,
                'group_column': args.group_column,
                'inference': {'model_mode': 'eval', 'gradients': False, 'weights_updated': False,
                              'decision_rule': 'argmax over saved-model logits'},
                'bootstrap': metrics.get('macro_f1_bootstrap'),
                'versions': {'torch': torch.__version__, 'numpy': np.__version__}}
    serialized_metrics = json.dumps(metrics, indent=2, allow_nan=False, default=json_default) + '\n'
    serialized_manifest = json.dumps(manifest, indent=2, allow_nan=False, default=json_default) + '\n'
    output.mkdir(parents=True, exist_ok=False)
    fields = ['uid', 'prediction', 'predicted_class'] + ['probability_' + str(index) for index in range(len(names))]
    if labels is not None:
        fields += ['label', 'true_class']
    if groups is not None:
        fields += ['bootstrap_group']
    with (output / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, uid in enumerate(uids):
            row = {'uid': uid, 'prediction': predictions[index], 'predicted_class': names[predictions[index]]}
            row.update({'probability_' + str(label): probabilities[index, label] for label in range(len(names))})
            if labels is not None:
                row.update({'label': labels[index], 'true_class': names[labels[index]]})
            if groups is not None:
                row['bootstrap_group'] = groups[index]
            writer.writerow(row)
    (output / 'metrics.json').write_text(serialized_metrics)
    (output / 'manifest.json').write_text(serialized_manifest)
    if distribution is not None:
        with (output / 'bootstrap_scores.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['replicate', 'macro_f1'])
            writer.writerows(enumerate(distribution))
        interval = metrics['macro_f1_bootstrap']
        print(f"Macro-F1: {metrics['macro_f1']:.6f}; {args.confidence_level:.0%} bootstrap CI "
              f"[{interval['lower']:.6f}, {interval['upper']:.6f}]", flush=True)
    print(f'Wrote evaluation results to {output}', flush=True)
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Explicit locally trained .chkpt file')
    parser.add_argument('--dataset_to_evaluate', type=Path, required=True, help='Prepared audio/text feature directory; all pairs are evaluated')
    parser.add_argument('--labels_path', type=Path, help='Ground-truth CSV; omission performs inference only')
    parser.add_argument('--id_column', help='Override label CSV recording-ID column')
    parser.add_argument('--label_column', help='Override ground-truth label column')
    parser.add_argument('--group_column', help='Participant/group column in labels CSV; resample groups together')
    parser.add_argument('--bootstrap_samples', type=int, default=10000)
    parser.add_argument('--bootstrap_seed', type=int, default=1234)
    parser.add_argument('--confidence_level', type=float, default=0.95)
    parser.add_argument('--output_dir', type=Path, required=True, help='New directory; existing output directories are refused')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args(argv)


if __name__ == '__main__':
    evaluate(parse_args())
