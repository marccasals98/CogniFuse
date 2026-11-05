"""ADReSSo paper ablations, anchored to the selected word-feature CV baseline.

WordDataset, position encoding and diagnostic prediction/timing methods are
adapted from outputs/paper_20261021/source/utils/paper_experiments.py.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'scripts'))
from model import Classifier
from scripts.adresso import ADReSSoIndex
from scripts.poolings import MultiHeadAttention, sequence_mask
from utils.adresso_paper_features import atomic_json, atomic_tensor, stable_seed, CORRUPTION_SEEDS, RATES

CLASSES = ['cn', 'ad']
SHUFFLE_SEEDS = (2701, 2702, 2703, 2704, 2705)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(truth, predictions):
    return {'macro_f1': float(f1_score(truth, predictions, labels=[0, 1], average='macro', zero_division=0)),
            'per_class_f1': dict(zip(CLASSES, f1_score(truth, predictions, labels=[0, 1], average=None, zero_division=0).tolist()))}


class WordDataset(Dataset):
    def __init__(self, records, cache, config, root, corruption=None, shuffle_seed=None):
        self.records, self.cache, self.config = records, cache, config
        self.root, self.corruption, self.shuffle_seed = Path(root), corruption, shuffle_seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        data = self.cache[record['uid']]
        config = self.config
        audio = data['audio'][f"{config['alignment']}_{config['context']}"]
        text = data['text'][config['encoder']]
        amask = data['alignment_mask' if config.get('shared_alignment_mask') else 'mask'].clone()
        tmask = amask.clone()
        timing, confidence = data['timing'], data['confidence']
        limit = config['prefix'] or len(audio)
        audio, text, amask, tmask = [x[:limit].clone() for x in (audio, text, amask, tmask)]
        timing, confidence = timing[:limit].clone(), confidence[:limit].clone()
        if self.corruption is not None:
            rate, seed = self.corruption
            corrupted = torch.load(self.root / 'corruption' / f'{rate}_{seed}' / f"{record['uid']}.pt", weights_only=True)
            # Text stays in original audio slots; insertions occupy extra slots.
            owners = corrupted['owners']
            extra = sum(owner < 0 for owner in owners)
            count = len(audio)
            audio = torch.cat([audio, torch.zeros(extra, audio.shape[1])])
            amask = torch.cat([amask, torch.zeros(extra, dtype=torch.bool)])
            timing = torch.cat([timing, torch.zeros(extra, 3)])
            # Inserted words have no observed ASR confidence: use recording mean.
            confidence = torch.cat([confidence, confidence.mean().repeat(extra)])
            text = torch.zeros(count + extra, text.shape[1])
            tmask = torch.zeros(count + extra, dtype=torch.bool)
            cursor = count
            for i, owner in enumerate(owners):
                if owner < 0:
                    slot = cursor
                    cursor += 1
                elif owner >= count:
                    continue
                else:
                    slot = owner
                text[slot] = corrupted['text'][i]
                tmask[slot] = corrupted['mask'][i] and (owner < 0 or bool(amask[slot]))
        if self.shuffle_seed is not None:
            indices = torch.where(tmask)[0]
            generator = torch.Generator().manual_seed(stable_seed(record['uid'], self.shuffle_seed))
            permutation = indices[torch.randperm(len(indices), generator=generator)]
            text[indices] = text[permutation]
        return audio, text, amask, tmask, timing, confidence, record['label']

    @staticmethod
    def collate(batch):
        columns = list(zip(*batch))
        return tuple(pad_sequence(column, batch_first=True) for column in columns[:-1]) + (torch.tensor(columns[-1]),)


def position_encoding(length, dimension, device):
    positions = torch.arange(length, device=device)[:, None]
    scales = torch.exp(torch.arange(0, dimension, 2, device=device) * (-np.log(10000.) / dimension))
    encoding = torch.zeros(length, dimension, device=device)
    encoding[:, 0::2] = torch.sin(positions * scales)
    encoding[:, 1::2] = torch.cos(positions * scales)
    return encoding


@torch.inference_mode()
def predict(model, dataset, device, batch_size=1):
    model.eval()
    truth, predictions = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=WordDataset.collate):
        *inputs, labels = batch
        logits = model(*(x.to(device) for x in inputs))
        if not torch.isfinite(logits).all():
            raise ValueError('Non-finite evaluation logits')
        truth.extend(labels.tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
    return truth, predictions


@torch.inference_mode()
def benchmark(model, dataset, device):
    """Batch size 1, data transfer excluded, 3 repeats after 5 warmups."""
    model.eval()
    samples = [tuple(x.to(device) for x in WordDataset.collate([dataset[i]])[:-1]) for i in range(len(dataset))]
    def sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
    for i in range(5):
        model(*samples[i % len(samples)])
    sync()
    start = time.perf_counter()
    for _ in range(3):
        for inputs in samples:
            model(*inputs)
    sync()
    return (time.perf_counter() - start) * 1000 / (3 * len(samples))


def experiment_matrix():
    base = dict(modality='both', alignment='ctc', context=0, encoder='distilbert',
                fusion='MultiHeadAttention', pooling='AttentionPooling', prefix=0,
                temporal='none', temporal_projection=False, gated=False, shared_alignment_mask=False)
    rows = []
    def add(name, aliases, **changes):
        rows.append(dict(base, name=name, aliases=aliases, **changes))
    add('global', ['M-AT', 'C-G', 'F-MA', 'L-FULL', 'R-standard', 'E-D', 'V-G'])
    add('audio', ['M-A', 'V-A'], modality='audio')
    add('text', ['M-T', 'V-T'], modality='text')
    for alignment in ('whisper', 'ctc'):
        for context in (0, 100, 200):
            if alignment == 'ctc' and context == 0:
                continue
            add(f'{alignment}_{context}', [f"A-{'W' if alignment == 'whisper' else 'C'}{context}"], alignment=alignment, context=context, shared_alignment_mask=True)
    add('positions', ['C-P'], fusion='PositionalMultiHeadAttention')
    add('word_pair', ['C-W', 'V-W'], fusion='WordPairMLP')
    add('word_context', ['C-C'], fusion='WordPairContext')
    for mode, code in [('none', '0'), ('duration', 'D'), ('pause', 'P'), ('rate', 'R'), ('all', 'ALL')]:
        add(f'timing_{mode}', [f'T-{code}'], temporal=mode, temporal_projection=True)
    add('none_mean', ['F-NM'], fusion='NoneSeqToSeq', pooling='StatisticalPooling')
    add('none_attention', ['F-NA'], fusion='NoneSeqToSeq')
    add('cross_attention', ['F-CA'], fusion='CrossAttention')
    add('transformer', ['F-TR'], fusion='TransformerStacked')
    for prefix in (50, 100, 200, 400):
        add(f'prefix_{prefix}', [f'L-{prefix}'], prefix=prefix)
    add('gated', ['R-gated'], gated=True)
    add('alignment_control', ['A-C0'], shared_alignment_mask=True)
    return rows


class PaperModel(Classifier):
    """Keep production state keys and initialization for exact baseline reuse."""
    def __init__(self, config, settings, device='cpu'):
        self.config = config
        params = copy.deepcopy(settings)
        params['seq_to_seq_method'] = config['fusion'] if config['fusion'] in ('NoneSeqToSeq', 'CrossAttention', 'TransformerStacked') else 'MultiHeadAttention'
        params['seq_to_one_method'] = config['pooling']
        if config['fusion'] == 'TransformerStacked':
            params.update(transformer_n_blocks=2, transformer_expansion_coef=4, transformer_drop_out=.3)
        super().__init__(SimpleNamespace(**params), device)
        if config['modality'] == 'audio':
            self.text_feature_extractor_norm_layer = None
        elif config['modality'] == 'text':
            self.speech_feature_extractor_norm_layer = None
        dimension = params['speech_feature_extractor_output_vectors_dimension']
        if config['temporal_projection']:
            self.temporal_projection = nn.Linear(3, dimension, bias=False)
        if config['gated']:
            self.gate = nn.Sequential(nn.Linear(2 * dimension + 1, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())
        if config['fusion'].startswith('WordPair'):
            self.pair = nn.Sequential(nn.Linear(2 * dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension), nn.LayerNorm(dimension))
            if config['fusion'] == 'WordPairMLP':
                self.seq_to_seq_layer = None

    def forward(self, audio, text, amask, tmask, timing, confidence):
        config = self.config
        if self.speech_feature_extractor_norm_layer is not None:
            audio = self.speech_feature_extractor_norm_layer(audio)
        if self.text_feature_extractor_norm_layer is not None:
            text = self.text_feature_extractor_norm_layer(text)
        if config['temporal_projection']:
            selection = {'none': [0, 0, 0], 'duration': [1, 0, 0], 'pause': [0, 1, 0], 'rate': [0, 0, 1], 'all': [1, 1, 1]}[config['temporal']]
            audio = audio + self.temporal_projection(timing * timing.new_tensor(selection))
        if config['gated']:
            text = text * self.gate(torch.cat([audio, text, confidence.unsqueeze(-1)], -1))
        if config['fusion'] == 'PositionalMultiHeadAttention':
            positions = position_encoding(audio.shape[1], audio.shape[2], audio.device)
            audio, text = audio + positions, text + positions
        audio, text = self.seq_to_seq_input_dropout(audio), self.seq_to_seq_input_dropout(text)
        if config['modality'] == 'audio':
            text, tmask = text[:, :0], tmask[:, :0]
        elif config['modality'] == 'text':
            audio, amask = audio[:, :0], amask[:, :0]
        if config['fusion'].startswith('WordPair'):
            mask = amask & tmask
            values = self.pair(torch.cat([audio, text], -1))
            if self.seq_to_seq_layer is not None:
                values = self.seq_to_seq_layer(values, values[:, :0], mask, mask[:, :0])
        else:
            mask = sequence_mask(audio, text, amask, tmask)
            values = self.seq_to_seq_layer(audio, text, amask, tmask)
        return self.classifier_layer(self.seq_to_one_layer(self.seq_to_one_input_dropout(values), mask))


def plan(root, baseline, features_root, dataset_root):
    root, baseline, features_root = map(lambda p: Path(p).resolve(), (root, baseline, features_root))
    if (root / 'study.json').exists():
        raise ValueError('Study already exists; use its frozen source to resume')
    assert read(features_root / 'summary.json')['training_ready']
    settings = read(baseline / 'config.json')
    summary = read(baseline / 'summary.json')
    assert summary['num_folds'] == 5 and summary['recordings'] == 166
    index = ADReSSoIndex(dataset_root)
    splits = []
    for fold in range(5):
        select = lambda partition: [{k: r[k] for k in ('uid', 'patient_id', 'label')} for r in index.select(partition, fold)]
        splits.append({'fold': fold, 'train': select('train'), 'validation': select('val')})
        old = read(baseline / f'fold_{fold}.json')
        assert {r['uid'] for r in splits[-1]['validation']} == {r['uid'] for r in old['predictions']}
    matrix = experiment_matrix()
    study = {'format': 'adresso_paper_study_v1', 'baseline_id': 'T25', 'baseline_results': str(baseline),
        'baseline_mean_macro_f1': summary['mean_macro_f1'], 'baseline_std_macro_f1': summary['std_macro_f1'],
        'features_root': str(features_root), 'dataset_root': str(index.root),
        'records': [{k: r[k] for k in ('uid', 'patient_id', 'label')} for r in index.training],
        'text_model': 'distilbert-base-uncased', 'text_revision': '12040accade4e8a0f71eabdb258fecc2e7e948be',
        'audio_model': 'facebook/wav2vec2-base-960h', 'audio_revision': '22aad52d435eb6dbaf354bdad9b0da84ce7d6156',
        'text_chunk_size': 200, 'settings': settings, 'matrix': matrix,
        'new_cv_configurations': 25, 'new_fold_fits': 125,
        'selection': 'Highest completed five-fold validation macro-F1 mean; SD then trial ID break ties',
        'excluded': ['BETO and MrBERT encoder comparisons (DistilBERT only)', 'Optional transfer table (not run in reference)'],
        'input_sha256': {str(p): sha256(p) for p in [baseline / 'config.json', baseline / 'summary.json', baseline / 'splits.json', features_root / 'config.json', features_root / 'summary.json']}}
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'study.json', study)
    atomic_json(root / 'splits.json', splits)
    for directory in ('scripts', 'utils'):
        for path in (SOURCE / directory).glob('*.py'):
            target = root / 'source' / directory / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    atomic_json(root / 'source_sha256.json', {str(p.relative_to(root)): sha256(p) for p in (root / 'source').rglob('*.py')})
    print(json.dumps({k: v for k, v in study.items() if k not in ('records', 'matrix', 'settings', 'input_sha256')}, indent=2))


def validate_inputs(root):
    study = read(root / 'study.json')
    for path, expected in read(root / 'source_sha256.json').items():
        if sha256(root / path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    for path, expected in study['input_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Input changed: ' + path)
    manifest = read(root / 'features_manifest.json')
    if not manifest['complete'] or len(manifest['records']) != 166:
        raise ValueError('Feature preparation incomplete')
    if not (root / 'corruption_manifest.json').exists():
        raise ValueError('Corruption preparation incomplete')
    return study, manifest


def summarize_folds(output, config):
    folds = [read(output / f'fold_{fold}.json') for fold in range(5)]
    records = [r for f in folds for r in f['predictions']]
    if len(records) != 166 or len({r['uid'] for r in records}) != 166:
        raise ValueError('Incomplete or repeated validation recordings')
    values = [f['metrics']['macro_f1'] for f in folds]
    summary = {'name': config['name'], 'aliases': config['aliases'], 'classes': CLASSES,
        'num_folds': 5, 'recordings': 166, 'fold_macro_f1': values,
        'mean_macro_f1': float(np.mean(values)), 'std_macro_f1': float(np.std(values, ddof=1)),
        'out_of_fold': score([r['label'] for r in records], [r['prediction'] for r in records]),
        'trainable_params': folds[0]['trainable_params'], 'shuffle': {}, 'corruption': {},
        'inference_ms_per_recording': float(np.average([f['inference_ms_per_recording'] for f in folds], weights=[len(f['predictions']) for f in folds])),
        'hardware': sorted({f['hardware'] for f in folds})}
    for diagnostic in ('shuffle', 'corruption'):
        for key in folds[0][diagnostic]:
            summary[diagnostic][key] = score([v for f in folds for v in f[diagnostic][key]['truth']], [v for f in folds for v in f[diagnostic][key]['predictions']])
    if summary['shuffle']:
        summary['shuffled_macro_f1_mean'] = float(np.mean([v['macro_f1'] for v in summary['shuffle'].values()]))
    atomic_json(output / 'summary.json', summary)
    return summary


def evaluate_fold(model, train, validation, root, config, device, fold, selected_epoch, selected_step, checkpoint):
    truth, pred = predict(model, validation, device)
    train_truth, train_pred = predict(model, train, device)
    result = {'fold': fold, 'selected_step': selected_step, 'selected_epoch': selected_epoch,
        'metrics': score(truth, pred), 'training_macro_f1': score(train_truth, train_pred)['macro_f1'],
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad), 'checkpoint': str(checkpoint),
        'predictions': [{**r, 'prediction': p} for r, p in zip(validation.records, pred)],
        'shuffle': {}, 'corruption': {}, 'inference_ms_per_recording': benchmark(model, validation, device),
        'hardware': torch.cuda.get_device_name() if str(device).startswith('cuda') else 'cpu'}
    if config['name'] in ('global', 'positions', 'word_pair', 'word_context'):
        for seed in SHUFFLE_SEEDS:
            y, p = predict(model, WordDataset(validation.records, validation.cache, config, root, shuffle_seed=seed), device)
            result['shuffle'][str(seed)] = {'truth': y, 'predictions': p, 'metrics': score(y, p)}
    if config['name'] in ('global', 'gated'):
        for rate in RATES:
            for seed in CORRUPTION_SEEDS:
                y, p = predict(model, WordDataset(validation.records, validation.cache, config, root, corruption=(rate, seed)), device)
                result['corruption'][f'{rate}_{seed}'] = {'truth': y, 'predictions': p, 'metrics': score(y, p)}
    return result


def checkpoint_candidate(path, journal, model, value, best, step, epoch):
    """A failed validation improvement appends evidence without rewriting weights."""
    improves = best is None or value > best
    with journal.open('a') as stream:
        stream.write(json.dumps({'completed_steps': step, 'epoch': epoch, 'validation_macro_f1': value,
            'previous_best_validation_macro_f1': best, 'saved': improves,
            'reason': 'strict_improvement' if improves else 'validation_did_not_strictly_improve'}) + '\n')
    if improves:
        atomic_tensor(path, {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            'selected_epoch': epoch, 'selected_step': step, 'validation_macro_f1': value})
    return value if improves else best


def run(root, experiment, device='cuda'):
    root = Path(root).resolve()
    study, manifest = validate_inputs(root)
    config = study['matrix'][int(experiment)] if str(experiment).isdigit() else next(c for c in study['matrix'] if c['name'] == experiment)
    output = root / 'experiments' / config['name']
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        print('Already complete: ' + config['name'], flush=True)
        return
    settings = study['settings']
    atomic_json(output / 'config.json', {**config, 'baseline_id': 'T25', 'training_settings': settings,
        'transformer_defaults': {'blocks': 2, 'expansion': 4, 'dropout': .3} if config['name'] == 'transformer' else None})
    cache = {r['uid']: torch.load(r['features'], weights_only=True) for r in manifest['records']}
    for split in read(root / 'splits.json'):
        fold = split['fold']
        result_path = output / f'fold_{fold}.json'
        if result_path.exists():
            continue
        random.seed(settings['random_seed'])
        np.random.seed(settings['random_seed'])
        torch.manual_seed(settings['torch_seed'])
        torch.cuda.manual_seed_all(settings['torch_seed'])
        train = WordDataset(split['train'], cache, config, root)
        validation = WordDataset(split['validation'], cache, config, root)
        model = PaperModel(config, settings, device).to(device)
        if config['name'] == 'global':
            original = read(Path(study['baseline_results']) / f'fold_{fold}.json')
            checkpoint = Path(original['checkpoint'])
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(payload['model'], strict=True)
            selected_epoch, selected_step = original['selected_epoch'], original['selected_step']
        else:
            checkpoint = output / f'fold_{fold}.pt'
            journal = output / f'fold_{fold}_checkpoint_decisions.jsonl'
            # An unfinished fold restarts from its seed; completed folds are immutable.
            if journal.exists():
                journal.unlink()
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
            counts = np.bincount([r['label'] for r in split['train']], minlength=2)
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(train) / counts, device=device, dtype=torch.float32))
            loader = DataLoader(train, batch_size=settings['training_batch_size'], shuffle=True, collate_fn=WordDataset.collate)
            best, step = None, 0
            def consider(epoch):
                nonlocal best
                # Production evaluates shuffled training data before validation.
                for batch in loader:
                    model.eval()
                    with torch.no_grad():
                        model(*(x.to(device) for x in batch[:-1]))
                y, p = predict(model, validation, device)
                best = checkpoint_candidate(checkpoint, journal, model, score(y, p)['macro_f1'], best, step, epoch)
                model.train()
            for epoch in range(settings['max_epochs']):
                model.train()
                losses = []
                for batch in loader:
                    *inputs, target = batch
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(*(x.to(device) for x in inputs)), target.to(device))
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite training loss')
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    step += 1
                    if step % settings['eval_and_save_best_model_every'] == 0:
                        consider(epoch)
                print(f"{config['name']} fold={fold} epoch={epoch + 1}/60 loss={np.mean(losses):.5f} best={best}", flush=True)
            consider(settings['max_epochs'] - 1)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(payload['model'])
            selected_epoch, selected_step = payload['selected_epoch'], payload['selected_step']
            del optimizer
        result = evaluate_fold(model, train, validation, root, config, device, fold, selected_epoch, selected_step, checkpoint)
        if config['name'] == 'global':
            if {r['uid']: r['prediction'] for r in original['predictions']} != {r['uid']: r['prediction'] for r in result['predictions']}:
                raise ValueError('Reused baseline checkpoint predictions changed')
            if abs(result['training_macro_f1'] - original['training_macro_f1']) > 1e-12:
                raise ValueError('Reused baseline training F1 changed')
            result['reused_from'] = study['baseline_results']
        atomic_json(result_path, result)
        del model, payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_folds(output, config)
    print(json.dumps({'name': config['name'], 'mean_macro_f1': summary['mean_macro_f1'], 'completed_folds': 5}), flush=True)


def build_sheet(root):
    """Aggregate only completed folds; never treat queued scores as zero."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import statistics
    from utils.adresso_test_report import TEST_HEADERS, test_columns
    root = Path(root).resolve()
    study = read(root / 'study.json')
    template = read(root / 'sheet_template.json')
    parameter_counts = read(root / 'model_parameter_counts.json') if (root / 'model_parameter_counts.json').exists() else {}
    headers = template['headers'] + ['Reference table rows', 'Pooled OOF macro-F1', 'cn F1', 'ad F1'] + [f'Fold {i} macro-F1' for i in range(5)] + [
        'Inference (ms/recording)', 'Shuffled OOF macro-F1 mean', 'Shuffle drop (pp)',
        'Corruption rate (%)', 'Standard OOF macro-F1 mean', 'Gated OOF macro-F1 mean', 'Gated gain (pp)']
    stamp = datetime.now(ZoneInfo('Europe/Madrid')).isoformat(timespec='seconds')
    rows, summaries = [], {}
    for config in study['matrix']:
        name = config['name']
        folder = root / 'experiments' / name
        folds = [read(p) for p in sorted(folder.glob('fold_*.json'))]
        summary = read(folder / 'summary.json') if (folder / 'summary.json').exists() else None
        if summary:
            summaries[name] = summary
        if name == 'global' and not folds:
            folds = [read(Path(study['baseline_results']) / f'fold_{i}.json') for i in range(5)]
        train = [f['training_macro_f1'] for f in folds]
        val = [f['metrics']['macro_f1'] for f in folds]
        mean = lambda v: statistics.mean(v) if v else ''
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else ''
        row = list(template['baseline_row']) + [''] * (len(headers) - 24)
        row[0] = 'B0 — T25 selected baseline' if name == 'global' else name
        changes = ', '.join(f'{k}={v}' for k, v in config.items() if k not in ('name', 'aliases') and v != study['matrix'][0][k])
        row[1] = 'Selected highest mean validation macro-F1 from hyperparameter tuning new; reuse original checkpoints.' if name == 'global' else 'T25 reference; changes: ' + changes
        row[2:8] = [f'{len(folds)}/5', mean(train), sd(train), mean(val), sd(val), mean(train) - mean(val) if folds else '']
        row[8] = summary['trainable_params'] if summary else parameter_counts.get(name, folds[0].get('trainable_params', row[8]) if folds else '')
        row[11] = 'Complete' if summary else 'Complete CV; diagnostics pending' if name == 'global' else 'Running (provisional)' if folder.exists() else 'Queued'
        if not summary and (folder / 'failed.json').exists():
            row[11] = 'Failed / incomplete'
        row[20] = stamp
        row[21] = str(folder)
        row[22] = '; '.join(f"{f['fold']}: {f['selected_epoch']}" for f in folds)
        row[23] = 'DistilBERT + Wav2Vec2-base-960h; T25 settings; 166 development recordings; official test excluded from development. Separate test scores, when available, are in the test columns. F1 values are fractions; SD is sample SD.'
        if name == 'transformer':
            row[23] += ' Required new defaults: 2 transformer blocks, expansion 4, block dropout 0.3.'
        if name.startswith('timing_'):
            row[23] += ' All timing conditions include the same 3→768 projection; inactive channels zeroed. No diarization.'
        if config['prefix']:
            row[23] += ' Prefix of full-context embeddings; later context may enter upstream encoders.'
        if config.get('shared_alignment_mask'):
            row[23] += ' Alignment family shares the six-variant intersection mask; compare to alignment_control, not full-mask B0.'
        row[24] = ', '.join(config['aliases'])
        if summary:
            row[25:29] = [summary['out_of_fold']['macro_f1'], summary['out_of_fold']['per_class_f1']['cn'], summary['out_of_fold']['per_class_f1']['ad'], *summary['fold_macro_f1'][:1]]
            row[28:33] = summary['fold_macro_f1']
            row[33] = summary['inference_ms_per_recording']
            if 'shuffled_macro_f1_mean' in summary:
                row[34:36] = [summary['shuffled_macro_f1_mean'], 100 * (summary['out_of_fold']['macro_f1'] - summary['shuffled_macro_f1_mean'])]
        elif folds:
            for f in folds:
                row[28 + f['fold']] = f['metrics']['macro_f1']
            if name == 'global':
                clean = read(Path(study['baseline_results']) / 'summary.json')['out_of_fold']
                row[25:28] = [clean['macro_f1'], clean['per_class_f1']['cn'], clean['per_class_f1']['ad']]
        rows.append(row)
    for rate in (0, *RATES):
        row = [''] * len(headers)
        row[0:3] = [f'R-{rate}', 'Clean-training evaluation; balanced substitutions/deletions/insertions; DistilBERT re-encoding; seeds 1701–1703.', '']
        row[11] = 'Awaiting baseline/gated diagnostics'
        row[20:22] = [stamp, str(root / 'experiments')]
        row[23:25] = ['English fixed vocabulary, 200-token chunks, observed Whisper confidence; insertions use recording-mean confidence, no error flags.', f'R-{rate} (encoder duplicates collapsed)']
        row[36] = rate
        for name, column in (('global', 37), ('gated', 38)):
            if name in summaries:
                s = summaries[name]
                row[column] = (s['out_of_fold']['macro_f1'] if rate == 0 else
                               statistics.mean(s['corruption'][f'{rate}_{seed}']['macro_f1'] for seed in CORRUPTION_SEEDS))
        if 'global' in summaries and 'gated' in summaries:
            row[39] = 100 * (row[38] - row[37])
            row[11] = 'Complete'
        elif 'global' in summaries:
            row[11] = 'Standard complete; gated diagnostics pending'
        elif 'gated' in summaries:
            row[11] = 'Gated complete; standard diagnostics pending'
        rows.append(row)
    # This already-completed original-feature condition is an input control.
    old_results = Path(study['baseline_results'].replace('adresso_architecture_sweep_words_', 'adresso_architecture_sweep_'))
    if (old_results / 'summary.json').exists():
        fs = [read(old_results / f'fold_{i}.json') for i in range(5)]
        summary = read(old_results / 'summary.json')
        row = list(template['baseline_row']) + [''] * (len(headers) - 24)
        train, val = [f['training_macro_f1'] for f in fs], [f['metrics']['macro_f1'] for f in fs]
        row[0:2] = ['Input control — original token T25', 'Same T25 architecture/training; original token features. Reuse completed original-search result.']
        row[2:8] = ['5/5', mean(train), sd(train), mean(val), sd(val), mean(train) - mean(val)]
        row[11], row[20], row[21] = 'Complete (reused control)', stamp, str(old_results)
        row[22] = '; '.join(f"{f['fold']}: {f['selected_epoch']}" for f in fs)
        row[23] = 'Input representation control only; changes token/word pooling, alignment and masks together. Original results preserved.'
        row[24] = 'Baseline diagnosis: Original-feature control'
        row[25:28] = [summary['out_of_fold']['macro_f1'], summary['out_of_fold']['per_class_f1']['cn'], summary['out_of_fold']['per_class_f1']['ad']]
        row[28:33] = val
        rows.append(row)
    for identity, reason in [('E-B / E-M', 'Encoder comparison omitted: only DistilBERT is authorized; E-D reuses B0.'),
            ('X-W0 / X-WA / X-A0 / X-AW', 'Optional transfer experiments were excluded in the reference and remain excluded.'),
            ('BETO duplicate table rows', 'Same interventions reproduced once with DistilBERT; no BETO/MrBERT training.')]:
        row = [''] * len(headers)
        row[0], row[1], row[11], row[20], row[24] = identity, reason, 'Not applicable / excluded', stamp, identity
        rows.append(row)
    if (root / 'sheet_job_ids.json').exists():
        job_ids = read(root / 'sheet_job_ids.json')
        headers.append('Job ID')
        for row in rows:
            row.append(job_ids.get(row[0], 'N/A'))
    headers.extend(TEST_HEADERS)
    for row in rows:
        source = study['baseline_results'] if row[0] == 'B0 — T25 selected baseline' else row[21]
        row.extend(test_columns(source))
    report = {'headers': headers, 'rows': rows, 'completed_configurations': len(summaries), 'expected_configurations': len(study['matrix'])}
    atomic_json(root / 'spreadsheet_export.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--features-root', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--experiment')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'plan':
        plan(args.root, args.baseline, args.features_root, args.dataset_root)
    elif args.action == 'run':
        try:
            run(args.root, args.experiment, args.device)
        except Exception as error:
            study = read(args.root / 'study.json')
            config = study['matrix'][int(args.experiment)] if str(args.experiment).isdigit() else next(c for c in study['matrix'] if c['name'] == args.experiment)
            atomic_json(args.root / 'experiments' / config['name'] / 'failed.json', {'error': str(error)})
            raise
    else:
        print(json.dumps(build_sheet(args.root), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()


