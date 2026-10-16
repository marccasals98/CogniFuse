"""Resumable full-cohort ADReSSo exports using the verified recording exporter."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from scripts.adresso import ADReSSoIndex
from scripts.ctc_encoder import SpanishCTCEncoder
from scripts.data import PrecomputedADDataset, make_precomputed_dataset
from scripts.cross_validation import plan_folds
from scripts.settings import TRAIN_DEFAULT_SETTINGS
from utils.batch_word_embeddings import publish_file
from utils.ctc_forced_alignment import export_alignment
from utils.ctc_word_windows import export_word_windows
from utils.prepro_ctc import export_ctc
from utils.prepro_adresso_recording import (
    TEXT_MODEL, AUDIO_MODEL, identity, prepare_recording, export_recording,
)
from utils.prepro_embeddings import build_audio_encoder


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def english_vocabulary(vocabulary):
    """Alias English uppercase letters to the existing lowercase normalizer."""
    result = {key.lower() if len(key) == 1 else key: value for key, value in vocabulary.items()}
    if len(result) != len(vocabulary):
        raise ValueError('English CTC vocabulary aliases are ambiguous')
    return result


def load_english_ctc(device, local_files_only, revision):
    # The existing wrapper also accepts an explicit trained English checkpoint.
    encoder = SpanishCTCEncoder.from_pretrained(AUDIO_MODEL, device=device,
                                               local_files_only=local_files_only, revision=revision)
    encoder.vocabulary = english_vocabulary(encoder.vocabulary)
    return encoder


def repair_timestamps(args, index, record, attempt, encoder, reason):
    """Force-align the unchanged supplied transcript, writing new timestamps."""
    split = 'train' if record['partition'] == 'train' else 'test'
    transcript_path = index.root / record['partition'] / 'text/text_transcriptions.csv'
    with transcript_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['uid'] == record['uid']]
    if len(rows) != 1:
        raise ValueError('Repair requires exactly one supplied transcript row')
    column = 'transcription_pause' if args.pauses else 'transcription'
    words = [w for w in re.findall(r'\S+', rows[0][column]) if any(c.isalnum() for c in w)]
    # Validate all targets before inference. Unsupported characters still fail.
    encoder.normalize_words(words)
    attempt.mkdir(parents=True, exist_ok=True)
    targets_path = attempt / 'supplied_transcript_words.csv'
    with targets_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['word'])
        writer.writerows([[word] for word in words])
    export_ctc(record['audio_path'], targets_path, attempt / 'ctc', encoder)
    alignment = export_alignment(attempt / 'ctc', attempt / 'alignment')
    windows = export_word_windows(attempt / 'alignment', attempt / 'windows')
    if windows['aligned_word_count'] != len(words):
        raise ValueError('CTC repair did not align every supplied lexical word')
    repaired = attempt / 'repaired_words.csv'
    with repaired.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['word', 'start', 'end'])
        writer.writeheader()
        for row in windows['word_windows']:
            writer.writerow({'word': row['word'], 'start': row['start_time'], 'end': row['end_time']})
    prepared = prepare_recording(args.dataset_root, record['uid'], split, args.pauses,
                                 words_override=repaired, index=index)
    prepared['sources']['ctc_alignment'] = identity(attempt / 'alignment/alignment.json')
    prepared['timestamp_repair'] = {'method': 'English CTC forced alignment of unchanged supplied transcript',
                                  'reason': reason, 'model': AUDIO_MODEL,
                                  'revision': getattr(encoder.model.config, '_commit_hash', None),
                                  'vocabulary_alias': 'uppercase English letters mapped to lowercase target keys',
                                  'lexical_words': len(words), 'path_log_score': alignment['path_log_score'],
                                  'note': 'Structural alignment passed; acoustic boundary accuracy is not established'}
    return prepared


def file_names(uid, pauses):
    stem = uid + ('distil_pauses' if pauses else 'distil')
    return [stem + suffix for suffix in ('.pt', '_mask.pt', '_audio.pt', '_audio_mask.pt', '_extraction.json')]


def validate_export(directory, record, config, expected_hashes=None):
    directory = Path(directory)
    names = file_names(record['uid'], config['pauses'])
    metadata = json.loads((directory / names[-1]).read_text())
    for key, expected in (('uid', record['uid']), ('partition', record['partition']), ('label', record['label']),
                          ('text_model', TEXT_MODEL), ('audio_model', AUDIO_MODEL), ('chunk_size', config['chunk_size']),
                          ('text_model_revision', config['text_revision']), ('audio_model_revision', config['audio_revision'])):
        if metadata[key] != expected:
            raise ValueError(f'Reused export has different {key}')
    column = 'transcription_pause' if config['pauses'] else 'transcription'
    if metadata['transcript_column'] != column or metadata['truncated_tokens'] != 0:
        raise ValueError('Reused export has different transcript selection or truncation')
    if metadata.get('timestamp_repair') and not config['repair_timestamps']:
        raise ValueError('Strict batch cannot reuse regenerated timestamps')
    for source in metadata['sources'].values():
        if identity(source['path']) != source:
            raise ValueError('Reused export source hash changed')
    text = PrecomputedADDataset._load_features(directory / names[0])
    tm = PrecomputedADDataset._load_mask(directory / names[1], len(text))
    audio = PrecomputedADDataset._load_features(directory / names[2])
    am = PrecomputedADDataset._load_mask(directory / names[3], len(audio))
    if (audio.shape != text.shape or list(audio.shape) != metadata['audio_shape']
            or list(text.shape) != metadata['text_shape'] or audio.shape[1] != 768
            or int(am.sum()) != metadata['audio_valid_positions']
            or int(tm.sum()) != metadata['text_valid_positions'] or (audio[~am] != 0).any()):
        raise ValueError('Reused tensor/mask dimensions or layout disagree with metadata')
    hashes = {name: identity(directory / name)['sha256'] for name in names}
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError('Completed export file hash changed')
    return metadata, hashes


def verify_cohort(args, index, root):
    params = SimpleNamespace(**{**TRAIN_DEFAULT_SETTINGS, 'dataset': 'adresso',
        'dataset_root': str(index.root), 'number_classes': 2, 'num_folds': args.num_folds,
        'precomputed_features_dir': str(root / 'train/embeddings'),
        'precomputed_test_features_dir': str(root / 'test/embeddings'),
        'precomputed_audio_suffix': ('distil_pauses_audio.pt' if args.pauses else 'distil_audio.pt'),
        'precomputed_text_suffix': ('distil_pauses.pt' if args.pauses else 'distil.pt')})
    plans = plan_folds(params)
    count = 0
    for split in ('train', 'val', 'test'):
        dataset = make_precomputed_dataset(params, split=split, fold=0)
        for item in dataset:
            PrecomputedADDataset.collate_fn([item])
            count += 1
    if count != len(index.training) + len(index.test):
        raise ValueError('Loader did not read every recording exactly once in readiness check')
    return {'folds': plans, 'loader_checked_recordings': count, 'test_labels_available': False}


def run_batch(args):
    index = ADReSSoIndex(args.dataset_root)
    index.development_folds(args.num_folds)
    records = index.training + index.test
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {'format': 'adresso_batch_v1', 'dataset_root': str(index.root), 'pauses': args.pauses,
              'chunk_size': args.chunk_size, 'repair_timestamps': args.repair_timestamps,
              'text_revision': AutoConfig.from_pretrained(TEXT_MODEL, local_files_only=args.local_files_only)._commit_hash,
              'audio_revision': AutoConfig.from_pretrained(AUDIO_MODEL, local_files_only=args.local_files_only)._commit_hash,
              'split_sources': [identity(p) for p in sorted((index.root / 'train/splits').glob('*.npy'))]}
    config_path = root / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Batch configuration changed; use a new output root')
    else:
        write_json(config_path, config)
    state_path = root / 'summary.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'records': {}}
    state.update({'expected_training': len(index.training), 'expected_test': len(index.test),
                  'training_ready': False, 'status': 'running'})
    state.pop('readiness', None)
    state.pop('readiness_error', None)
    write_json(state_path, state)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True, local_files_only=args.local_files_only,
                                             revision=config['text_revision'])
    if not tokenizer.num_special_tokens_to_add(pair=False) < args.chunk_size <= tokenizer.model_max_length:
        raise ValueError('Invalid text chunk size')
    text_model = audio_encoder = ctc = None
    for number, record in enumerate(records, 1):
        split = 'train' if record['partition'] == 'train' else 'test'
        uid, key = record['uid'], split + '/' + record['uid']
        target = root / split / 'embeddings'
        target.mkdir(parents=True, exist_ok=True)
        print(f'[{number}/{len(records)}] {key}', flush=True)
        try:
            previous = state['records'].get(key, {})
            if previous.get('status') == 'complete':
                metadata, hashes = validate_export(target, record, config, previous['file_sha256'])
                print('  validated completed export', flush=True)
                continue
            reused = None
            base = root / 'recordings' / split / uid
            candidates = list(args.reuse_recording_dir) + sorted(base.glob('attempt_*/export'), reverse=True)
            for candidate in candidates:
                if (candidate / file_names(uid, args.pauses)[-1]).is_file():
                    metadata, hashes = validate_export(candidate, record, config)
                    reused = candidate
                    break
            if reused is None:
                attempt = base / f"attempt_{len(list(base.glob('attempt_*'))) + 1}"
                try:
                    prepared = prepare_recording(args.dataset_root, uid, split, args.pauses, index=index)
                except ValueError as error:
                    if not args.repair_timestamps:
                        raise
                    if ctc is None:
                        ctc = load_english_ctc(device, args.local_files_only, config['audio_revision'])
                    prepared = repair_timestamps(args, index, record, attempt, ctc, str(error))
                if text_model is None:
                    text_model = AutoModel.from_pretrained(TEXT_MODEL, local_files_only=args.local_files_only,
                                                          revision=config['text_revision']).to(device).eval()
                    audio_encoder = build_audio_encoder('wav2vec2', device, args.local_files_only)
                reused = attempt / 'export'
                export_recording(prepared, reused, tokenizer, text_model, audio_encoder, device, args.chunk_size)
                metadata, hashes = validate_export(reused, record, config)
            for name in file_names(uid, args.pauses):
                publish_file(reused / name, target / name)
            validate_export(target, record, config, hashes)
            state['records'][key] = {'status': 'complete', 'file_sha256': hashes,
                                     'timestamp_repaired': bool(metadata.get('timestamp_repair')),
                                     'export_source': str(reused), 'shape': metadata['audio_shape']}
            print('  exported with valid masks' + ('; English CTC timestamps' if metadata.get('timestamp_repair') else ''), flush=True)
        except Exception as error:
            state['records'][key] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
            print('  FAILED: ' + str(error), flush=True)
            if device == 'cuda':
                torch.cuda.empty_cache()
        state['completed'] = sum(r['status'] == 'complete' for r in state['records'].values())
        state['failed'] = sum(r['status'] == 'failed' for r in state['records'].values())
        write_json(state_path, state)
    state['status'] = 'complete' if all(state['records'].get(('train' if r['partition'] == 'train' else 'test') + '/' + r['uid'], {}).get('status') == 'complete' for r in records) else 'incomplete'
    if state['status'] == 'complete':
        try:
            state['readiness'] = verify_cohort(args, index, root)
            state['training_ready'] = True
        except Exception as error:
            state['readiness_error'] = f'{type(error).__name__}: {error}'
            state['status'] = 'incomplete'
    state['timestamp_repairs'] = sum(r.get('timestamp_repaired', False) for r in state['records'].values())
    write_json(state_path, state)
    print(f"Completed {state.get('completed', 0)}/{len(records)}; training_ready={state['training_ready']}", flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--pauses', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--repair-timestamps', action='store_true')
    parser.add_argument('--reuse-recording-dir', type=Path, action='append', default=[])
    parser.add_argument('--chunk-size', type=int, default=200)
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    state = run_batch(args)
    if not state['training_ready']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
