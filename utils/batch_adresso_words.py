"""Run existing CTC word-preparation stages on the unchanged ADReSSo cohort."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

# Python -m normally searches the working directory before PYTHONPATH. Honor
# the launcher's explicit snapshot before importing any preparation helpers.
if __name__ == '__main__':
    _snapshot = Path(__file__).resolve().parents[1] / 'outputs/adresso_word_repeat_launch_20261029/source'
    _frozen = _snapshot / 'utils' / Path(__file__).name
    if os.environ.get('PYTHONPATH') == str(_snapshot) and _frozen.is_file() and _frozen.resolve() != Path(__file__).resolve():
        os.execv(sys.executable, [sys.executable, '-u', str(_frozen), *sys.argv[1:]])

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, Wav2Vec2Processor

from scripts.adresso import ADReSSoIndex
from scripts.cross_validation import plan_folds
from scripts.data import PrecomputedADDataset, make_precomputed_dataset
from scripts.settings import TRAIN_DEFAULT_SETTINGS
from utils.batch_adresso_embeddings import english_vocabulary, load_english_ctc, write_json
from utils.batch_word_embeddings import audit_record, completed_stage, publish_file, validate_package
from utils.compare_word_alignment import inspect_word_alignment
from utils.ctc_forced_alignment import export_alignment
from utils.ctc_skipped_words import normalize_skipping_unsupported
from utils.ctc_word_windows import export_word_windows
from utils.package_word_embeddings import package_word_embeddings, AUDIO_SUFFIX, TEXT_SUFFIX
from utils.prepro_adresso_recording import AUDIO_MODEL, TEXT_MODEL, identity
from utils.prepro_audio_frames import export_recording_frames
from utils.prepro_ctc import export_ctc, read_words
from utils.prepro_word_text import export_text_words
from utils.word_audio_pool import export_audio_words


def inputs(args):
    index = ADReSSoIndex(args.dataset_root)
    index.development_folds(5)
    config = {
        'format': 'adresso_ctc_word_batch_v1', 'dataset_root': str(index.root),
        'text_model': TEXT_MODEL, 'audio_model': AUDIO_MODEL, 'ctc_model': AUDIO_MODEL,
        'text_revision': AutoConfig.from_pretrained(TEXT_MODEL, local_files_only=True)._commit_hash,
        'audio_revision': AutoConfig.from_pretrained(AUDIO_MODEL, local_files_only=True)._commit_hash,
        'chunk_size': args.chunk_size, 'context_frames': 0, 'skip_unsupported_words': True,
        'audio_suffix': AUDIO_SUFFIX, 'text_suffix': TEXT_SUFFIX,
        'word_policy': 'Exact original Whisper CSV words and IDs; no retranscription or lexical substitutions',
        'mask_policy': 'Audio AND text validity for both modalities; skipped CTC words retain invalid rows',
        'split_sources': [identity(p) for p in sorted((index.root / 'train/splits').glob('*.npy'))],
    }
    processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL, revision=config['audio_revision'], local_files_only=True)
    vocabulary = english_vocabulary(processor.tokenizer.get_vocab())
    records, issues = [], []
    for original in index.training + index.test:
        record = {**original, 'audio': original['audio_path'], 'skip_unsupported_words': True}
        folder = index.root / original['partition'] / 'text'
        if original['partition'] == 'train':
            folder /= original['diagnosis']
        record['words'] = str(folder / (record['uid'] + '.csv'))
        problems = audit_record(record, vocabulary, processor.tokenizer.pad_token_id)
        if problems:
            issues.append({'uid': record['uid'], 'problems': problems})
        else:
            transcript = normalize_skipping_unsupported(read_words(record['words']), vocabulary,
                                                        processor.tokenizer.pad_token_id)
            record['skipped_words'] = transcript.get('skipped_words', [])
            record['source_sha256'] = {name: identity(record[name]) for name in ('audio', 'words')}
        records.append(record)
    if issues:
        raise ValueError('Input audit failed: ' + json.dumps(issues))
    return index, records, config


def verify(index, root):
    params = SimpleNamespace(**{**TRAIN_DEFAULT_SETTINGS, 'dataset': 'adresso',
        'dataset_root': str(index.root), 'number_classes': 2,
        'precomputed_features_dir': str(root / 'train/embeddings'),
        'precomputed_test_features_dir': str(root / 'test/embeddings'),
        'precomputed_audio_suffix': AUDIO_SUFFIX, 'precomputed_text_suffix': TEXT_SUFFIX})
    plans = plan_folds(params)
    checked = 0
    for split in ('train', 'val', 'test'):
        dataset = make_precomputed_dataset(params, split=split, fold=0)
        for item in dataset:
            PrecomputedADDataset.collate_fn([item])
            checked += 1
    if checked != len(index.training) + len(index.test):
        raise ValueError('Incomplete loader coverage')
    return {'development_folds': len(plans), 'loader_checked_recordings': checked,
            'test_labels_available': False}


def run(args):
    index, records, config = inputs(args)
    omitted = sum(len(r['skipped_words']) for r in records)
    print(f'Input audit: {len(index.training)} training / {len(index.test)} test; '
          f'{omitted} unsupported words retained with invalid acoustic masks', flush=True)
    if args.dry_run:
        return
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Preparation settings changed; use a new output root')
    if not config_path.exists():
        write_json(config_path, config)
    state_path = root / 'summary.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'records': {}}
    state.update(status='running', training_ready=False, expected_training=len(index.training),
                 expected_test=len(index.test), skipped_word_count=omitted)
    write_json(state_path, state)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(args.threads)
    ctc = tokenizer = text_model = None

    def acoustic():
        nonlocal ctc
        if ctc is None:
            ctc = load_english_ctc(device, True, config['audio_revision'])
            ctc.skip_unsupported_words = True
        return ctc

    def frames(record, output):
        encoder = acoustic()
        export_recording_frames(record['audio'], output, encoder.processor, encoder.model.wav2vec2, device)

    def text(audio, output):
        nonlocal tokenizer, text_model
        if text_model is None:
            tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=config['text_revision'],
                                                     use_fast=True, local_files_only=True)
            text_model = AutoModel.from_pretrained(TEXT_MODEL, revision=config['text_revision'],
                                                  local_files_only=True).to(device).eval()
        export_text_words(audio, output, tokenizer, text_model, TEXT_MODEL, device, args.chunk_size)

    selected = [r for r in records if args.uid is None or r['uid'] == args.uid]
    if not selected:
        raise ValueError('Requested UID not in ADReSSo')
    for number, record in enumerate(selected, 1):
        split = 'train' if record['partition'] == 'train' else 'test'
        key = split + '/' + record['uid']
        print(f'[{number}/{len(selected)}] {key}', flush=True)
        try:
            destination = root / split / 'embeddings'
            previous = state['records'].get(key, {})
            if previous.get('status') == 'complete':
                if previous['source_sha256'] != record['source_sha256']:
                    raise ValueError('Completed recording source hash changed')
                validate_package(destination, record, config)
                for name, checksum in previous['file_sha256'].items():
                    if identity(destination / name)['sha256'] != checksum:
                        raise ValueError('Published word features changed')
                print('  validated completed package', flush=True)
                continue
            base = root / 'recordings' / split / record['uid']
            signature = {'settings': config, 'sources': record['source_sha256']}
            stage = lambda name, action: completed_stage(base, name, signature, action)
            audio = stage('step2_audio_frames', lambda out: frames(record, out))
            emissions = stage('step3_ctc', lambda out: export_ctc(record['audio'], record['words'], out, acoustic()))
            alignment = stage('step4_alignment', lambda out: export_alignment(emissions, out))
            windows = stage('step5_windows', lambda out: export_word_windows(alignment, out))
            comparison = stage('step6_comparison', lambda out: inspect_word_alignment(windows, out))
            words = stage('step7_audio_words', lambda out: export_audio_words(audio, windows, out))
            linguistic = stage('step8_text_words', lambda out: text(words, out))
            package = stage('step9_package', lambda out: package_word_embeddings(words, linguistic, out))
            metadata, _ = validate_package(package, record, config)
            for source in record['source_sha256'].values():
                if identity(source['path']) != source:
                    raise ValueError('Input changed during extraction')
            names = [*metadata['output_sha256'], record['uid'] + '_word_metadata.json']
            for name in names:
                publish_file(package / name, destination / name)
            validate_package(destination, record, config)
            state['records'][key] = {
                'status': 'complete', 'file_sha256': {name: identity(destination / name)['sha256'] for name in names},
                'shape': metadata['audio_feature_shape'], 'valid_paired_words': metadata['valid_paired_words'],
                'skipped_words': record['skipped_words'], 'source_sha256': record['source_sha256'],
                'comparison_report': str(comparison / 'comparison.json'),
            }
            print(f'  exported {metadata["audio_feature_shape"]}; {metadata["valid_paired_words"]} valid pairs', flush=True)
        except Exception as error:
            state['records'][key] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
            print(f'  FAILED: {error}', flush=True)
            if device == 'cuda':
                torch.cuda.empty_cache()
        state['completed'] = sum(r['status'] == 'complete' for r in state['records'].values())
        state['failed'] = sum(r['status'] == 'failed' for r in state['records'].values())
        write_json(state_path, state)
    complete = all(state['records'].get(('train' if r['partition'] == 'train' else 'test') + '/' + r['uid'], {}).get('status') == 'complete' for r in records)
    state['status'] = 'complete' if complete else 'single_recording_complete' if args.uid and not state.get('failed') else 'incomplete'
    if complete:
        state['readiness'] = verify(index, root)
        state['training_ready'] = True
        write_json(root / 'alignment_verification.json', {'status': 'passed', 'config': config,
                   'checked_recordings': state['readiness']['loader_checked_recordings'],
                   'limitation': 'Structural/loader validation; CTC boundary accuracy is not ground truth'})
    write_json(state_path, state)
    if state.get('failed') or not (complete or args.uid):
        raise RuntimeError('Word preparation incomplete; see summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--uid', help='Prepare one recording before the full cohort')
    parser.add_argument('--chunk-size', type=int, default=200)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
