"""Separate, resumable DeBERTa-v3-large/WavLM-large ADReSSo word exports.

The trained English Wav2Vec2 CTC aligner supplies boundaries independently of
WavLM. Encoders run in separate phases; full recordings are never truncated or
split to fit GPU memory. A CUDA OOM retries the same float32 computation on CPU.
"""

import argparse
import fcntl
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor

from utils.batch_adresso_embeddings import load_english_ctc, write_json
from utils.batch_adresso_words import inputs, verify
from utils.batch_word_embeddings import completed_stage, publish_file, validate_package
from utils.compare_word_alignment import inspect_word_alignment
from utils.ctc_forced_alignment import export_alignment
from utils.ctc_word_windows import export_word_windows
from utils.package_word_embeddings import package_word_embeddings
from utils.prepro_adresso_recording import identity
from utils.prepro_audio_frames import export_recording_frames
from utils.prepro_ctc import export_ctc
from utils.prepro_word_text import export_text_words
from utils.word_audio_pool import export_audio_words


TEXT_MODEL = 'microsoft/deberta-v3-large'
AUDIO_MODEL = 'microsoft/wavlm-large'
REVISIONS = {
    TEXT_MODEL: '64a8c8eab3e352a784c658aef62be1662607476f',
    AUDIO_MODEL: 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c',
}


def checkpoint_options(root, model_id):
    return {'revision': REVISIONS[model_id], 'local_files_only': True,
            'cache_dir': str(root / 'model_cache')}


def load_model(root, model_id, device):
    model, loading = AutoModel.from_pretrained(
        model_id, output_loading_info=True, **checkpoint_options(root, model_id))
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError(f'Incomplete pretrained {model_id} weights: {loading}')
    if model.config.hidden_size != 1024:
        raise ValueError('Expected 1024-dimensional large-encoder hidden states')
    return model.float().requires_grad_(False).to(device).eval()


def full_recording_with_cpu_fallback(action, model, device):
    """Keep full context and precision when a recording exceeds GPU memory."""
    try:
        return action(device)
    except torch.cuda.OutOfMemoryError:
        if device != 'cuda':
            raise
    # Exit the exception handler before retrying, releasing traceback tensors.
    print('  GPU memory exceeded; retrying the complete recording on CPU (float32)', flush=True)
    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    try:
        return action('cpu')
    finally:
        model.to(device)


def record_key(record):
    return ('train' if record['partition'] == 'train' else 'test') + '/' + record['uid']


def run(args):
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args, root)


def prepare(args, root):
    torch.set_num_threads(args.threads)
    torch.manual_seed(3407)
    index, records, old_config = inputs(args)
    config = {**old_config, 'format': 'adresso_large_ctc_word_batch_v1',
              'text_model': TEXT_MODEL, 'text_revision': REVISIONS[TEXT_MODEL],
              'audio_model': AUDIO_MODEL, 'audio_revision': REVISIONS[AUDIO_MODEL],
              'ctc_revision': old_config['audio_revision'],
              'acoustic_layer': 'last_hidden_state', 'precision': 'float32',
              'acoustic_context': 'whole_recording; CPU fallback on CUDA OOM; no chunking',
              'text_pooling': 'mean_subwords', 'audio_pooling': 'mean_frames',
              'torch_seed': 3407}
    selected = [r for r in records if not args.uid or r['uid'] in args.uid]
    if args.uid and set(args.uid) - {r['uid'] for r in selected}:
        raise ValueError('Requested UID not in ADReSSo')
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True,
                                             **checkpoint_options(root, TEXT_MODEL))
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer with original word IDs is required')
    # Audit actual DeBERTa tokenization before expensive acoustic inference.
    from scripts.word_text_encoder import word_chunks
    from utils.prepro_ctc import read_words
    for record in records:
        word_chunks(read_words(record['words']), tokenizer, args.chunk_size)
    print(f'Audited {len(index.training)} training / {len(index.test)} test recordings; '
          f'{sum(len(r["skipped_words"]) for r in records)} unsupported CTC words masked', flush=True)
    if args.dry_run:
        return
    config_path = root / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Preparation settings changed; use a new output root')
    if not config_path.exists():
        write_json(config_path, config)
    state_path = root / 'summary.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'records': {}}
    state.update(status='running', training_ready=False, expected_training=len(index.training),
                 expected_test=len(index.test), selected_uids=[r['uid'] for r in selected])
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    def save():
        state['completed'] = sum(r.get('status') == 'complete' for r in state['records'].values())
        state['failed'] = sum(r.get('status') == 'failed' for r in state['records'].values())
        write_json(state_path, state)

    def stage(record, name, action):
        return completed_stage(root / 'recordings' / record_key(record), name,
                               {'settings': config, 'sources': record['source_sha256']}, action)

    for record in selected:
        previous = state['records'].get(record_key(record), {})
        if previous.get('status') == 'complete':
            if previous['source_sha256'] != record['source_sha256']:
                raise ValueError('Completed recording sources changed')
            destination = root / record_key(record).split('/')[0] / 'embeddings'
            validate_package(destination, record, config)
            for name, checksum in previous['file_sha256'].items():
                if identity(destination / name)['sha256'] != checksum:
                    raise ValueError('Completed package changed')
        else:
            state['records'][record_key(record)] = {'status': 'pending'}
    save()

    # Separate encoder phases avoid keeping three large models on the GPU.
    for phase in ('alignment', 'audio', 'text'):
        print(f'=== {phase} phase ===', flush=True)
        model = processor = ctc = None
        for number, record in enumerate(selected, 1):
            item = state['records'][record_key(record)]
            if item['status'] in ('complete', 'failed'):
                continue
            print(f'[{number}/{len(selected)}] {record_key(record)}', flush=True)
            try:
                if phase == 'alignment':
                    def emissions(out):
                        nonlocal ctc
                        if ctc is None:
                            ctc = load_english_ctc(device, True, config['ctc_revision'])
                            ctc.skip_unsupported_words = True
                        export_ctc(record['audio'], record['words'], out, ctc)
                    emissions_dir = stage(record, 'step3_ctc', emissions)
                    alignment = stage(record, 'step4_alignment', lambda out: export_alignment(emissions_dir, out))
                    windows = stage(record, 'step5_windows', lambda out: export_word_windows(alignment, out))
                    comparison = stage(record, 'step6_comparison', lambda out: inspect_word_alignment(windows, out))
                    item.update(status='aligned', windows=str(windows), comparison_report=str(comparison / 'comparison.json'))
                elif phase == 'audio':
                    def frames(out):
                        nonlocal processor, model
                        if model is None:
                            processor = Wav2Vec2FeatureExtractor.from_pretrained(
                                AUDIO_MODEL, **checkpoint_options(root, AUDIO_MODEL))
                            model = load_model(root, AUDIO_MODEL, device)
                        return full_recording_with_cpu_fallback(
                            lambda target: export_recording_frames(record['audio'], out, processor, model,
                                                                  target, model_id=AUDIO_MODEL), model, device)
                    audio = stage(record, 'step2_audio_frames', frames)
                    words = stage(record, 'step7_audio_words', lambda out: export_audio_words(audio, item['windows'], out))
                    item.update(status='audio_ready', audio_words=str(words))
                else:
                    def text(out):
                        nonlocal model
                        if model is None:
                            model = load_model(root, TEXT_MODEL, device)
                        export_text_words(item['audio_words'], out, tokenizer, model, TEXT_MODEL, device, args.chunk_size)
                    linguistic = stage(record, 'step8_text_words', text)
                    package = stage(record, 'step9_package', lambda out: package_word_embeddings(item['audio_words'], linguistic, out))
                    metadata, _ = validate_package(package, record, config)
                    if metadata['audio_feature_shape'][1] != 1024 or metadata['text_feature_shape'][1] != 1024:
                        raise ValueError('Unexpected feature width')
                    for source in record['source_sha256'].values():
                        if identity(source['path']) != source:
                            raise ValueError('Input changed during extraction')
                    destination = root / record_key(record).split('/')[0] / 'embeddings'
                    names = [*metadata['output_sha256'], record['uid'] + '_word_metadata.json']
                    for name in names:
                        publish_file(package / name, destination / name)
                    validate_package(destination, record, config)
                    item.update(status='complete', source_sha256=record['source_sha256'],
                                file_sha256={name: identity(destination / name)['sha256'] for name in names},
                                audio_shape=metadata['audio_feature_shape'], text_shape=metadata['text_feature_shape'],
                                valid_paired_words=metadata['valid_paired_words'], skipped_words=record['skipped_words'])
                    print(f'  exported audio {item["audio_shape"]}, text {item["text_shape"]}', flush=True)
            except Exception as error:
                item.update(status='failed', failed_phase=phase, error=f'{type(error).__name__}: {error}')
                print(f'  FAILED: {item["error"]}', flush=True)
                if device == 'cuda':
                    torch.cuda.empty_cache()
            save()
        model = processor = ctc = None
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    complete = all(state['records'].get(record_key(r), {}).get('status') == 'complete' for r in records)
    selected_complete = all(state['records'][record_key(r)]['status'] == 'complete' for r in selected)
    state['status'] = 'complete' if complete else 'pilot_complete' if selected_complete and args.uid else 'incomplete'
    if complete:
        state['readiness'] = verify(index, root)
        state['training_ready'] = True
        write_json(root / 'alignment_verification.json', {
            'status': 'passed', 'config': config, 'checked_recordings': len(records),
            'limitation': 'Structural/loader validation; CTC boundary accuracy is not ground truth'})
    save()
    if not selected_complete:
        raise RuntimeError('Word preparation incomplete; see summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--uid', action='append', help='Prepare selected recordings before the full cohort')
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
