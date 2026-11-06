"""Prepare, execute, and rank a reproducible ADReSSo architecture grid."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


if __name__ == '__main__':
    _snapshot = Path(__file__).resolve().parents[1] / 'outputs/adresso_word_repeat_launch_20261029/source'
    _frozen = _snapshot / 'utils' / Path(__file__).name
    if os.environ.get('PYTHONPATH') == str(_snapshot) and _frozen.is_file() and _frozen.resolve() != Path(__file__).resolve():
        os.execv(sys.executable, [sys.executable, '-u', str(_frozen), *sys.argv[1:]])


REPO = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trial_grid():
    trials = []
    for index, (heads, dropout, layers) in enumerate(itertools.product(
            (2, 4, 8), (0.0, 0.1, 0.3), (1, 2, 4))):
        trials.append({
            'index': index, 'id': f'trial_{index:02d}_h{heads}_d{dropout:g}_l{layers}',
            'seq_to_seq_heads_number': heads,
            'seq_to_seq_input_dropout': dropout,
            'seq_to_one_input_dropout': dropout,
            'classifier_layer_drop_out': dropout,
            'classifier_hidden_layers': layers,
        })
    return trials


def prepare(root, features, baseline):
    features, baseline = features.resolve(), baseline.resolve()
    feature_config = read_json(features / 'config.json')
    if not read_json(features / 'summary.json')['training_ready']:
        raise ValueError('Aligned feature cohort is not training ready')
    baseline_config = read_json(baseline / 'config.json')
    if (baseline_config['dataset'] != 'adresso'
            or Path(baseline_config['precomputed_features_dir']).resolve() != features / 'train/embeddings'
            or baseline_config['max_epochs'] != 60
            or baseline_config['num_folds'] != 5
            or baseline_config['eval_and_save_best_model_every'] != 100):
        raise ValueError('Baseline does not match the sweep feature cohort and training budget')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'slurm').mkdir()
    (root / 'trials').mkdir()
    source = root / 'source'
    for directory in ('scripts', 'utils'):
        destination = source / directory
        destination.mkdir(parents=True)
        for path in sorted((REPO / directory).glob('*.py')):
            shutil.copy2(path, destination / path.name)
    reference = root / 'baseline'
    reference.mkdir()
    for path in sorted(baseline.glob('*.json')):
        shutil.copy2(path, reference / path.name)
    fixed = {
        'dataset': 'adresso', 'dataset_root': feature_config['dataset_root'],
        'precomputed_features_dir': str(features / 'train/embeddings'),
        'precomputed_text_suffix': 'distil_pauses.pt',
        'precomputed_audio_suffix': 'distil_pauses_audio.pt',
        'num_folds': 5, 'number_classes': 2, 'random_seed': 1234, 'torch_seed': 3407,
        'speech_adapter': 'NoneAdapter', 'text_adapter': 'NoneAdapter',
        'seq_to_seq_method': 'MultiHeadAttention', 'seq_to_one_method': 'AttentionPooling',
        'classifier_hidden_layers_width': 512, 'max_epochs': 60,
        'training_batch_size': 1, 'evaluation_batch_size': 1, 'num_workers': 0,
        'training_augmentation_prob': 0, 'evaluation_augmentation_prob': 0,
        'eval_and_save_best_model_every': 100, 'print_training_info_every': 100,
        'early_stopping': 0, 'update_optimizer_every': 0,
        'loss': 'CrossEntropy', 'optimizer': 'adamw', 'learning_rate': 0.0001,
        'weight_decay': 0.01,
    }
    if not feature_config['pauses']:
        raise ValueError('This grid expects the verified pause-feature export')
    for key, value in fixed.items():
        if key in baseline_config and baseline_config[key] != value:
            # Dataset roots may be provided with or without the diagnosis suffix.
            if key != 'dataset_root':
                raise ValueError(f'Baseline differs in fixed setting {key}')
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'text_model': 'distilbert-base-uncased', 'audio_model': 'facebook/wav2vec2-base-960h',
        'ranking_metric': 'mean_macro_f1', 'tie_breaker': 'lower std_macro_f1, then trial ID',
        'dropout_policy': 'Same probability at sequence input, pooling input and classifier',
        'classifier_depth_note': 'classifier_hidden_layers counts additional hidden blocks after the input block',
        'checkpoint_policy': 'Evaluate every 100 updates and at training end; save first eligible model and strict validation macro-F1 improvements only',
        'official_test_used': False, 'max_concurrent_gpu_jobs': 2,
        'fixed_parameters': fixed, 'trials': trial_grid(),
        'baseline_results_directory': str(baseline),
        'source_sha256': {str(p.relative_to(root)): sha256(p) for p in sorted(source.rglob('*.py'))},
        'input_sha256': {str(features / name): sha256(features / name)
                         for name in ('config.json', 'summary.json', 'alignment_verification.json')},
        'split_sources': feature_config['split_sources'],
    }
    write_json(root / 'manifest.json', manifest)
    summarize(root)
    print(f'Prepared {len(manifest["trials"])} trials / 135 fold models: {root}', flush=True)
    return manifest


def build_command(root, manifest, trial):
    directory = root / 'trials' / trial['id']
    options = {**manifest['fixed_parameters'],
               **{key: value for key, value in trial.items() if key not in ('id', 'index')},
               'model_output_folder': str(directory / 'models'),
               'log_file_folder': str(directory / 'logs'),
               'cross_validation_output_dir': str(directory / 'results')}
    command = [sys.executable, str(root / 'source/scripts/train.py')]
    for key, value in options.items():
        command.extend(['--' + key, str(value)])
    command.extend(['--cross_validate', '--no-skip_connections', '--weighted_loss',
                    '--no-load_checkpoint', '--no-use_weights_and_biases'])
    return command


def prepare_word_repeat(root, features, original):
    """Repeat the exact original grid after verified CTC word preparation."""
    features, original = features.resolve(), original.resolve()
    config = read_json(features / 'config.json')
    readiness = read_json(features / 'summary.json')
    reference = read_json(original / 'manifest.json')
    if (config.get('format') != 'adresso_ctc_word_batch_v1'
            or not readiness.get('training_ready') or readiness.get('status') != 'complete'
            or readiness.get('completed') != 237):
        raise ValueError('The complete ADReSSo CTC word cohort must pass verification first')
    if (config['text_model'] != 'distilbert-base-uncased'
            or config['audio_model'] != 'facebook/wav2vec2-base-960h'
            or config['ctc_model'] != config['audio_model']
            or config['split_sources'] != reference['split_sources']):
        raise ValueError('Repeated search must preserve the encoders and saved development folds')
    if reference['trials'] != trial_grid():
        raise ValueError('Original architecture grid changed')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'slurm').mkdir()
    (root / 'trials').mkdir()
    for directory in ('scripts', 'utils'):
        destination = root / 'source' / directory
        destination.mkdir(parents=True)
        for path in sorted((REPO / directory).glob('*.py')):
            shutil.copy2(path, destination / path.name)
    shutil.copytree(original / 'baseline', root / 'baseline')
    fixed = {**reference['fixed_parameters'],
             'dataset_root': config['dataset_root'],
             'precomputed_features_dir': str(features / 'train/embeddings'),
             'precomputed_audio_suffix': config['audio_suffix'],
             'precomputed_text_suffix': config['text_suffix'],
             'speech_feature_extractor_output_vectors_dimension': 768,
             'text_feature_extractor_output_vectors_dimension': 768}
    manifest = {**reference, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'fixed_parameters': fixed, 'original_sweep_root': str(original),
        'baseline_is_matched': False,
        'baseline_comparison_note': 'Historical token-feature baseline is contextual; winner is selected only among new word-feature trials',
        'representation': 'English CTC word boundaries; mean acoustic/subword pooling; paired word masks; 200-token text chunks',
        'source_sha256': {str(p.relative_to(root)): sha256(p) for p in sorted((root / 'source').rglob('*.py'))},
        'input_sha256': {str(features / name): sha256(features / name)
                         for name in ('config.json', 'summary.json', 'alignment_verification.json')},
        'split_sources': config['split_sources']}
    write_json(root / 'manifest.json', manifest)
    summarize(root)
    print(f'Prepared repeated grid: {len(manifest["trials"])} trials with verified CTC word features', flush=True)
    return manifest


def large_encoder_settings(config, readiness, reference, features):
    """Retain the word-feature search budget while changing the two encoders."""
    if (config.get('format') != 'adresso_large_ctc_word_batch_v1'
            or not readiness.get('training_ready') or readiness.get('status') != 'complete'
            or readiness.get('completed') != 237 or readiness.get('failed') != 0
            or readiness.get('readiness', {}).get('loader_checked_recordings') != 237):
        raise ValueError('The complete large-encoder feature cohort must pass verification first')
    if (config['text_model'] != 'microsoft/deberta-v3-large'
            or config['audio_model'] != 'microsoft/wavlm-large'
            or config['ctc_model'] != 'facebook/wav2vec2-base-960h'
            or config['context_frames'] != 0 or config['chunk_size'] != 200
            or config['split_sources'] != reference['split_sources']):
        raise ValueError('Requested encoders, CTC word protocol and saved folds must match')
    if reference['trials'] != trial_grid():
        raise ValueError('Reference architecture grid changed')
    return {**reference['fixed_parameters'], 'dataset_root': config['dataset_root'],
            'precomputed_features_dir': str(features / 'train/embeddings'),
            'precomputed_audio_suffix': config['audio_suffix'],
            'precomputed_text_suffix': config['text_suffix'],
            'speech_feature_extractor_output_vectors_dimension': 1024,
            'text_feature_extractor_output_vectors_dimension': 1024}


def prepare_encoder_repeat(root, features, original):
    """Repeat the 27-trial word-feature search using DeBERTa/WavLM-large."""
    features, original = features.resolve(), original.resolve()
    config = read_json(features / 'config.json')
    readiness = read_json(features / 'summary.json')
    reference = read_json(original / 'manifest.json')
    fixed = large_encoder_settings(config, readiness, reference, features)
    # Pin every published feature/mask/metadata file, including the cohort audit.
    inputs = {str(features / name): sha256(features / name)
              for name in ('config.json', 'summary.json', 'alignment_verification.json')}
    for key, record in readiness['records'].items():
        if record['status'] != 'complete' or record['audio_shape'][1] != 1024 or record['text_shape'][1] != 1024:
            raise ValueError('A feature package is incomplete or has the wrong encoder width')
        for source in record['source_sha256'].values():
            if sha256(source['path']) != source['sha256']:
                raise ValueError('An original recording or transcript changed')
        for name, checksum in record['file_sha256'].items():
            path = features / key.split('/')[0] / 'embeddings' / name
            if sha256(path) != checksum:
                raise ValueError(f'Published feature package changed: {path}')
            inputs[str(path)] = checksum
    root.mkdir(parents=True, exist_ok=False)
    (root / 'slurm').mkdir()
    (root / 'trials').mkdir()
    for directory in ('scripts', 'utils'):
        destination = root / 'source' / directory
        destination.mkdir(parents=True)
        for path in sorted((REPO / directory).glob('*.py')):
            shutil.copy2(path, destination / path.name)
    shutil.copytree(original / 'baseline', root / 'baseline')
    manifest = {**reference, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'text_model': config['text_model'], 'audio_model': config['audio_model'],
        'text_revision': config['text_revision'], 'audio_revision': config['audio_revision'],
        'ctc_model': config['ctc_model'], 'ctc_revision': config['ctc_revision'],
        'fixed_parameters': fixed, 'original_sweep_root': str(original),
        'baseline_is_matched': False,
        'baseline_comparison_note': 'Historical baseline is contextual; select the winner only among these 27 large-encoder trials',
        'representation': 'English CTC word boundaries, 0 extra context; mean WavLM frames/DeBERTa subwords; paired masks; 200-token text chunks',
        'changed_settings': 'DeBERTa-v3-large and WavLM-large; both feature dimensions 768 -> 1024. All grid and training settings retained.',
        'source_sha256': {str(p.relative_to(root)): sha256(p) for p in sorted((root / 'source').rglob('*.py'))},
        'input_sha256': inputs, 'split_sources': config['split_sources']}
    write_json(root / 'manifest.json', manifest)
    summarize(root)
    print(f'Prepared {len(manifest["trials"])} large-encoder trials / 135 fold models: {root}', flush=True)
    return manifest


def run_trial(root, index, dry_run=False):
    manifest = read_json(root / 'manifest.json')
    if not 0 <= index < len(manifest['trials']):
        raise ValueError('Trial index is outside the prepared grid')
    for name, expected in manifest['source_sha256'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'Snapshotted source changed: {name}')
    for name, expected in manifest['input_sha256'].items():
        if sha256(name) != expected:
            raise ValueError(f'Feature cohort metadata changed: {name}')
    for source in manifest['split_sources']:
        if sha256(source['path']) != source['sha256']:
            raise ValueError(f'Saved split changed: {source["path"]}')
    trial = manifest['trials'][index]
    command = build_command(root, manifest, trial)
    environment = {k: v for k, v in os.environ.items()
                   if k not in {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
                                'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'}
                   and not k.startswith('TORCHELASTIC_')}
    environment.update(PYTHONPATH=str(root / 'source'), PYTHONDONTWRITEBYTECODE='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if dry_run:
        subprocess.run(command + ['--cv_dry_run'], cwd=root / 'source', env=environment, check=True)
        return
    directory = root / 'trials' / trial['id']
    directory.mkdir(exist_ok=False)
    write_json(directory / 'trial.json', {**trial, 'command': command,
               'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    print(f'Starting {trial["id"]}; training log: {directory / "training.log"}', flush=True)
    try:
        with (directory / 'training.log').open('w') as log:
            subprocess.run(command, cwd=root / 'source', env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    except Exception as error:
        write_json(directory / 'failed.json', {'error': f'{type(error).__name__}: {error}'})
        raise
    # Validate all five selected checkpoints and decision journals immediately.
    row = completed_trial(root, trial)
    write_json(directory / 'verified_result.json', row)
    print(f'Finished {trial["id"]}: mean macro-F1 {row["mean_macro_f1"]:.4f}', flush=True)


def audit_checkpoint(fold):
    checkpoint = Path(fold['checkpoint'])
    if not checkpoint.is_file():
        raise ValueError(f'Missing checkpoint: {checkpoint}')
    decisions = [json.loads(line) for line in
                 (checkpoint.parent / 'checkpoint_decisions.jsonl').read_text().splitlines()]
    last_saved = None
    for decision in decisions:
        score = decision['validation_macro_f1']
        previous = decision['previous_best_validation_macro_f1']
        expected = decision['eligible'] and score is not None and (previous is None or score > previous)
        if decision['saved'] != expected:
            raise ValueError('Checkpoint decision violates strict improvement policy')
        if previous != (last_saved['validation_macro_f1'] if last_saved else None):
            raise ValueError('Checkpoint decision history has an inconsistent best score')
        step = decision['completed_steps']
        if step % 100 != 0 and step != fold['epochs'] * fold['training_recordings']:
            raise ValueError('Checkpoint evaluated outside the 100-update or terminal schedule')
        if decision['saved']:
            last_saved = decision
    if (last_saved is None or last_saved['completed_steps'] != fold['selected_step']
            or abs(last_saved['validation_macro_f1'] - fold['metrics']['macro_f1']) > 1e-12):
        raise ValueError('Selected checkpoint disagrees with its decision journal')


def completed_trial(root, trial):
    directory = root / 'trials' / trial['id'] / 'results'
    summary = read_json(directory / 'summary.json')
    config = read_json(directory / 'config.json')
    manifest = read_json(root / 'manifest.json')
    for key, expected in {**manifest['fixed_parameters'],
                          **{k: v for k, v in trial.items() if k not in ('id', 'index')}}.items():
        if config[key] != expected:
            raise ValueError(f'Trial configuration changed: {key}')
    if read_json(directory / 'splits.json') != read_json(root / 'baseline/splits.json'):
        raise ValueError('Trial did not use the baseline development folds')
    folds = [read_json(directory / f'fold_{index}.json') for index in range(5)]
    for fold in folds:
        audit_checkpoint(fold)
    if summary['num_folds'] != 5 or summary['recordings'] != 166:
        raise ValueError('Incomplete ADReSSo CV result')
    return {**trial, 'mean_macro_f1': summary['mean_macro_f1'],
            'std_macro_f1': summary['std_macro_f1'],
            'pooled_macro_f1': summary['out_of_fold']['macro_f1'],
            'worst_fold_macro_f1': min(summary['fold_macro_f1']),
            'fold_macro_f1': summary['fold_macro_f1'],
            'selected_epochs': [fold['selected_epoch'] for fold in folds],
            'results_directory': str(directory), 'checkpoint_audit': 'passed'}


def summarize(root, require_complete=False):
    manifest = read_json(root / 'manifest.json')
    completed, pending, failed = [], [], []
    for trial in manifest['trials']:
        directory = root / 'trials' / trial['id']
        if not (directory / 'results/summary.json').is_file():
            pending.append(trial['id'])
            continue
        try:
            completed.append(completed_trial(root, trial))
        except (ValueError, KeyError, OSError) as error:
            failed.append({'id': trial['id'], 'error': str(error)})
    completed.sort(key=lambda row: (-row['mean_macro_f1'], row['std_macro_f1'], row['id']))
    baseline = read_json(root / 'baseline/summary.json')
    report = {
        'status': 'complete' if len(completed) == len(manifest['trials']) else 'incomplete',
        'expected_trials': len(manifest['trials']), 'completed_trials': len(completed),
        'pending_or_failed_trials': pending, 'invalid_results': failed,
        'ranking_metric': 'mean_macro_f1', 'ranking': completed,
        'best_completed_trial': completed[0] if completed else None,
        'baseline_mean_macro_f1': baseline['mean_macro_f1'],
        'baseline_std_macro_f1': baseline['std_macro_f1'],
        'baseline_results_directory': manifest['baseline_results_directory'],
        'interpretation': 'Development CV used for architecture and checkpoint selection; official test set unused',
    }
    if completed:
        report['best_minus_baseline_macro_f1'] = completed[0]['mean_macro_f1'] - baseline['mean_macro_f1']
        report['baseline_is_matched'] = manifest.get('baseline_is_matched', True)
        report['best_configuration_including_baseline'] = (completed[0]['id']
            if not report['baseline_is_matched'] or completed[0]['mean_macro_f1'] > baseline['mean_macro_f1']
            else 'original_baseline')
        if report['status'] == 'complete':
            architecture_keys = [key for key in completed[0]
                                 if key.startswith(('seq_to_', 'classifier_'))]
            winning = (completed[0] if report['best_configuration_including_baseline'] != 'original_baseline'
                       else read_json(root / 'baseline/config.json'))
            write_json(root / 'best_parameters.json', {
                'winner': report['best_configuration_including_baseline'],
                'parameters': {**manifest['fixed_parameters'],
                               **{key: winning[key] for key in architecture_keys}},
                'mean_macro_f1': (completed[0]['mean_macro_f1'] if not report['baseline_is_matched']
                                 else max(completed[0]['mean_macro_f1'], baseline['mean_macro_f1'])),
                'selection_metric': 'mean validation macro-F1 across five development folds',
                'official_test_used': False,
            })
    write_json(root / 'leaderboard.json', report)
    columns = ['id', 'seq_to_seq_heads_number', 'seq_to_seq_input_dropout',
               'seq_to_one_input_dropout', 'classifier_layer_drop_out', 'classifier_hidden_layers',
               'mean_macro_f1', 'std_macro_f1', 'pooled_macro_f1', 'worst_fold_macro_f1',
               'checkpoint_audit', 'results_directory']
    temporary = root / 'leaderboard.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(completed)
    temporary.replace(root / 'leaderboard.csv')
    print(f'Validated completed trials: {len(completed)}/{len(manifest["trials"])}', flush=True)
    if completed:
        print(f'Best completed trial: {completed[0]["id"]}; macro-F1 {completed[0]["mean_macro_f1"]:.4f}', flush=True)
    if require_complete and report['status'] != 'complete':
        raise RuntimeError('Sweep incomplete; inspect leaderboard.json and Slurm task logs')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--features-root', type=Path,
        default=REPO / 'outputs/adresso_full_whisper_distil_pauses')
    prepare_parser.add_argument('--baseline-results', type=Path,
        default=REPO / 'outputs/adresso_baseline_cv/run_2698224/results')
    repeat_parser = commands.add_parser('prepare-word-repeat')
    repeat_parser.add_argument('--sweep-root', type=Path, required=True)
    repeat_parser.add_argument('--features-root', type=Path, required=True)
    repeat_parser.add_argument('--original-root', type=Path,
        default=REPO / 'outputs/adresso_architecture_sweep_20261029')
    encoder_parser = commands.add_parser('prepare-encoder-repeat')
    encoder_parser.add_argument('--sweep-root', type=Path, required=True)
    encoder_parser.add_argument('--features-root', type=Path, required=True)
    encoder_parser.add_argument('--original-root', type=Path,
        default=REPO / 'outputs/adresso_architecture_sweep_words_20261029')
    for name, child in [('prepare', prepare_parser), ('run-trial', commands.add_parser('run-trial')),
                        ('summarize', commands.add_parser('summarize'))]:
        child.add_argument('--sweep-root', type=Path, required=True)
        if name == 'run-trial':
            child.add_argument('--trial-index', type=int, required=True)
            child.add_argument('--dry-run', action='store_true')
        elif name == 'summarize':
            child.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    root = args.sweep_root.resolve()
    if args.command == 'prepare':
        prepare(root, args.features_root, args.baseline_results)
    elif args.command == 'prepare-word-repeat':
        prepare_word_repeat(root, args.features_root, args.original_root)
    elif args.command == 'prepare-encoder-repeat':
        prepare_encoder_repeat(root, args.features_root, args.original_root)
    elif args.command == 'run-trial':
        run_trial(root, args.trial_index, args.dry_run)
    else:
        summarize(root, args.require_complete)


if __name__ == '__main__':
    main()
