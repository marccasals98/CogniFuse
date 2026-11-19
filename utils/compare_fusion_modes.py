"""Freeze, run, and audit matched development CV for three fusion methods."""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import torch

from scripts.cross_validation import plan_folds, summarize_results
from scripts.reliability import fit_quality_normalization
from scripts.data import make_precomputed_dataset
from scripts.evaluate import load_model
from scripts.word_fusion import DECOMPOSED_METHODS
from utils.package_fusion_frames import digest, read_fusion_package
from utils.prepare_fusion_cohort import read_json, write_json
from utils.tune_adresso import audit_checkpoint

REPO = Path(__file__).resolve().parents[1]


def options_to_cli(options):
    result = []
    for name, value in options.items():
        if isinstance(value, bool):
            result.append('--' + ('' if value else 'no-') + name)
        else:
            result.extend(['--' + name, str(value)])
    return result


def parsed_parameters(options):
    # The historical model imports siblings from scripts.
    scripts = str(REPO / 'scripts')
    if scripts not in sys.path:
        sys.path.append(scripts)
    from train import ArgsParser
    parser = ArgsParser()
    parser.main(options_to_cli(options))
    return parser.arguments


def initialization_audit(options, schema, normalization):
    results = {}
    for method in DECOMPOSED_METHODS:
        params = parsed_parameters({**options, 'fusion_method': method})
        params.frame_package_schema = schema
        params.frame_package_schema_sha256 = normalization['schema_sha256']
        params.frame_quality_normalization = normalization
        from model import Classifier
        torch.manual_seed(params.torch_seed)
        model = Classifier(params, 'cpu')
        common = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            if not name.startswith('word_fusion.gate.'):
                common.update(name.encode())
                common.update(str(tuple(value.shape)).encode())
                common.update(value.cpu().numpy().tobytes())
        results[method] = {'common_state_sha256': common.hexdigest(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'gate_parameters': sum(p.numel() for n, p in model.named_parameters()
                                   if n.startswith('word_fusion.gate.'))}
    if len({r['common_state_sha256'] for r in results.values()}) != 1:
        raise ValueError('Common layers do not have matched initialization across fusion methods')
    return results


def prepare_experiment(cohort_root, experiment_root, epochs=60, num_folds=5, torch_seed=3407):
    cohort, root = Path(cohort_root).resolve(), Path(experiment_root).resolve()
    config, ready = read_json(cohort / 'config.json'), read_json(cohort / 'summary.json')
    if (ready['status'] != 'complete' or not ready['training_ready']
            or ready['completed'] != len(config['development_uids']) or config['num_folds'] != num_folds):
        raise ValueError('Complete, verified development quality packages are required')
    if root.exists():
        raise FileExistsError('Comparison already exists; use a new experiment root')
    options = {
        'dataset': 'adresso', 'dataset_root': config['dataset_root'], 'number_classes': 2,
        'precomputed_features_dir': str(cohort / 'quality/train'), 'precomputed_input_type': 'frames',
        'num_folds': num_folds, 'random_seed': 1234, 'torch_seed': torch_seed,
        'speech_feature_extractor_output_vectors_dimension': 1024,
        'text_feature_extractor_output_vectors_dimension': 1024,
        'speech_adapter': 'NoneAdapter', 'text_adapter': 'NoneAdapter',
        'acoustic_word_pooling': 'attention', 'acoustic_word_attention_hidden_dim': 64,
        'decomposition_dim': 256, 'decomposition_heads': 4, 'decomposition_cross_radius': 1,
        'fusion_gate_hidden_dim': 64, 'shared_agreement_weight': .1,
        'private_shared_separation_weight': .01, 'decomposition_reconstruction_weight': .1,
        'shared_variance_weight': .1, 'seq_to_one_method': 'AttentionPooling',
        'seq_to_seq_input_dropout': .3, 'seq_to_one_input_dropout': .3,
        'classifier_hidden_layers': 1, 'classifier_hidden_layers_width': 512,
        'classifier_layer_drop_out': .3, 'max_epochs': epochs,
        'training_batch_size': 1, 'evaluation_batch_size': 1, 'num_workers': 0,
        'training_augmentation_prob': 0., 'evaluation_augmentation_prob': 0.,
        'optimizer': 'adamw', 'learning_rate': .0001, 'weight_decay': .01,
        'loss': 'CrossEntropy', 'weighted_loss': True, 'skip_connections': False,
        'eval_and_save_best_model_every': 100, 'print_training_info_every': 100,
        'early_stopping': 0, 'update_optimizer_every': 0,
        'load_checkpoint': False, 'use_weights_and_biases': False,
    }
    params = parsed_parameters(options)
    plans = plan_folds(params)
    receipts = {}
    schema = None
    for uid in config['development_uids']:
        path = cohort / 'quality/train' / uid
        metadata, _ = read_fusion_package(path)
        checksum = digest(path / 'metadata.json')
        if checksum != ready['records'][uid]['metadata_sha256']:
            raise ValueError('Completed cohort package changed')
        if metadata['schema_sha256'] != ready['schema_sha256']:
            raise ValueError('Cohort schema changed')
        schema = metadata['schema']
        receipts[uid] = checksum
    if schema['audio_width'] != 1024 or schema['text_width'] != 1024:
        # Synthetic integration fixtures may use other widths; infer both explicitly.
        options.update(speech_feature_extractor_output_vectors_dimension=schema['audio_width'],
                       text_feature_extractor_output_vectors_dimension=schema['text_width'])
        if schema['audio_width'] != schema['text_width']:
            options.update(speech_adapter='LinearAdapter', speech_adapter_output_vectors_dimension=256,
                           text_adapter='LinearAdapter', text_adapter_output_vectors_dimension=256)
    params = parsed_parameters(options)
    train = make_precomputed_dataset(params, split='train', fold=0)
    normalization = fit_quality_normalization(train)
    initialization = initialization_audit(options, schema, normalization)
    root.mkdir(parents=True)
    (root / 'slurm').mkdir()
    for directory in ('scripts', 'utils'):
        destination = root / 'source' / directory
        destination.mkdir(parents=True)
        for path in sorted((REPO / directory).glob('*.py')):
            shutil.copy2(path, destination / path.name)
    manifest = {'format': 'matched_fusion_comparison_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'cohort_root': str(cohort), 'official_test_used': False, 'common_options': options,
        'variants': list(DECOMPOSED_METHODS), 'folds': plans, 'package_metadata_sha256': receipts,
        'schema_sha256': ready['schema_sha256'], 'initialization': initialization,
        'input_sha256': {str(cohort / name): digest(cohort / name) for name in ('config.json', 'summary.json')},
        'split_sha256': config['split_sha256'],
        'source_sha256': {str(p.relative_to(root)): digest(p) for p in sorted((root / 'source').rglob('*.py'))},
        'interpretation': 'Development CV with validation-based checkpoint selection; one seed; '
                          'not an unbiased test estimate or evidence of degradation robustness'}
    write_json(root / 'manifest.json', manifest)
    summarize(root)
    return manifest


def verify_inputs(root, manifest, tensors=True):
    for name, checksum in manifest['source_sha256'].items():
        if digest(root / name) != checksum:
            raise ValueError('Snapshotted source changed')
    for name, checksum in {**manifest['input_sha256'], **manifest['split_sha256']}.items():
        if digest(name) != checksum:
            raise ValueError('Pinned cohort or split changed')
    for uid, checksum in manifest['package_metadata_sha256'].items():
        path = Path(manifest['cohort_root']) / 'quality/train' / uid
        if digest(path / 'metadata.json') != checksum:
            raise ValueError('Pinned quality package changed')
        if tensors:
            read_fusion_package(path)


def variant_command(root, manifest, method):
    if method not in manifest['variants']:
        raise ValueError('Unknown comparison variant')
    directory = root / method
    options = {**manifest['common_options'], 'fusion_method': method,
        'model_output_folder': str(directory / 'models'), 'log_file_folder': str(directory / 'logs'),
        'cross_validation_output_dir': str(directory / 'results'), 'cross_validate': True}
    return [sys.executable, str(root / 'source/scripts/train.py'), *options_to_cli(options)]


def run_variant(experiment_root, method, dry_run=False):
    root = Path(experiment_root).resolve()
    manifest = read_json(root / 'manifest.json')
    verify_inputs(root, manifest)
    command = variant_command(root, manifest, method)
    environment = {key: value for key, value in os.environ.items()
        if key not in {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                       'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'}
        and not key.startswith('TORCHELASTIC_')}
    environment.update(PYTHONPATH=str(root / 'source'), PYTHONDONTWRITEBYTECODE='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if dry_run:
        subprocess.run(command + ['--cv_dry_run'], cwd=root / 'source', env=environment, check=True)
        return
    directory = root / method
    directory.mkdir(exist_ok=False)
    write_json(directory / 'run.json', {'command': command, 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    try:
        with (directory / 'training.log').open('w') as stream:
            subprocess.run(command, cwd=root / 'source', env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = audit_variant(root, manifest, method)
        write_json(directory / 'verified_result.json', result)
    except Exception as error:
        write_json(directory / 'failed.json', {'error': f'{type(error).__name__}: {error}'})
        raise
    summarize(root)


def audit_variant(root, manifest, method):
    results = root / method / 'results'
    config = read_json(results / 'config.json')
    for name, expected in {**manifest['common_options'], 'fusion_method': method}.items():
        if config[name] != expected:
            raise ValueError(f'Variant configuration changed: {name}')
    if read_json(results / 'splits.json') != manifest['folds']:
        raise ValueError('Variant development folds changed')
    folds = []
    for plan in manifest['folds']:
        fold = read_json(results / f'fold_{plan["fold"]}.json')
        predictions = fold['predictions']
        if (fold['fold'] != plan['fold'] or fold['training_patients'] != plan['training_patients']
                or fold['validation_patients'] != plan['validation_patients']
                or sorted(r['uid'] for r in predictions) != plan['validation_uids']):
            raise ValueError('Fold results differ from the frozen split plan')
        audit_checkpoint(fold)
        checkpoint = torch.load(fold['checkpoint'], map_location='cpu', weights_only=False)
        settings = checkpoint['settings']
        for name, expected in {**manifest['common_options'], 'fusion_method': method}.items():
            if getattr(settings, name) != expected:
                raise ValueError(f'Checkpoint configuration differs: {name}')
        normalization = settings.frame_quality_normalization
        if (set(normalization['fit_uids']) != set(plan['training_uids'])
                or normalization['fit_metadata_sha256'] != {
                    uid: manifest['package_metadata_sha256'][uid] for uid in plan['training_uids']}):
            raise ValueError('Checkpoint normalization has different training records or receipts')
        load_model(settings, checkpoint['model'], 'cpu')
        folds.append(fold)
    computed = summarize_results(folds)
    if computed != read_json(results / 'summary.json'):
        raise ValueError('CV summary differs from verified fold predictions')
    return {'method': method, 'summary': computed, 'checkpoint_audit': 'passed',
            'predictions': sorted([r for fold in folds for r in fold['predictions']], key=lambda r: r['uid']),
            'selected_epochs': [fold['selected_epoch'] for fold in folds]}


def summarize(experiment_root, require_complete=False):
    root = Path(experiment_root).resolve()
    manifest = read_json(root / 'manifest.json')
    verify_inputs(root, manifest, tensors=False)
    completed, pending = {}, []
    for method in manifest['variants']:
        if not (root / method / 'results/summary.json').exists():
            pending.append(method)
        else:
            completed[method] = audit_variant(root, manifest, method)
    report = {'format': 'matched_fusion_results_v1', 'complete': not pending,
              'official_test_used': False, 'pending': pending,
              'variants': {name: {k: v for k, v in result.items() if k != 'predictions'}
                           for name, result in completed.items()},
              'interpretation': manifest['interpretation'], 'paired_fold_differences': {}}
    if not pending:
        for first, second in itertools.combinations(manifest['variants'], 2):
            a, b = completed[first], completed[second]
            if [(r['uid'], r['label']) for r in a['predictions']] != [(r['uid'], r['label']) for r in b['predictions']]:
                raise ValueError('Paired variants have different labels or recording order')
            delta = np.array(b['summary']['fold_macro_f1']) - np.array(a['summary']['fold_macro_f1'])
            report['paired_fold_differences'][second + '_minus_' + first] = {
                'fold_deltas': delta.tolist(), 'mean_delta': float(delta.mean()),
                'sample_sd': float(delta.std(ddof=1))}
        rows = ['method,mean_macro_f1,sample_sd,pooled_macro_f1,gate_parameters']
        for method, result in completed.items():
            summary = result['summary']
            rows.append(f'{method},{summary["mean_macro_f1"]},{summary["std_macro_f1"]},'
                        f'{summary["out_of_fold"]["macro_f1"]},{manifest["initialization"][method]["gate_parameters"]}')
        (root / 'comparison.csv').write_text('\n'.join(rows) + '\n')
    # Several array tasks may finish together; publish through unique temp files.
    path = root / f'comparison.{os.getpid()}.tmp'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    path.replace(root / 'comparison.json')
    if require_complete and pending:
        raise RuntimeError('Comparison incomplete: ' + ', '.join(pending))
    return report


def submit(experiment_root):
    root = Path(experiment_root).resolve()
    manifest = read_json(root / 'manifest.json')
    verify_inputs(root, manifest)
    if (root / 'submission.json').exists() or any((root / method).exists() for method in manifest['variants']):
        raise FileExistsError('Comparison already submitted or started')
    interpreter = shlex.quote(sys.executable)
    source = shlex.quote(str(root / 'source'))
    target = shlex.quote(str(root))
    runner = root / 'slurm/run.sh'
    runner.write_text('#!/bin/bash\nset -euo pipefail\n'
        'export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n'
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8\n'
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"\n'
        f'export PYTHONPATH={source}\ncd {source}\n'
        f'methods=({" ".join(manifest["variants"])})\n'
        f'{interpreter} -m utils.compare_fusion_modes run --experiment-root {target} '
        '--variant "${methods[$SLURM_ARRAY_TASK_ID]}"\n')
    summary_script = root / 'slurm/summarize.sh'
    summary_script.write_text('#!/bin/bash\nset -euo pipefail\n'
        f'export PYTHONPATH={source}\ncd {source}\n'
        f'{interpreter} -m utils.compare_fusion_modes summarize --experiment-root {target} --require-complete\n')
    common = ['sbatch', '--parsable', '-A', 'veu', '-p', 'veu', '--ntasks=1',
              '--chdir=' + str(root / 'source'), '--output=' + str(root / 'slurm/%x-%j.log')]
    result = subprocess.run([*common, '--job-name=fusion_cv', '--array=0-2%2', '--gres=gpu:1',
        '--cpus-per-task=4', '--mem=32G', '--time=24:00:00', '--exclude=veuc01', str(runner)],
        check=True, text=True, capture_output=True)
    job = result.stdout.strip().split(';')[0]
    submission = {'array_job_id': job, 'variants': manifest['variants'], 'max_concurrent_gpu_jobs': 2}
    write_json(root / 'submission.json', submission)
    followup = subprocess.run([*common, '--job-name=fusion_summary', '--dependency=afterok:' + job,
        '--cpus-per-task=1', '--mem=4G', '--time=00:30:00', str(summary_script)],
        check=True, text=True, capture_output=True)
    submission['summary_job_id'] = followup.stdout.strip().split(';')[0]
    write_json(root / 'submission.json', submission)
    return submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--cohort-root', type=Path, required=True)
    prepare.add_argument('--experiment-root', type=Path, required=True)
    prepare.add_argument('--epochs', type=int, default=60)
    prepare.add_argument('--num-folds', type=int, default=5)
    prepare.add_argument('--torch-seed', type=int, default=3407)
    run = sub.add_parser('run')
    run.add_argument('--experiment-root', type=Path, required=True)
    run.add_argument('--variant', choices=DECOMPOSED_METHODS, required=True)
    run.add_argument('--dry-run', action='store_true')
    summary = sub.add_parser('summarize')
    summary.add_argument('--experiment-root', type=Path, required=True)
    summary.add_argument('--require-complete', action='store_true')
    launch = sub.add_parser('submit')
    launch.add_argument('--experiment-root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == 'prepare':
        result = prepare_experiment(args.cohort_root, args.experiment_root, args.epochs, args.num_folds, args.torch_seed)
        print(f'Prepared {len(result["variants"])} variants / {len(result["variants"]) * len(result["folds"])} fold models')
    elif args.action == 'run':
        run_variant(args.experiment_root, args.variant, args.dry_run)
    elif args.action == 'summarize':
        print(json.dumps(summarize(args.experiment_root, args.require_complete), indent=2))
    else:
        print(json.dumps(submit(args.experiment_root), indent=2))


if __name__ == '__main__':
    main()
