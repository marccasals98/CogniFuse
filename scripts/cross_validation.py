"""Sequential patient-level cross-validation with a fixed epoch budget."""

import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
from sklearn.metrics import f1_score

if __package__:
    from .data import PrecomputedADDataset
else:
    from data import PrecomputedADDataset


def plan_folds(params):
    if not getattr(params, 'precomputed_features_dir', None):
        raise ValueError('The CV runner currently requires --precomputed_features_dir')
    if Path(params.train_labels_path).resolve() != Path(params.validation_labels_path).resolve():
        raise ValueError('Patient cross-validation requires the same labels CSV for training and validation')
    if params.num_folds < 2 or params.max_epochs < 1:
        raise ValueError('CV requires at least two folds and one epoch')
    if params.load_checkpoint:
        raise ValueError('Cross-validation must start fresh; omit --load_checkpoint')
    plans, seen_validation_patients, seen_recordings = [], set(), set()
    cohort = None
    for fold in range(params.num_folds):
        train = PrecomputedADDataset(params, split='train', fold=fold)
        validation = PrecomputedADDataset(params, split='val', fold=fold)
        if not len(train) or not len(validation):
            raise ValueError(f'Fold {fold} has an empty training or validation set')
        if params.number_classes != len(train.label_names):
            raise ValueError('number_classes must match the selected dataset classes')
        if not all(validation.class_counts.values()) or not all(train.class_counts.values()):
            raise ValueError('Each class needs enough distinct patients to appear in every fold; reduce num_folds')
        train_patients, val_patients = set(train.df.patient_id), set(validation.df.patient_id)
        train_uids = {r['uid'] for r in train.segments}
        val_uids = {r['uid'] for r in validation.segments}
        if train_patients & val_patients or train_uids & val_uids:
            raise ValueError(f'Patient/recording leakage in fold {fold}')
        if val_patients & seen_validation_patients or val_uids & seen_recordings:
            raise ValueError('A patient or recording is held out in more than one fold')
        seen_validation_patients.update(val_patients)
        seen_recordings.update(val_uids)
        current_cohort = train_uids | val_uids
        if cohort is not None and current_cohort != cohort:
            raise ValueError('The selected cohort changed between folds')
        cohort = current_cohort
        plans.append({
            'fold': fold, 'training_recordings': len(train), 'validation_recordings': len(validation),
            'training_patients': sorted(train_patients), 'validation_patients': sorted(val_patients),
            'training_uids': sorted(train_uids), 'validation_uids': sorted(val_uids),
        })
    if seen_recordings != cohort:
        raise ValueError('Some recordings never appear in validation')
    return plans


def scores(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    return {
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'per_class_f1': dict(zip(classes, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist())),
    }


def write_fold_results(trainer):
    train = trainer.training_generator.dataset
    validation = trainer.evaluating_generator.dataset
    predictions, labels = trainer.validation_predictions, trainer.validation_labels
    if len(predictions) != len(validation.segments) or labels != [r['label'] for r in validation.segments]:
        raise ValueError('Validation predictions do not match recording order')
    records = [
        {'uid': r['uid'], 'patient_id': r['patient_id'], 'label': label, 'prediction': prediction}
        for r, label, prediction in zip(validation.segments, labels, predictions)
    ]
    result = {
        'fold': trainer.params.fold, 'num_folds': trainer.params.num_folds,
        'random_seed': trainer.params.random_seed, 'epochs': trainer.params.max_epochs,
        'protocol': 'fixed_epochs_final_model', 'classes': validation.label_names,
        'training_recordings': len(train), 'validation_recordings': len(validation),
        'training_patients': sorted(set(train.df.patient_id)),
        'validation_patients': sorted(set(validation.df.patient_id)),
        'checkpoint': str(Path(trainer.params.model_output_folder) / trainer.params.model_name / (trainer.params.model_name + '.chkpt')),
        'metrics': scores(labels, predictions, validation.label_names),
        'predictions': records,
    }
    path = Path(trainer.params.fold_results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)


def summarize_results(results):
    if not results or len(results) != results[0]['num_folds']:
        raise ValueError('Cannot summarize incomplete cross-validation')
    count = len(results)
    if {result['fold'] for result in results} != set(range(count)):
        raise ValueError('Missing or duplicate fold results')
    classes = results[0]['classes']
    validation_patients, uids, records = set(), set(), []
    for result in results:
        for key in ('num_folds', 'random_seed', 'epochs', 'protocol', 'classes'):
            if result[key] != results[0][key]:
                raise ValueError(f'Inconsistent fold setting: {key}')
        val_patients = set(result['validation_patients'])
        if val_patients & set(result['training_patients']) or val_patients & validation_patients:
            raise ValueError('Patient leakage or repeated validation patients')
        validation_patients.update(val_patients)
        if len(result['predictions']) != result['validation_recordings']:
            raise ValueError('Missing validation predictions')
        for record in result['predictions']:
            if record['uid'] in uids or record['patient_id'] not in val_patients:
                raise ValueError('Repeated or incorrectly assigned validation recording')
            uids.add(record['uid'])
            records.append(record)
    values = [result['metrics']['macro_f1'] for result in results]
    return {
        'protocol': results[0]['protocol'], 'num_folds': count,
        'random_seed': results[0]['random_seed'], 'epochs_per_fold': results[0]['epochs'],
        'classes': classes, 'recordings': len(records), 'patients': len(validation_patients),
        'fold_macro_f1': values,
        'mean_macro_f1': float(np.mean(values)),
        'std_macro_f1': float(np.std(values, ddof=1)),
        'std_definition': 'sample standard deviation across folds (ddof=1)',
        'out_of_fold': scores([r['label'] for r in records], [r['prediction'] for r in records], classes),
    }


def run_cross_validation(params, cli_args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('The sequential CV runner requires one process/GPU; use --nproc_per_node=1')
    if params.fold_results_path:
        raise ValueError('fold_results_path is managed by the CV runner')
    plans = plan_folds(params)
    print('Patient-level cross-validation (fold indices start at zero):', flush=True)
    for plan in plans:
        print(f"  Fold {plan['fold']}: {plan['training_recordings']} train / {plan['validation_recordings']} validation recordings; "
              f"{len(plan['training_patients'])} train / {len(plan['validation_patients'])} validation patients", flush=True)
    print(f'Each model starts fresh and trains for {params.max_epochs} epochs. '
          'Validation-driven checkpoint selection, early stopping and learning-rate updates are disabled.', flush=True)
    if params.cv_dry_run:
        return
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    output = Path(params.cross_validation_output_dir or Path(params.log_file_folder) / 'cross_validation' / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'splits.json').write_text(json.dumps(plans, indent=2) + '\n')
    (output / 'config.json').write_text(json.dumps({**vars(params), 'protocol': 'fixed_epochs_final_model',
        'eval_and_save_best_model_every': 0, 'early_stopping': 0, 'update_optimizer_every': 0}, indent=2) + '\n')
    print(f'CV results directory: {output}', flush=True)
    # Each fold gets a new interpreter and CUDA context. A torchrun parent has
    # one rank; its rendezvous variables must not initialize DDP in the children.
    distributed_keys = {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                        'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'}
    environment = {k: v for k, v in os.environ.items()
                   if k not in distributed_keys and not k.startswith('TORCHELASTIC_')}
    results = []
    for plan in plans:
        result_path = output / f"fold_{plan['fold']}.json"
        command = [sys.executable, str(Path(__file__).with_name('train.py')), *cli_args,
                   '--no-cross_validate', '--no-cv_dry_run', '--fold', str(plan['fold']),
                   '--fold_results_path', str(result_path), '--no-load_checkpoint',
                   '--eval_and_save_best_model_every', '0', '--early_stopping', '0', '--update_optimizer_every', '0']
        subprocess.run(command, env=environment, check=True)
        result = json.loads(result_path.read_text())
        if (result['fold'] != plan['fold']
                or result['training_patients'] != plan['training_patients']
                or result['validation_patients'] != plan['validation_patients']
                or sorted(r['uid'] for r in result['predictions']) != plan['validation_uids']):
            raise ValueError('Completed fold disagrees with the planned patient split')
        results.append(result)
        print(f"Fold {plan['fold']} validation macro-F1: {result['metrics']['macro_f1']:.4f}", flush=True)
    summary = summarize_results(results)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'fold_metrics.csv').open('w', newline='') as handle:
        columns = ['fold', 'training_recordings', 'validation_recordings', 'macro_f1']
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({**{key: result[key] for key in columns[:-1]}, 'macro_f1': result['metrics']['macro_f1']})
    print(f"CV macro-F1: {summary['mean_macro_f1']:.4f} +/- {summary['std_macro_f1']:.4f} (mean +/- sample SD)", flush=True)
    print(f"Pooled out-of-fold macro-F1: {summary['out_of_fold']['macro_f1']:.4f}", flush=True)
    print(f'Full summary: {output / "summary.json"}', flush=True)
