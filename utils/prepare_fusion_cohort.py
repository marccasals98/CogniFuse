"""Resumable development-only frame packaging and raw quality extraction."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.adresso import ADReSSoIndex
from scripts.data import make_precomputed_dataset
from scripts.reliability import fit_quality_normalization
from scripts.settings import TRAIN_DEFAULT_SETTINGS
from utils.extract_fusion_quality import enrich_package
from utils.package_fusion_frames import digest, package_fusion_frames, read_fusion_package


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def cohort_plan(feature_root, dataset_root, num_folds):
    source = Path(feature_root).resolve()
    index = ADReSSoIndex(dataset_root)
    folds = index.development_folds(num_folds)
    summary = read_json(source / 'summary.json')
    config = read_json(source / 'config.json')
    if Path(config['dataset_root']).resolve() != index.root:
        raise ValueError('Feature sources and requested cohort have different dataset roots')
    uids = sorted(record['uid'] for record in index.training)
    if any(summary['records'].get('train/' + uid, {}).get('status') != 'complete' for uid in uids):
        raise ValueError('All development recordings need complete cached source exports')
    splits = {str(index.root / 'train/splits' / f'{split}_uids{fold}.npy'):
              digest(index.root / 'train/splits' / f'{split}_uids{fold}.npy')
              for fold in range(num_folds) for split in ('train', 'val')}
    if {item['path']: item['sha256'] for item in config['split_sources']} != splits:
        raise ValueError('Saved cohort folds differ from source extraction provenance')
    return {'format': 'fusion_cohort_v1', 'feature_root': str(source), 'dataset_root': str(index.root),
            'num_folds': num_folds, 'development_uids': uids, 'folds': folds,
            'official_test_used': False, 'split_sha256': splits,
            'source_sha256': {str(source / name): digest(source / name)
                              for name in ('config.json', 'summary.json')}}


def package_hashes(path):
    return {p.name: digest(p) for p in Path(path).iterdir() if p.is_file()}


def verify_lineage(metadata, plan, uid):
    provenance = metadata['provenance']
    if (metadata['uid'] != uid or metadata['partition'] != 'train'
            or provenance['feature_root'] != plan['feature_root']
            or provenance['recording_key'] != 'train/' + uid
            or any(provenance['verified_sha256'].get(path) != checksum
                   for path, checksum in plan['source_sha256'].items())):
        raise ValueError('Reused package differs from pinned development sources')


def prepare_cohort(feature_root, dataset_root, output_root, num_folds=5):
    plan = cohort_plan(feature_root, dataset_root, num_folds)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = root / 'config.json'
        if config_path.exists() and read_json(config_path) != plan:
            raise ValueError('Cohort settings or source identities changed; choose a new output root')
        if not config_path.exists():
            write_json(config_path, plan)
        state_path = root / 'summary.json'
        state = read_json(state_path) if state_path.exists() else {'records': {}}
        state.update(status='running', training_ready=False, expected_development=len(plan['development_uids']),
                     official_test_used=False)
        write_json(state_path, state)
        schema = None
        for ordinal, uid in enumerate(plan['development_uids'], 1):
            pending, ready = root / 'frames/train' / uid, root / 'quality/train' / uid
            try:
                if not pending.exists():
                    package_fusion_frames(feature_root, uid, pending)
                pending_meta, _ = read_fusion_package(pending)
                verify_lineage(pending_meta, plan, uid)
                if not ready.exists():
                    enrich_package(pending, ready)
                metadata, _ = read_fusion_package(ready)
                verify_lineage(metadata, plan, uid)
                if (metadata['schema']['quality']['status'] != 'ready'
                        or metadata['quality_extraction']['source_package_sha256'] != package_hashes(pending)):
                    raise ValueError('Ready package differs from its pending source package')
                receipt = digest(ready / 'metadata.json')
                previous = state['records'].get(uid, {})
                if previous.get('status') == 'complete' and previous['metadata_sha256'] != receipt:
                    raise ValueError('Completed quality package changed')
                if schema is not None and metadata['schema_sha256'] != schema:
                    raise ValueError('Development packages have incompatible schemas')
                schema = metadata['schema_sha256']
                state['records'][uid] = {'status': 'complete', 'metadata_sha256': receipt,
                    'valid_counts': metadata['valid_counts'],
                    'word_observed_counts': metadata['quality_extraction']['word_observed_counts'],
                    'recording_observed': metadata['quality_extraction']['recording_observed']}
            except Exception as error:
                state.update(status='failed', training_ready=False)
                state['last_error'] = {'uid': uid, 'error': f'{type(error).__name__}: {error}'}
                write_json(state_path, state)
                raise
            write_json(state_path, state)
            print(f'[{ordinal}/{len(plan["development_uids"])}] ready train/{uid}', flush=True)
        params = SimpleNamespace(**{**TRAIN_DEFAULT_SETTINGS, 'dataset': 'adresso',
            'dataset_root': plan['dataset_root'], 'number_classes': 2, 'num_folds': num_folds,
            'precomputed_input_type': 'frames', 'precomputed_features_dir': str(root / 'quality/train')})
        fold_checks = []
        for fold in range(num_folds):
            train = make_precomputed_dataset(params, split='train', fold=fold)
            validation = make_precomputed_dataset(params, split='val', fold=fold)
            normalization = fit_quality_normalization(train)
            fold_checks.append({'fold': fold, 'training_recordings': len(train),
                'validation_recordings': len(validation),
                'word_observations': normalization['word']['observations'],
                'recording_observations': normalization['recording']['observations']})
        # Fold checks above are discarded statistics, not a cohort-wide scaler.
        if any(digest(path) != checksum for path, checksum in plan['source_sha256'].items()):
            raise ValueError('Source export changed during cohort preparation')
        state.update(status='complete', training_ready=True, schema_sha256=schema,
            completed=len(state['records']), fold_checks=fold_checks,
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        state.pop('last_error', None)
        write_json(state_path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.dry_run:
        print(json.dumps(cohort_plan(args.feature_root, args.dataset_root, args.num_folds), indent=2))
    else:
        prepare_cohort(args.feature_root, args.dataset_root, args.output_root, args.num_folds)


if __name__ == '__main__':
    main()
