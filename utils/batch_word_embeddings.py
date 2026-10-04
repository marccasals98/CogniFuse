"""Resume-safe dataset preprocessing for the existing CTC + MrBERT workflow."""

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import uuid

from utils.package_word_embeddings import AUDIO_SUFFIX, TEXT_SUFFIX, package_word_embeddings


STAGES = ('frames', 'ctc', 'alignment', 'windows', 'comparison', 'audio_words', 'text_words', 'package')
DEPENDENCIES = {'frames': (), 'ctc': (), 'alignment': ('ctc',), 'windows': ('alignment',),
                'comparison': ('windows',), 'audio_words': ('frames', 'windows'),
                'text_words': ('audio_words',), 'package': ('audio_words', 'text_words')}
TARGETS = ('lvPPA', 'nfPPA', 'svPPA')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new_json(path, data):
    path = Path(path)
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.json-') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, path)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def cohort(labels_path, audio_dir, words_dir):
    """Match the existing dataset's column aliases, class filtering and UID rules."""
    with Path(labels_path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if not names:
            raise ValueError('Labels CSV has no columns')
        def column(aliases, default=None):
            for alias in aliases:
                for name in names:
                    if name.strip().lower() == alias.lower():
                        return name
            if default is not None:
                return default
            raise ValueError(f'Missing label column: {aliases}')
        filename_col = column(('filename', 'audio_path', 'audio', 'file'), names[0])
        patient_col = column(('NHC ID HSP', 'nhc_id_hsp', 'patient_id'))
        label_col = column(('DX_Pilar', 'dx_pilar', 'diagnosis', 'label'))
        patients, records = {}, {}
        for row in reader:
            filename, patient, label = (row[k].strip() for k in (filename_col, patient_col, label_col))
            if label not in TARGETS or filename in ('', 'nan') or patient in ('', 'nan'):
                continue
            if patient in patients and patients[patient] != label:
                raise ValueError(f'Conflicting diagnoses for patient {patient}')
            patients[patient] = label
            uid = Path(filename).stem
            if not uid or uid in ('.', '..'):
                raise ValueError(f'Invalid recording filename: {filename}')
            record = {'uid': uid, 'filename': filename, 'patient_id': patient, 'label': label,
                      'audio': str((Path(audio_dir) / Path(filename).name).resolve()),
                      'words': str((Path(words_dir) / (uid + '.csv')).resolve())}
            if uid in records and record != records[uid]:
                raise ValueError(f'Recording UID collision: {uid}')
            records[uid] = record
    if not records:
        raise ValueError('No target recordings in the labels CSV')
    return [records[uid] for uid in sorted(records)]


def audit_record(record, vocabulary, blank_id):
    """Check all word rows before GPU work; never edit or discard a transcript."""
    from utils.transcript_normalization import normalize_ctc_words
    if record.get('skip_unsupported_words', False):
        from utils.ctc_skipped_words import normalize_skipping_unsupported as normalize_ctc_words
    problems = []
    for field in ('audio', 'words'):
        if not Path(record[field]).is_file():
            problems.append({'word_id': None, 'word': '', 'error': f'Missing {field}: {record[field]}'})
    if problems:
        return problems
    with Path(record['words']).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'word', 'start', 'end'}.issubset(reader.fieldnames or []):
            return [{'word_id': None, 'word': '', 'error': 'Word CSV requires word/start/end'}]
        rows = list(reader)
    alignable = False
    for i, row in enumerate(rows):
        try:
            normalized = normalize_ctc_words([row['word']], vocabulary, blank_id)
            alignable |= bool(normalized['target_ids'])
            start, end = float(row['start']), float(row['end'])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                raise ValueError('Invalid Whisper timestamps')
            probability = row.get('probability')
            if probability not in (None, '') and not 0 <= float(probability) <= 1:
                raise ValueError('Invalid Whisper probability')
        except (ValueError, TypeError) as error:
            problems.append({'word_id': i, 'word': row['word'], 'error': str(error)})
    if not rows or not alignable:
        problems.append({'word_id': None, 'word': '', 'error': 'No alignable words'})
    return problems


def snapshot(directory):
    return {p.name: digest(p) for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.name != '_batch_complete.json'}


def completed_stage(root, stage, signature, action):
    """Reuse only hashed successes; interrupted attempts remain intact for inspection."""
    parent = Path(root) / stage
    if parent.exists():
        for marker in sorted(parent.glob('*/_batch_complete.json')):
            saved = read_json(marker)
            if saved['signature'] == signature:
                if snapshot(marker.parent) != saved['files']:
                    raise ValueError(f'Completed stage was modified: {marker.parent}')
                return marker.parent
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / uuid.uuid4().hex
    action(output)
    files = snapshot(output)
    if not files:
        raise ValueError(f'Stage produced no files: {stage}')
    write_new_json(output / '_batch_complete.json', {'signature': signature, 'files': files})
    return output


def publish_file(source, target):
    """Publish an independent copy without ever replacing an existing file."""
    source, target = Path(source), Path(target)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f'Conflicting collected export: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # Only this call's temporary file is removed. An interrupted copy never
    # appears under a training filename, and completed outputs are immutable.
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.copy-') as stream:
        with source.open('rb') as reader:
            shutil.copyfileobj(reader, stream)
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, target)


def validate_package(directory, record, config):
    from utils.prepro_word_text import read_audio_words
    import torch
    directory = Path(directory)
    metadata = read_json(directory / (record['uid'] + '_word_metadata.json'))
    if metadata['format'] != 'packaged_word_embeddings_v1' or metadata['uid'] != record['uid']:
        raise ValueError('Wrong packaged recording')
    if metadata['audio_source'] != identity(record['audio']) or metadata['word_source'] != identity(record['words']):
        raise ValueError('Packaged source recording or transcript changed')
    if metadata['text_model'] != config['text_model'] or metadata['text_model_revision'] != config['text_revision']:
        raise ValueError('Packaged text model differs from this experiment')
    for sources in metadata['inputs'].values():
        for source in sources.values():
            if digest(source['path']) != source['sha256']:
                raise ValueError(f'Packaged input changed: {source["path"]}')
    audio_dir = Path(metadata['inputs']['audio']['metadata.json']['path']).parent
    audio_meta, audio, audio_mask = read_audio_words(audio_dir)
    if (audio_meta['audio_model'] != config['audio_model'] or audio_meta['ctc_model'] != config['ctc_model']
            or audio_meta['context_frames'] != config['context_frames']):
        raise ValueError('Packaged acoustic model or CTC context differs')
    text_meta = read_json(metadata['inputs']['text']['metadata.json']['path'])
    from utils.ctc_skipped_words import skip_metadata
    omitted = skip_metadata(audio_meta)
    if omitted != skip_metadata(text_meta) or omitted != skip_metadata(metadata):
        raise ValueError('Packaged skipped-word audit differs from its sources')
    if omitted and not config.get('skip_unsupported_words', False):
        raise ValueError('A strict experiment cannot reuse exports that skipped unsupported words')
    if record.get('skip_unsupported_words', False):
        if record.get('skipped_words', []) != omitted.get('skipped_words', []):
            raise ValueError('Packaged skipped words differ from the current input audit')
    if omitted and audio_mask[omitted['skipped_word_ids']].any():
        raise ValueError('Skipped words must be invalid in acoustic masks')
    if text_meta['chunk_size'] != config['chunk_size'] or text_meta['words'] != audio_meta['words']:
        raise ValueError('Packaged text context or original words differ')
    expected = {record['uid'] + suffix for suffix in (AUDIO_SUFFIX, TEXT_SUFFIX,
                 AUDIO_SUFFIX[:-3] + '_mask.pt', TEXT_SUFFIX[:-3] + '_mask.pt')}
    if set(metadata['output_sha256']) != expected:
        raise ValueError('Incomplete packaged file manifest')
    for name, checksum in metadata['output_sha256'].items():
        if Path(name).name != name or digest(directory / name) != checksum:
            raise ValueError(f'Packaged output changed: {name}')
    text_dir = Path(metadata['inputs']['text']['metadata.json']['path']).parent
    text = torch.load(text_dir / 'text_word_embeddings.pt', map_location='cpu', weights_only=True)
    text_mask = torch.load(text_dir / 'text_word_mask.pt', map_location='cpu', weights_only=True)
    if (text.ndim != 2 or len(text) != len(audio) or not text.is_floating_point()
            or not torch.isfinite(text).all() or text_mask.dtype != torch.bool
            or text_mask.shape != audio_mask.shape):
        raise ValueError('Invalid source text vectors or masks')
    paired = audio_mask & text_mask
    if not paired.any() or metadata['words'] != audio_meta['words']:
        raise ValueError('Packaged word coverage is invalid')
    for suffix, original in ((AUDIO_SUFFIX, audio), (TEXT_SUFFIX, text)):
        saved = torch.load(directory / (record['uid'] + suffix), map_location='cpu', weights_only=True)
        mask = torch.load(directory / (record['uid'] + suffix[:-3] + '_mask.pt'), map_location='cpu', weights_only=True)
        if not torch.equal(saved, original.float()) or not torch.equal(mask, paired):
            raise ValueError('Packaged vectors or paired masks differ from their sources')
    return metadata, Path(audio_meta['inputs']['word_windows']['path']).parent


class StageWorker:
    """Load each pretrained encoder once per stage, keeping only one on the GPU."""
    def __init__(self, config, device, local_files_only):
        self.config, self.device, self.local = config, device, local_files_only
        self.loaded = None

    def close(self):
        import torch
        self.loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, stage, record, previous, output):
        cfg = self.config
        if stage == 'frames':
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            from utils.prepro_audio_frames import export_recording_frames
            if self.loaded is None:
                options = {'revision': cfg['audio_revision'], 'local_files_only': self.local}
                self.loaded = (Wav2Vec2Processor.from_pretrained(cfg['audio_model'], **options),
                               Wav2Vec2Model.from_pretrained(cfg['audio_model'], **options).to(self.device).eval())
            export_recording_frames(record['audio'], output, *self.loaded, self.device)
        elif stage == 'ctc':
            from scripts.ctc_encoder import SpanishCTCEncoder
            from utils.prepro_ctc import export_ctc
            if self.loaded is None:
                self.loaded = SpanishCTCEncoder.from_pretrained(cfg['ctc_model'], device=self.device,
                              revision=cfg['ctc_revision'], local_files_only=self.local)
            self.loaded.skip_unsupported_words = cfg.get('skip_unsupported_words', False)
            export_ctc(record['audio'], record['words'], output, self.loaded)
        elif stage == 'alignment':
            from utils.ctc_forced_alignment import export_alignment
            export_alignment(previous['ctc'], output)
        elif stage == 'windows':
            from utils.ctc_word_windows import export_word_windows
            export_word_windows(previous['alignment'], output, cfg['context_frames'])
        elif stage == 'comparison':
            from utils.compare_word_alignment import inspect_word_alignment
            inspect_word_alignment(previous['windows'], output)
        elif stage == 'audio_words':
            from utils.word_audio_pool import export_audio_words
            export_audio_words(previous['frames'], previous['windows'], output)
        elif stage == 'text_words':
            from transformers import AutoModel, AutoTokenizer
            from utils.prepro_word_text import export_text_words
            if self.loaded is None:
                options = {'revision': cfg['text_revision'], 'local_files_only': self.local}
                model, loading = AutoModel.from_pretrained(cfg['text_model'], output_loading_info=True, **options)
                missing = [key for key in loading.get('missing_keys', []) if 'pooler.' not in key]
                if missing or loading.get('mismatched_keys') or loading.get('error_msgs'):
                    raise ValueError(f'Incomplete text encoder weights: {loading}')
                self.loaded = (AutoTokenizer.from_pretrained(cfg['text_model'], use_fast=True, **options), model)
            export_text_words(previous['audio_words'], output, *self.loaded,
                              cfg['text_model'], self.device, cfg['chunk_size'])
        elif stage == 'package':
            package_word_embeddings(previous['audio_words'], previous['text_words'], output)
        else:
            raise ValueError(stage)


def verify_training_cohort(records, labels, embeddings):
    """Use the unchanged real loader and verify every fold and tensor pair."""
    from scripts.data import PrecomputedADDataset
    from scripts.settings import TRAIN_DEFAULT_SETTINGS
    params = SimpleNamespace(**{**TRAIN_DEFAULT_SETTINGS, 'train_labels_path': str(labels),
               'validation_labels_path': str(labels), 'precomputed_features_dir': str(embeddings),
               'precomputed_audio_suffix': AUDIO_SUFFIX, 'precomputed_text_suffix': TEXT_SUFFIX})
    expected = {r['uid'] for r in records}
    folds = []
    held_out = []
    for fold in range(params.num_folds):
        train = PrecomputedADDataset(params, split='train', fold=fold)
        val = PrecomputedADDataset(params, split='val', fold=fold)
        if not len(train) or not len(val):
            raise ValueError(f'Empty split in fold {fold}')
        if set(train.df.patient_id) & set(val.df.patient_id):
            raise ValueError('Patient overlap between training and validation')
        if {r['uid'] for d in (train, val) for r in d.segments} != expected:
            raise ValueError('Loader cohort differs from preprocessing cohort')
        held_out.extend(r['uid'] for r in val.segments)
        if fold == 0:
            shapes = set()
            for dataset in (train, val):
                for index in range(len(dataset)):
                    speech, _, text, _, _ = dataset[index]
                    shapes.add((speech.shape[1], text.shape[1]))
            if len(shapes) != 1:
                raise ValueError('Mixed feature dimensions in training cohort')
        folds.append({'fold': fold, 'train_recordings': len(train), 'validation_recordings': len(val)})
    if len(held_out) != len(expected) or set(held_out) != expected:
        raise ValueError('Every recording must be held out exactly once')
    return folds


def run_batch(args, records, config, issues):
    from utils.compare_word_alignment import inspect_word_alignment
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another batch process is using this output root')
        plan = {'format': 'word_batch_plan_v1', 'config': config,
                'labels': {'path': str(args.labels_csv.resolve()), 'sha256': digest(args.labels_csv)},
                'records': [{key: value for key, value in record.items()
                             if not key.endswith(('_identity', '_sha256'))} for record in records]}
        plan_path = root / 'plan.json'
        if plan_path.exists():
            if read_json(plan_path) != plan:
                raise ValueError('Inputs or settings changed. Choose a new --output-root; existing exports are preserved.')
        else:
            write_new_json(plan_path, plan)
        plan_hash = digest(plan_path)
        run_id = uuid.uuid4().hex
        reports = root / 'reports'
        reports.mkdir(exist_ok=True)
        with (reports / (run_id + '_input_issues.csv')).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=('uid', 'words_csv', 'word_id', 'word', 'error'))
            writer.writeheader()
            for uid, problems in issues.items():
                record = next(r for r in records if r['uid'] == uid)
                for problem in problems:
                    writer.writerow({'uid': uid, 'words_csv': record['words'], **problem})
        failures = {uid: {'stage': 'input', 'errors': problems} for uid, problems in issues.items()}
        states = {r['uid']: {} for r in records}
        packages, comparisons = {}, {}
        # Explicit reuse directories only: never mistake a legacy embedding for this experiment.
        for record in records:
            uid = record['uid']
            if uid in failures:
                continue
            for directory in args.reuse_packaged_dir:
                if (directory / (uid + '_word_metadata.json')).is_file():
                    _, windows = validate_package(directory, record, config)
                    packages[uid] = directory.resolve()
                    comparisons[uid] = inspect_word_alignment(windows)['summary']
                    print(f'REUSE completed recording: {uid}', flush=True)
                    break
        worker = StageWorker(config, args.device, args.local_files_only)
        for stage in STAGES:
            print(f'STAGE {stage}', flush=True)
            try:
                for index, record in enumerate(records, 1):
                    uid = record['uid']
                    if uid in failures or uid in packages:
                        continue
                    previous = states[uid]
                    signature = {'plan_sha256': plan_hash, 'uid': uid, 'stage': stage,
                                 'sources': {key: record.get(key) for key in
                                             (('audio_identity', 'audio_sha256') if stage == 'frames' else
                                              ('audio_identity', 'audio_sha256', 'words_identity', 'words_sha256'))},
                                 'dependencies': {key: digest(previous[key] / '_batch_complete.json')
                                                  for key in DEPENDENCIES[stage]}}
                    try:
                        path = completed_stage(root / 'recordings' / uid, stage, signature,
                               lambda output: worker.run(stage, record, previous, output))
                        previous[stage] = path
                        if stage == 'comparison':
                            comparisons[uid] = read_json(path / 'comparison.json')['summary']
                        if stage == 'package':
                            validate_package(path, record, config)
                            packages[uid] = path
                        print(f'[{index}/{len(records)}] {stage} OK: {uid}', flush=True)
                    except Exception as error:
                        failures[uid] = {'stage': stage, 'error': f'{type(error).__name__}: {error}'}
                        write_new_json(reports / (run_id + '_' + uid + '_' + stage + '_error.json'), failures[uid])
                        print(f'FAILED {uid} ({stage}): {error}', flush=True)
                        worker.close()
            finally:
                worker.close()
        embeddings = root / 'embeddings'
        for uid, directory in packages.items():
            metadata = read_json(directory / (uid + '_word_metadata.json'))
            for name in [*metadata['output_sha256'], uid + '_word_metadata.json']:
                publish_file(directory / name, embeddings / name)
        summary = {'format': 'word_batch_report_v1', 'total_recordings': len(records),
                   'skipped_words': {r['uid']: r['skipped_words'] for r in records if r.get('skipped_words')},
                   'skipped_word_count': sum(len(r.get('skipped_words', [])) for r in records),
                   'packaged_recordings': len(packages), 'failures': failures,
                   'alignment_review': comparisons, 'embeddings_dir': str(embeddings),
                   'training_ready': False, 'folds': []}
        if not failures and len(packages) == len(records):
            try:
                summary['folds'] = verify_training_cohort(records, args.labels_csv, embeddings)
                summary['training_ready'] = True
            except Exception as error:
                summary['loader_error'] = f'{type(error).__name__}: {error}'
        report_path = reports / (run_id + '_summary.json')
        write_new_json(report_path, summary)
        print(f"Packaged {len(packages)}/{len(records)}; training_ready={summary['training_ready']}", flush=True)
        print(f'Report: {report_path}', flush=True)
        return summary


def main():
    from transformers import AutoConfig, AutoTokenizer
    from utils.prepro_audio_frames import MODEL_ID
    from scripts.ctc_encoder import DEFAULT_CTC_MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels-csv', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--words-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reuse-packaged-dir', type=Path, action='append', default=[])
    parser.add_argument('--text-model', default='BSC-LT/MrBERT')
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--ctc-context-frames', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--skip-unsupported-words', action='store_true',
                        help='Keep recordings and word IDs, but mask unsupported CTC word pairs.')
    parser.add_argument('--dry-run', action='store_true', help='Audit inputs without weights, inference, or writes.')
    args = parser.parse_args()
    if args.chunk_size < 3 or args.ctc_context_frames < 0:
        parser.error('chunk-size must be >= 3 and context frames >= 0')
    records = cohort(args.labels_csv, args.audio_dir, args.words_dir)
    config = {'audio_model': MODEL_ID, 'ctc_model': DEFAULT_CTC_MODEL, 'text_model': args.text_model,
              'chunk_size': args.chunk_size, 'context_frames': args.ctc_context_frames}
    if args.skip_unsupported_words:
        config['skip_unsupported_words'] = True
    configs = {}
    for kind in ('audio', 'ctc', 'text'):
        configs[kind] = AutoConfig.from_pretrained(config[kind + '_model'], local_files_only=args.local_files_only)
        config[kind + '_revision'] = configs[kind]._commit_hash
    if args.chunk_size > configs['text'].max_position_embeddings:
        parser.error('chunk-size exceeds the text model context limit')
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CTC_MODEL, revision=config['ctc_revision'],
                                             local_files_only=args.local_files_only)
    issues = {}
    for record in records:
        if args.skip_unsupported_words:
            record['skip_unsupported_words'] = True
        problems = audit_record(record, tokenizer.get_vocab(), configs['ctc'].pad_token_id)
        if args.skip_unsupported_words and not problems:
            from utils.ctc_skipped_words import normalize_skipping_unsupported
            from utils.prepro_ctc import read_words
            normalized = normalize_skipping_unsupported(read_words(record['words']), tokenizer.get_vocab(),
                                                        configs['ctc'].pad_token_id)
            record['skipped_words'] = normalized.get('skipped_words', [])
        if problems:
            issues[record['uid']] = problems
        for field in ('audio', 'words'):
            if Path(record[field]).is_file():
                record[field + '_identity'] = identity(record[field])
                record[field + '_sha256'] = digest(record[field])
    print(f'Recordings: {len(records)}; valid inputs: {len(records)-len(issues)}; input issues: {len(issues)}', flush=True)
    if args.skip_unsupported_words:
        print(f"Words to mask: {sum(len(r.get('skipped_words', [])) for r in records)} "
              f"across {sum(bool(r.get('skipped_words')) for r in records)} recordings; original rows retained.", flush=True)
    if args.dry_run:
        for uid, problems in issues.items():
            for problem in problems:
                print(f"{uid}: word {problem['word_id']} {problem['word']!r}: {problem['error']}")
        print('Dry run only; no embeddings or model weights created.')
        return
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but no GPU is available. Run in a GPU allocation or use --device cpu.')
    summary = run_batch(args, records, config, issues)
    if not summary['training_ready']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
