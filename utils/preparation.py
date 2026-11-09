"""Source provenance, immutable stage caching and word-package validation."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid
from utils.package_word_embeddings import AUDIO_SUFFIX, TEXT_SUFFIX

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

def source_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest()}

def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)
