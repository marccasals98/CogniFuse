"""Enrich an existing frame package with raw reliability measurements."""

import argparse
import copy
from pathlib import Path

import torch
import torchaudio

from utils.fusion_quality import quality_schema, recording_quality, word_quality
from utils.package_fusion_frames import digest, publish_fusion_package, read_fusion_package, schema_fingerprint


def enrich_package(package_dir, output_dir, audio_path=None):
    source, output = Path(package_dir), Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Export already exists: {output}')
    metadata, tensors = read_fusion_package(source)
    if metadata['schema']['quality']['status'] != 'pending':
        raise ValueError('Quality extraction requires a pending source package')
    source_hashes = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    candidates = [(path, checksum) for path, checksum in metadata['provenance']['verified_sha256'].items()
                  if Path(path).stem == metadata['uid']
                  and Path(path).suffix.lower() in ('.wav', '.mp3', '.flac', '.alac', '.m4a', '.ogg')]
    if len(candidates) != 1:
        raise ValueError('Expected one original audio identity in package provenance')
    original, checksum = candidates[0]
    audio = Path(audio_path) if audio_path is not None else Path(original)
    if digest(audio) != checksum:
        raise ValueError('Waveform differs from the original package audio')
    word, word_observed = word_quality(metadata)
    waveform, rate = torchaudio.load(audio)
    recording, recording_observed, diagnostics = recording_quality(waveform, rate)
    enriched = copy.deepcopy(metadata)
    enriched['schema']['quality'] = quality_schema()
    enriched['schema_sha256'] = schema_fingerprint(enriched['schema'])
    enriched['quality_extraction'] = {'version': 'fusion_quality_v1',
        'source_package': str(source.resolve()), 'source_package_sha256': source_hashes,
        'audio': {'path': str(audio.resolve()), 'sha256': checksum}, 'diagnostics': diagnostics,
        'word_observed_counts': word_observed.sum(dim=0).tolist(),
        'recording_observed': recording_observed.tolist()}
    tensors.update({'word_quality.pt': word, 'word_quality_observed.pt': word_observed,
                    'recording_quality.pt': recording, 'recording_quality_observed.pt': recording_observed})
    if source_hashes != {p.name: digest(p) for p in source.iterdir() if p.is_file()} or digest(audio) != checksum:
        raise ValueError('Quality source changed during extraction')
    return publish_fusion_package(enriched, tensors, output, copy_from=source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--audio-path', type=Path, help='Optional relocated original audio with matching SHA-256.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = enrich_package(args.package_dir, args.output_dir, args.audio_path)
    print(f'Enriched {metadata["partition"]}/{metadata["uid"]}: '
          f'{metadata["quality_extraction"]["word_observed_counts"]} word observations; '
          f'{metadata["quality_extraction"]["recording_observed"]} recording availability')


if __name__ == '__main__':
    main()
