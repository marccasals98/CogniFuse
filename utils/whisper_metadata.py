"""Additional Whisper metadata for the whole-recording preprocessing workflow."""

import json
import os
from pathlib import Path
import tempfile


def save_whisper_metadata(path, *, audio_path, result, words, text,
                          whisper_model, whisper_version, sample_rate,
                          decoding_settings, excluded_intervals):
    """Save raw output alongside the exact cleaned words retained in the CSV.

    Whole-recording timestamps already use the original recording's origin.
    Unspecified decoding options use the defaults of the recorded Whisper
    version. This records provenance; it does not implement cache reuse.
    """
    source = Path(audio_path)
    source_stat = source.stat()
    metadata = {
        'format': 'recording_whisper_v1',
        'text': text,
        'language': result.get('language'),
        'timestamp_reference': 'recording_seconds',
        'source': {
            'path': str(source.resolve()),
            'size_bytes': source_stat.st_size,
            'mtime_ns': source_stat.st_mtime_ns,
        },
        'configuration': {
            'whisper_model': whisper_model,
            'whisper_version': whisper_version,
            'sample_rate': sample_rate,
            'scope': 'whole_recording',
            'window_start': 0.0,
            'window_duration': None,
            'stride': None,
            'decoding_settings': dict(decoding_settings),
            'unspecified_decoding_settings': 'whisper_version_defaults',
            'text_cleaning': 'clean_text_v1',
            'speaker_filter': 'investigator_intervals_v1',
            'excluded_intervals': [list(interval) for interval in excluded_intervals],
        },
        'words': [
            dict(word, word_id=index, absolute_start=word['start'],
                 absolute_end=word['end'])
            for index, word in enumerate(words)
        ],
        # Includes words excluded from the CSV, segment scores and token IDs.
        'whisper_result': result,
    }
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=path.name + '.', suffix='.tmp', delete=False,
        ) as stream:
            temporary_path = stream.name
            json.dump(metadata, stream, ensure_ascii=False, indent=2,
                      allow_nan=False, default=lambda value: value.item())
            stream.write('\n')
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)
