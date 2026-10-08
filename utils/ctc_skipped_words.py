"""Opt-in omission of unsupported CTC words without deleting original rows."""

import copy

from utils.transcript_normalization import normalize_ctc_words


SKIP_FIELDS = ('unsupported_word_policy', 'skipped_word_ids', 'skipped_words')


def skip_metadata(source):
    """Keep strict exports unchanged; propagate an explicit omission audit otherwise."""
    if source.get('unsupported_word_policy') == 'skip':
        return {key: copy.deepcopy(source[key]) for key in SKIP_FIELDS}
    return {}


def normalize_skipping_unsupported(words, vocabulary, blank_id, delimiter='|'):
    """Omit whole unsupported words from targets, keeping their IDs and original text.

    Only unsupported-character errors are eligible. Invalid vocabulary, internal
    whitespace, and non-string inputs remain errors. Empty/punctuation-only words
    retain the existing no_ctc_units behavior and are not counted as skipped.
    """
    words = list(words)
    normalize_ctc_words([], vocabulary, blank_id, delimiter)
    projected = list(words)
    skipped = []
    for word_id, word in enumerate(words):
        try:
            normalize_ctc_words([word], vocabulary, blank_id, delimiter)
        except ValueError as error:
            if not str(error).startswith('Unsupported CTC character '):
                raise ValueError(f'Word {word_id}: {error}') from error
            projected[word_id] = ''
            skipped.append({'word_id': word_id, 'word': word,
                            'reason': 'unsupported_ctc_character',
                            'detail': str(error).replace('in word 0:', f'in word {word_id}:', 1)})
    result = normalize_ctc_words(projected, vocabulary, blank_id, delimiter)
    result['words'] = words
    for row, original in zip(result['word_mappings'], words):
        row['word'] = original
    if skipped:
        result.update(normalization='spanish_nfc_lower_remove_punctuation_skip_unsupported_v1',
                      unsupported_word_policy='skip',
                      skipped_word_ids=[row['word_id'] for row in skipped], skipped_words=skipped)
    return result


def _projection(source, vocabulary, blank_id, delimiter):
    normalized = normalize_skipping_unsupported(source['words'], vocabulary, blank_id, delimiter)
    if normalized.get('unsupported_word_policy') != 'skip':
        raise ValueError('Skip policy has no unsupported words to omit')
    for key in (*SKIP_FIELDS, 'word_mappings'):
        if source.get(key) != normalized[key]:
            raise ValueError(f'Inconsistent skipped-word metadata: {key}')
    excluded = set(normalized['skipped_word_ids'])
    projected = ['' if i in excluded else word for i, word in enumerate(source['words'])]
    return normalized, normalize_ctc_words(projected, vocabulary, blank_id, delimiter)


def align_skipping_unsupported(log_probs, valid_mask, transcript, metadata, strict_aligner):
    """Use the existing aligner on empty placeholders, then restore original text."""
    normalized, projected = _projection(transcript, metadata['vocabulary'], metadata['blank_id'],
                                        transcript['word_delimiter'])
    if any(transcript.get(key) != value for key, value in normalized.items()):
        raise ValueError('Saved skipped transcript targets are inconsistent')
    alignment, path, scores = strict_aligner(log_probs, valid_mask, projected, metadata)
    alignment.update(words=list(transcript['words']), word_mappings=copy.deepcopy(transcript['word_mappings']),
                     normalization_vocabulary=copy.deepcopy(metadata['vocabulary']),
                     word_delimiter=transcript['word_delimiter'], **skip_metadata(transcript))
    return alignment, path, scores


def merge_skipping_unsupported(alignment, context_frames, strict_merger):
    """Retain null windows at skipped IDs; validate against the full CTC vocabulary."""
    normalized, projected = _projection(alignment, alignment['normalization_vocabulary'],
                                        alignment['blank_id'], alignment['word_delimiter'])
    if normalized['target_ids'] != [unit['token_id'] for unit in alignment['units']]:
        raise ValueError('CTC units differ from the skipped transcript targets')
    projected_alignment = {key: value for key, value in alignment.items() if key not in SKIP_FIELDS}
    projected_alignment.update(words=projected['words'], word_mappings=projected['word_mappings'])
    result = strict_merger(projected_alignment, context_frames)
    result['words'] = list(alignment['words'])
    for row, original in zip(result['word_windows'], result['words']):
        row['word'] = original
    result.update(skip_metadata(alignment))
    return result
