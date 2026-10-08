"""Deterministic character-CTC targets with unchanged original word indices."""

import unicodedata


def normalize_ctc_words(words, vocabulary, blank_id, delimiter='|'):
    """Lowercase NFC text, remove punctuation, and preserve Spanish accents.

    Punctuation-only words remain in the mapping with status ``no_ctc_units``.
    Unsupported letters, numbers and symbols fail explicitly, never becoming
    unknown tokens. Delimiters belong to no word (word ID -1). Token spans use
    exclusive ends. No BERT tokenizer is involved.
    """
    if delimiter not in vocabulary or vocabulary[delimiter] == blank_id:
        raise ValueError('CTC vocabulary requires a nonblank word delimiter')
    words = list(words)
    targets, tokens, owners, mappings = [], [], [], []
    for word_id, original in enumerate(words):
        if not isinstance(original, str):
            raise ValueError(f'Word {word_id} is not a string')
        normalized = unicodedata.normalize('NFC', original.strip().lower())
        if any(char.isspace() for char in normalized):
            raise ValueError(f'Word {word_id} contains internal whitespace: {original!r}')
        normalized = ''.join(char for char in normalized
                             if not unicodedata.category(char).startswith('P'))
        for char in normalized:
            if (not char.isalpha() or char not in vocabulary
                    or vocabulary[char] == blank_id or char == delimiter):
                raise ValueError(f'Unsupported CTC character {char!r} in word {word_id}: {original!r}')
        if normalized and targets:
            tokens.append(delimiter)
            targets.append(vocabulary[delimiter])
            owners.append(-1)
        start = len(targets)
        for char in normalized:
            tokens.append(char)
            targets.append(vocabulary[char])
            owners.append(word_id)
        mappings.append({
            'word_id': word_id, 'word': original, 'ctc_form': normalized,
            'token_start': start, 'token_end': len(targets),
            'status': 'ready' if normalized else 'no_ctc_units',
        })
    return {
        'normalization': 'spanish_nfc_lower_remove_punctuation_v1',
        'words': words, 'word_mappings': mappings,
        'ctc_text': ' '.join(item['ctc_form'] for item in mappings if item['ctc_form']),
        'tokens': tokens, 'target_ids': targets, 'token_word_ids': owners,
        'blank_id': blank_id, 'word_delimiter': delimiter,
    }
