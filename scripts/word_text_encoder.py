"""Encode original words with a fast tokenizer and mean-pool their subwords."""

import torch


def word_chunks(words, tokenizer, chunk_size=512):
    """Tokenize the exact word list and pack complete words without truncation."""
    if not tokenizer.is_fast:
        raise ValueError('Word alignment requires a fast tokenizer with word_ids()')
    if any(not isinstance(word, str) for word in words):
        raise ValueError('Every original word must be a string')
    if type(chunk_size) is not int or not tokenizer.num_special_tokens_to_add(pair=False) < chunk_size <= tokenizer.model_max_length:
        raise ValueError('chunk_size must leave room for content and fit the tokenizer context')
    if not words:
        return [], []
    full = tokenizer(words, is_split_into_words=True, add_special_tokens=False,
                     truncation=False, verbose=False)
    full_ids, owners = full['input_ids'], full.word_ids()
    by_word = [[] for _ in words]
    previous = -1
    for token_id, owner in zip(full_ids, owners):
        if owner is None or not previous <= owner < len(words) or owner < 0:
            raise ValueError('Tokenizer returned invalid or nonmonotonic original word IDs')
        by_word[owner].append(token_id)
        previous = owner
    if len(owners) != len(full_ids):
        raise ValueError('Tokenizer word IDs and tokens have different lengths')
    capacity = chunk_size - tokenizer.num_special_tokens_to_add(pair=False)
    ranges, first, used = [], 0, 0
    for word_id, ids in enumerate(by_word):
        if len(ids) > capacity:
            raise ValueError(f'Word {word_id} needs {len(ids)} subwords, exceeding chunk capacity {capacity}')
        if used and used + len(ids) > capacity:
            ranges.append((first, word_id))
            first, used = word_id, 0
        used += len(ids)
    if used:
        ranges.append((first, len(words)))
    chunks, seen_ids, seen_owners = [], [], []
    for first, end in ranges:
        encoded = tokenizer(words[first:end], is_split_into_words=True,
                            truncation=False, padding=False, return_tensors='pt')
        if encoded['input_ids'].shape[1] > chunk_size:
            raise ValueError('Word chunk exceeds its declared token budget')
        local_owners = encoded.word_ids()
        global_owners = [None if owner is None else first + owner for owner in local_owners]
        for position, owner in enumerate(global_owners):
            if owner is not None:
                if not encoded['attention_mask'][0, position]:
                    raise ValueError('An original word token was marked as padding')
                seen_ids.append(int(encoded['input_ids'][0, position]))
                seen_owners.append(owner)
        inputs = {name: encoded[name] for name in tokenizer.model_input_names if name in encoded}
        chunks.append({'inputs': inputs, 'word_ids': global_owners,
                       'word_start': first, 'word_end_exclusive': end})
    if seen_ids != full_ids or seen_owners != owners:
        raise ValueError('Chunking changed original tokens or their word correspondence')
    return chunks, by_word


@torch.inference_mode()
def encode_word_text(words, tokenizer, model, device='cpu', chunk_size=512):
    """Return [N_words, D_text], [N_words] validity, and auditable subword maps.

    Added special tokens have no word ID and are excluded from pooling. Words
    yielding no tokens retain a zero row and false mask. Each nonempty word is
    encoded in exactly one chunk, so all its subwords share the same context.
    """
    words = list(words)
    if chunk_size > model.config.max_position_embeddings:
        raise ValueError('chunk_size exceeds the text encoder context limit')
    chunks, by_word = word_chunks(words, tokenizer, chunk_size)
    if not words:
        by_word = []
    model.to(device).eval()
    features = torch.zeros(len(words), model.config.hidden_size, dtype=torch.float32)
    mask = torch.zeros(len(words), dtype=torch.bool)
    rows = [
        {'word_id': i, 'word': word, 'status': 'no_text_tokens', 'chunk_index': None,
         'subword_ids': by_word[i], 'subwords': tokenizer.convert_ids_to_tokens(by_word[i]),
         'subword_count': len(by_word[i]), 'unknown_subword_count': sum(t == tokenizer.unk_token_id for t in by_word[i])}
        for i, word in enumerate(words)
    ]
    chunk_metadata = []
    for chunk_id, chunk in enumerate(chunks):
        inputs = {name: value.to(device) for name, value in chunk['inputs'].items()}
        output = model(**inputs).last_hidden_state
        if output.shape != (1, inputs['input_ids'].shape[1], features.shape[1]) or not torch.isfinite(output).all():
            raise ValueError('Text encoder produced invalid hidden states')
        owners = chunk['word_ids']
        for word_id in range(chunk['word_start'], chunk['word_end_exclusive']):
            positions = [i for i, owner in enumerate(owners) if owner == word_id]
            if not positions:
                continue
            if mask[word_id] or len(positions) != len(by_word[word_id]):
                raise ValueError('A word was split, duplicated or lost during text encoding')
            features[word_id] = output[0, positions].float().mean(dim=0).cpu()
            mask[word_id] = True
            rows[word_id].update(status='encoded', chunk_index=chunk_id)
        chunk_metadata.append({'chunk_index': chunk_id, 'word_start': chunk['word_start'],
                               'word_end_exclusive': chunk['word_end_exclusive'],
                               'input_tokens': inputs['input_ids'].shape[1]})
    if mask.tolist() != [bool(ids) for ids in by_word] or not torch.isfinite(features).all():
        raise ValueError('Text word coverage or pooled embeddings are invalid')
    return features, mask, {'words': words, 'word_mappings': rows, 'chunks': chunk_metadata,
                            'chunk_size': chunk_size, 'pooling': 'mean_subwords',
                            'special_tokens_pooled': False, 'truncated_tokens': 0}
