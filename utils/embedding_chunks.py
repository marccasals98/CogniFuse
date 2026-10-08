"""Encode every transcript token in bounded, non-overlapping model inputs."""

import torch


def transcript_chunks(tokenizer, transcription, chunk_size=200):
    """Return chunks with absolute character offsets and positions to retain.

    Every content token is retained once. Each model input has its own special
    tokens, but the joined recording keeps only the first prefix and last suffix.
    Padding and internal chunk boundary special tokens are discarded.
    """
    if not tokenizer.is_fast:
        raise ValueError("Full-transcript alignment requires a fast tokenizer")
    if chunk_size <= tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError("chunk_size must leave room for content tokens")
    if chunk_size > tokenizer.model_max_length:
        raise ValueError("chunk_size exceeds the tokenizer context limit")
    encoded = tokenizer(
        transcription, return_tensors="pt", padding="max_length",
        truncation=True, max_length=chunk_size, stride=0,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    chunks = []
    for i in range(encoded["input_ids"].shape[0]):
        valid = encoded["attention_mask"][i].bool()
        content = valid & ~encoded["special_tokens_mask"][i].bool()
        keep = content.clone()
        positions = content.nonzero().flatten()
        if not len(positions):
            keep = valid  # Empty transcript: retain its special tokens.
        else:
            if i == 0:
                keep[:positions[0]] = valid[:positions[0]]
            if i == encoded["input_ids"].shape[0] - 1:
                keep[positions[-1] + 1:] = valid[positions[-1] + 1:]
        inputs = {name: encoded[name][i:i + 1] for name in tokenizer.model_input_names if name in encoded}
        chunks.append((inputs, encoded["offset_mapping"][i][keep], keep))

    # Enforce complete ordered coverage, including repeated words and subwords.
    full = tokenizer(transcription, truncation=False, return_offsets_mapping=True, verbose=False)
    joined_ids = torch.cat([inputs["input_ids"][0][keep] for inputs, _, keep in chunks]).tolist()
    joined_offsets = torch.cat([offsets for _, offsets, _ in chunks]).tolist()
    if joined_ids != full["input_ids"] or joined_offsets != [list(x) for x in full["offset_mapping"]]:
        raise ValueError("Chunking did not preserve every transcript token and offset exactly once")
    return chunks


def encode_full_transcript(tokenizer, model, transcription, device, chunk_size=200):
    limit = getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
    if chunk_size > limit:
        raise ValueError("chunk_size exceeds the text encoder context limit")
    chunks = transcript_chunks(tokenizer, transcription, chunk_size)
    features, offsets = [], []
    with torch.no_grad():
        for inputs, chunk_offsets, keep in chunks:
            output = model(**{name: value.to(device) for name, value in inputs.items()})
            features.append(output.last_hidden_state[0].cpu()[keep])
            offsets.extend(chunk_offsets.tolist())
    features = torch.cat(features)
    if not torch.isfinite(features).all():
        raise ValueError("Text encoder produced non-finite embeddings")
    return features, offsets, len(chunks)
