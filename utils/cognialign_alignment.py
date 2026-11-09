"""Whisper word/pause intervals mapped to individual text subwords.

Implements Sections III-A--C of CogniAlign (2506.01890v2). There is no
CTC alignment, timestamp repair, text-word pooling, or short-window expansion.
"""

import math

import torch


def pause_symbol(seconds):
    """Paper thresholds: [0.5, 1) comma, [1, 1.5] period, >1.5 ellipsis."""
    if seconds > 1.5:
        return "..."
    if seconds >= 1.0:
        return "."
    if seconds >= 0.5:
        return ","
    return None


def build_transcript(rows, duration, pauses=True):
    """Build text and explicit character spans from unchanged Whisper word rows.

    Whitespace around a word is stripped. Its spelling and punctuation are
    preserved. Inserted pause symbols precede the next word and point to the
    actual intervening silence. Bad timestamps fail; they are never repaired.
    """
    if not rows or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Expected word rows and a positive recording duration")
    pieces, spans, cursor, previous = [], [], 0, None

    def append(text, kind, start, end, word_id=None, preceding_word_id=None):
        nonlocal cursor
        if pieces:
            cursor += 1
        spans.append({"text": text, "kind": kind, "start": start, "end": end,
                      "char_start": cursor, "char_end": cursor + len(text),
                      "word_id": word_id, "preceding_word_id": preceding_word_id})
        pieces.append(text)
        cursor += len(text)

    for word_id, row in enumerate(rows):
        word = row["word"]
        if not isinstance(word, str) or not word.strip():
            raise ValueError(f"Empty word at original word ID {word_id}")
        word = word.strip()
        if any(char.isspace() for char in word):
            raise ValueError(f"Word row {word_id} contains multiple words")
        start, end = float(row["start"]), float(row["end"])
        if not (math.isfinite(start) and math.isfinite(end)
                and 0 <= start <= end <= duration + 0.05):
            raise ValueError(f"Invalid Whisper timestamp at word ID {word_id}")
        if previous is not None:
            if start < previous[0]:
                raise ValueError(f"Nonmonotonic Whisper start at word ID {word_id}; no repair permitted")
            symbol = pause_symbol(start - previous[1]) if pauses else None
            if symbol:
                append(symbol, "pause", previous[1], start, preceding_word_id=word_id - 1)
        append(word, "word", start, end, word_id=word_id)
        previous = start, end
    return " ".join(pieces), spans


def align_token_audio(features, offsets, spans, frame_rate=50.0):
    """Average frames in each interval, repeat that vector for its subwords.

    Nominal frame timestamps are j/frame_rate, with inclusive interval bounds
    as in the paper's equation. Empty/zero-duration intervals stay zero and
    masked, with an audit entry; no neighbouring frames are borrowed. Special
    tokens have no interval and are masked in both modalities.
    """
    if (features.ndim != 2 or not len(features) or not features.shape[1]
            or not features.is_floating_point() or not torch.isfinite(features).all()
            or not math.isfinite(frame_rate) or frame_rate <= 0):
        raise ValueError("Expected finite acoustic frames and a positive frame rate")
    pools = []
    for span in spans:
        first = max(0, min(len(features), math.ceil(span["start"] * frame_rate - 1e-9)))
        last = max(first, min(len(features), math.floor(span["end"] * frame_rate + 1e-9) + 1))
        valid = span["end"] > span["start"] and first < last
        vector = features[first:last].mean(0) if valid else features.new_zeros(features.shape[1])
        pools.append((vector, valid, first, last))
    audio = features.new_zeros((len(offsets), features.shape[1]))
    audio_mask = torch.zeros(len(offsets), dtype=torch.bool)
    text_mask = torch.zeros(len(offsets), dtype=torch.bool)
    audit = []
    for token_id, (start, end) in enumerate(offsets):
        entry = {"position": token_id, "char_offsets": [start, end],
                 "kind": "special", "span_id": None, "audio_valid": False}
        if start == end == 0:
            audit.append(entry)
            continue
        if not 0 <= start < end:
            raise ValueError(f"Invalid text token offsets at position {token_id}")
        owners = [i for i, span in enumerate(spans)
                  if span["char_start"] <= start < end <= span["char_end"]]
        if len(owners) != 1:
            raise ValueError(f"Text token {token_id} does not belong to exactly one word/pause")
        span_id = owners[0]
        span = spans[span_id]
        vector, valid, first, last = pools[span_id]
        audio[token_id] = vector
        audio_mask[token_id] = valid
        text_mask[token_id] = True
        entry.update(kind=span["kind"], span_id=span_id, word_id=span["word_id"],
                     interval=[span["start"], span["end"]], frame_range=[first, last],
                     audio_valid=valid)
        audit.append(entry)
    if not audio_mask.any() or not text_mask.any():
        raise ValueError("Recording has no usable aligned audio/text tokens")
    return audio, audio_mask, text_mask, audit
