"""Match text-token character offsets to timestamped transcript words."""

import math
import unicodedata


def token_audio_intervals(transcription, words, offsets, audio_duration=None):
    """Return one (start, end) interval per token; special tokens get None.

    Word text is matched against the original transcript, so tokenizer lowercasing
    and accent removal do not affect alignment. Sentence punctuation uses the
    gap between neighboring spoken words, as in the reference preprocessing.
    """
    spans = []
    lexical_spans = []
    cursor = 0
    for word, start, end in words:
        word = unicodedata.normalize("NFC", str(word)).strip()
        if not word:
            raise ValueError("Empty word in timestamp data")
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError(f"Invalid timestamps for {word!r}: {start}, {end}")
        char_start = transcription.find(word, cursor)
        if char_start < 0 or any(c.isalnum() for c in transcription[cursor:char_start]):
            raise ValueError(f"Timestamp word {word!r} does not match transcript at character {cursor}")
        char_end = char_start + len(word)
        span = (char_start, char_end, start, end)
        spans.append(span)
        letters = [i for i, char in enumerate(word) if char.isalnum()]
        if letters:
            lexical_spans.append((char_start + letters[0], char_start + letters[-1] + 1, start, end))
        cursor = char_end
    if any(c.isalnum() for c in transcription[cursor:]):
        raise ValueError("Transcript contains words missing from timestamp data")

    intervals = []
    for char_start, char_end in offsets:
        if char_start == char_end == 0:
            intervals.append(None)
            continue
        if not 0 <= char_start < char_end <= len(transcription):
            raise ValueError(f"Invalid token offsets: {char_start}, {char_end}")
        token_text = transcription[char_start:char_end]
        owners = [s for s in spans if s[0] < char_end and s[1] > char_start]
        punctuation = token_text.strip() and all(c in '.,;:!?…¿¡' for c in token_text.strip())
        inside_word = any(s[0] <= char_start and char_end <= s[1] for s in lexical_spans)
        if punctuation and not inside_word:
            previous = [s for s in lexical_spans if s[1] <= char_start]
            following = [s for s in lexical_spans if s[0] >= char_end]
            start = previous[-1][3] if previous else 0.0
            end = following[0][2] if following else (audio_duration if audio_duration is not None else start)
            # Whisper word intervals can overlap; represent this as a zero gap.
            intervals.append((start, max(start, end)))
        elif len(owners) == 1:
            intervals.append((owners[0][2], owners[0][3]))
        else:
            raise ValueError(f"Token {token_text!r} does not map to exactly one timestamped word")
    return intervals


def frame_bounds(start, end, frames_per_second, frame_count):
    """Bound a word/gap interval and expand very short intervals for pooling."""
    if frame_count <= 0 or frames_per_second <= 0:
        raise ValueError("Audio features must contain frames at a positive frame rate")
    first = min(frame_count - 1, max(0, math.floor(start * frames_per_second)))
    last = min(frame_count, max(first, math.ceil(end * frames_per_second)))
    if last - first < 3:
        first = max(0, first - 2)
        last = min(frame_count, last + 2)
    return first, last
