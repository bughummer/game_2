"""Drop LLM sentences that cite a number not present in the deterministic context (Req 11.5, D14).

The reasoning service hands the governed context (built only from deterministic artifacts) and
the model's free-text narrative to :func:`drop_unverified_numbers`. Every sentence whose decimal
or percentage tokens do not all match a number in the context — at the precision the sentence
states — is removed before storage. Sentences with no numbers are always kept, so prose that
merely describes without quoting survives.

The filter is deliberately conservative and local: it looks only at the numeric tokens in a
sentence, never at wording, and it never rewrites a sentence. It applies in all modes (D14);
single-split narratives are filtered the same way.

Matching rule
-------------
A numeric token matches a context number when the context number, rounded to the number of
decimal places the token wrote, equals the token's value. A percentage token like ``12%`` or
``12.5%`` is accepted if *either* reading matches: the written value (``12``, ``12.5``) or the
fraction (``0.12``, ``0.125``). Rounding uses the token's own decimal count, so ``0.80`` matches
a context ``0.7987`` (rounds to ``0.80``) but ``0.81`` does not.
"""

from __future__ import annotations

import re
from typing import Any

# A decimal or integer, optionally followed by a percent sign. Captures the digits and the sign
# so the caller can tell "12%" from "12". Leading sign is kept out so "a-b" does not read "-b".
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")

# Sentence boundaries: a run of ., ! or ? followed by whitespace. Newlines also split, so a
# bulleted or line-broken narrative is filtered line by line.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _collect_numbers(value: Any, out: list[float]) -> None:
    """Gather every numeric leaf in the context (dicts, lists, numbers)."""
    if isinstance(value, bool):
        return  # booleans are not numeric citations
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_numbers(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_numbers(item, out)


def _decimals(token: str) -> int:
    """How many digits the token wrote after the decimal point (0 for an integer)."""
    _, _, frac = token.partition(".")
    return len(frac)


def _matches(token_value: float, decimals: int, numbers: list[float]) -> bool:
    """True when some context number rounds (at ``decimals`` places) to ``token_value``."""
    rounded_token = round(token_value, decimals)
    return any(round(number, decimals) == rounded_token for number in numbers)


def _token_is_grounded(digits: str, percent: bool, numbers: list[float]) -> bool:
    decimals = _decimals(digits)
    written = float(digits)
    if _matches(written, decimals, numbers):
        return True
    if percent:
        # "12%" may mean the fraction 0.12 in the context; accept either reading. The fraction
        # carries two more decimal places than the written percentage.
        fraction = written / 100.0
        if _matches(fraction, decimals + 2, numbers):
            return True
    return False


def _sentence_is_grounded(sentence: str, numbers: list[float]) -> bool:
    tokens = _NUMBER.findall(sentence)
    if not tokens:
        return True  # no number to verify
    return all(_token_is_grounded(digits, bool(sign), numbers) for digits, sign in tokens)


def drop_unverified_numbers(text: str, context: dict[str, Any]) -> str:
    """Return ``text`` with every number-citing sentence that the context cannot ground removed.

    Sentences are kept in order. A sentence survives iff every decimal or percentage token in it
    matches a number in ``context`` at the sentence's stated precision; sentences with no numbers
    always survive. The separators between kept sentences are preserved so the prose still reads
    naturally.
    """
    if not text:
        return text
    numbers: list[float] = []
    _collect_numbers(context, numbers)

    # Split while keeping the separators, so we can rejoin kept sentences with their original
    # spacing/punctuation.
    pieces = _SENTENCE_SPLIT.split(text)
    separators = _SENTENCE_SPLIT.findall(text)

    kept: list[str] = []
    kept_separators: list[str] = []
    for index, piece in enumerate(pieces):
        if piece.strip() == "":
            continue
        if _sentence_is_grounded(piece, numbers):
            kept.append(piece)
            # The separator that *follows* this piece, when there is one.
            if index < len(separators):
                kept_separators.append(separators[index])
            else:
                kept_separators.append("")

    if not kept:
        return ""
    out: list[str] = []
    for index, sentence in enumerate(kept):
        out.append(sentence)
        if index < len(kept) - 1:
            out.append(kept_separators[index] or " ")
    return "".join(out)
