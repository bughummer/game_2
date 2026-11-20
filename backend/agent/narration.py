"""Narration verifier (design C9). Pure: no I/O, no model, no AWS.

``verify_narration`` checks every number in the model's text against a ``NumberLedger`` built
from the session's stored tool results, with the same matching rule as
``ml_engine/reasoning/numbers.py`` (Req 10.2). ``build_final_answer`` adds attributions,
free-form metric labels, verbatim blocks, "not comparable" notes and the *Advisory* label
(Req 10.4 to 10.8).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ml_engine.contracts.agent import (
    AnswerSentence,
    Attribution,
    FinalAnswer,
    LedgerEntry,
    VerbatimBlock,
)

UNVERIFIED_MARKER = "[unverified] "
OVERCLAIM_MARKER = "[overclaim] "
ADVISORY_LABEL = "Advisory"
METRIC_LABELS = ("self_reported", "engine")
#: Attributions kept per numeric token (a common value like ``1`` can match many entries).
MAX_ATTRIBUTIONS_PER_TOKEN = 5

# Same patterns as ml_engine/reasoning/numbers.py.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")

_ADVISORY_TERM = re.compile(
    r"\b(?:leak\w*|propos\w*|candidate features?|rank\w*|best model|top model|winner)\b",
    re.IGNORECASE,
)
_CERTAINTY = re.compile(
    r"\b(?:guarantee\w*|certain(?:ly)?|prove[nd]?|definite(?:ly)?)\b", re.IGNORECASE
)
_RANKING = re.compile(
    r"\b(?:better|worse|beats?|outperform\w*|higher|lower|best|worst|ranks?|ranked|"
    r"stronger|weaker|wins?|superior|inferior)\b",
    re.IGNORECASE,
)

Policy = Literal["mark", "withhold"]


class NumberLedger:
    """Engine numbers a narration may cite. Same matching rule as ml_engine numbers.py.

    ``numbers.py`` keeps its rule in private helpers over bare floats, so the rule is mirrored
    here over ledger entries. ``tests/agent/test_narration.py`` checks that both agree.
    """

    def __init__(self, entries: Iterable[LedgerEntry] = ()) -> None:
        self._entries: list[LedgerEntry] = []
        self.add(entries)

    def add(self, entries: Iterable[LedgerEntry]) -> None:
        self._entries.extend(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def match(self, token: str, percent: bool) -> list[LedgerEntry]:
        """Entries equal to ``token`` at the token's own decimals (or its fraction reading)."""
        try:
            written = float(token)
        except ValueError:
            return []
        decimals = len(token.partition(".")[2])
        found = [e for e in self._entries if round(e.value, decimals) == round(written, decimals)]
        if percent:
            fraction = round(written / 100.0, decimals + 2)
            found += [
                e
                for e in self._entries
                if e not in found and round(e.value, decimals + 2) == fraction
            ]
        return found


@dataclass(frozen=True, slots=True)
class VerifiedSentence:
    text: str
    status: Literal["verified", "unverified", "overclaim"]
    unmatched: tuple[str, ...] = ()
    #: ``(token, matched entries)`` for every numeric token, in sentence order.
    matches: tuple[tuple[str, tuple[LedgerEntry, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class VerifiedNarration:
    sentences: tuple[VerifiedSentence, ...] = ()
    policy: Policy = "mark"
    withheld: tuple[VerifiedSentence, ...] = field(default=())

    @property
    def unverified(self) -> tuple[VerifiedSentence, ...]:
        """Sentences needing one ``narration_unverified`` audit record each (Req 10.3)."""
        return tuple(s for s in (*self.sentences, *self.withheld) if s.status != "verified")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One stored full tool result the final answer may draw on."""

    tool_call_id: str
    tool: str
    turn: int
    full: dict[str, Any]
    verbatim: bool = False
    advisory: bool = False


def verify_narration(
    text: str, ledger: NumberLedger, *, policy: Policy = "mark"
) -> VerifiedNarration:
    """A sentence is verified iff every numeric token in it matches the ledger (Req 10.2, 10.3)."""
    kept: list[VerifiedSentence] = []
    withheld: list[VerifiedSentence] = []
    for piece in _SENTENCE_SPLIT.split(text or ""):
        sentence = piece.strip()
        if not sentence:
            continue
        matches: list[tuple[str, tuple[LedgerEntry, ...]]] = []
        unmatched: list[str] = []
        for digits, sign in _NUMBER.findall(sentence):
            token = f"{digits}{sign}"
            found = tuple(ledger.match(digits, bool(sign)))
            matches.append((token, found))
            if not found:
                unmatched.append(token)
        if not unmatched:
            kept.append(VerifiedSentence(sentence, "verified", matches=tuple(matches)))
            continue
        item = VerifiedSentence(sentence, "unverified", tuple(unmatched), tuple(matches))
        (withheld if policy == "withhold" else kept).append(item)
    return VerifiedNarration(tuple(kept), policy, tuple(withheld))


# ----------------------------------------------------------------------------- final answer


def _labels_of(entries: Iterable[LedgerEntry]) -> list[str]:
    return sorted({e.metric_label for e in entries if e.metric_label in METRIC_LABELS})


def _label_tokens(text: str, matches: Sequence[tuple[str, tuple[LedgerEntry, ...]]]) -> str:
    """Put the free-form metric label right after each number that carries one (Req 10.5)."""
    pieces: list[str] = []
    cursor = 0
    tokens = iter(matches)
    for m in _NUMBER.finditer(text):
        _token, entries = next(tokens, ("", ()))
        labels = _labels_of(entries)
        pieces.append(text[cursor : m.end()])
        cursor = m.end()
        if labels:
            suffix = f" ({'/'.join(labels)})"
            if not text.startswith(suffix, cursor):
                pieces.append(suffix)
    pieces.append(text[cursor:])
    return "".join(pieces)


def _attributions(
    matches: Sequence[tuple[str, tuple[LedgerEntry, ...]]],
) -> list[Attribution]:
    out: list[Attribution] = []
    for token, entries in matches:
        for e in entries[:MAX_ATTRIBUTIONS_PER_TOKEN]:
            label = e.metric_label if e.metric_label in METRIC_LABELS else None
            out.append(
                Attribution(
                    tool_call_id=e.tool_call_id,
                    experiment_id=e.experiment_id,
                    artifact_ref=None,
                    path=e.path,
                    value_text=token,
                    metric_label=label,  # type: ignore[arg-type]
                )
            )
    return out


def _not_comparable(full: Any) -> list[dict[str, Any]]:
    """Every dict entry in ``full`` that carries ``comparable: false``."""
    found: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("comparable") is False:
                found.append(node)
            for item in node.values():
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(full)
    return found


def _entry_names(entry: dict[str, Any]) -> list[str]:
    return [str(entry[k]) for k in ("name", "run_id") if isinstance(entry.get(k), str) and entry[k]]


def build_final_answer(
    narration: VerifiedNarration,
    results: Sequence[ToolResult] = (),
    **_: Any,
) -> FinalAnswer:
    """Build the answer the user sees from verified narration and the stored full results."""
    cited = {
        e.tool_call_id
        for s in narration.sentences
        if s.status == "verified"
        for _token, entries in s.matches
        for e in entries
    }
    last_turn = max((r.turn for r in results), default=None)
    used = [r for r in results if r.tool_call_id in cited or r.turn == last_turn]

    not_comparable: list[dict[str, Any]] = []
    for r in used:
        not_comparable.extend(_not_comparable(r.full))
    blocked = {n for entry in not_comparable for n in _entry_names(entry)}

    sentences: list[AnswerSentence] = []
    dropped_rankings = 0
    for s in narration.sentences:
        if blocked and _RANKING.search(s.text) and any(n in s.text for n in blocked):
            dropped_rankings += 1  # ranks an entry the comparison marks not comparable
            continue
        overclaim = bool(_ADVISORY_TERM.search(s.text) and _CERTAINTY.search(s.text))
        if s.status == "verified":
            text = _label_tokens(s.text, s.matches)
            status = "overclaim" if overclaim else "verified"
            prefix = OVERCLAIM_MARKER if overclaim else ""
            attributions = _attributions(s.matches)
        else:
            text = s.text
            status = "unverified"
            prefix = UNVERIFIED_MARKER + (OVERCLAIM_MARKER if overclaim else "")
            attributions = []
        sentences.append(
            AnswerSentence(text=prefix + text, status=status, attributions=attributions)
        )

    notes: list[str] = []
    if any(s.status == "unverified" for s in narration.sentences):
        notes.append("Numbers in this answer could not be verified against tool results.")
    if narration.withheld:
        notes.append(
            f"{len(narration.withheld)} sentence(s) were withheld because their numbers could "
            "not be verified against tool results."
        )
    seen: set[str] = set()
    for entry in not_comparable:
        names = _entry_names(entry)
        name = names[0] if names else "an entry"
        if name in seen:
            continue
        seen.add(name)
        label = entry.get("metric_label")
        why = f" ({label} metric)" if isinstance(label, str) else ""
        notes.append(
            f"{name}{why} is not comparable with the other entries and is not ranked against them."
        )
    if dropped_rankings:
        notes.append(
            f"{dropped_rankings} sentence(s) that ranked entries which are not comparable were "
            "removed."
        )
    overclaimed = any(OVERCLAIM_MARKER in s.text for s in sentences)
    if overclaimed:
        notes.append(
            "Leakage findings, feature proposals and model rankings are advisory, not guaranteed."
        )

    verbatim = [
        VerbatimBlock(tool_call_id=r.tool_call_id, title=r.tool, content=copy.deepcopy(r.full))
        for r in used
        if r.verbatim
    ]
    labels = [ADVISORY_LABEL] if overclaimed or any(r.advisory for r in used) else []
    return FinalAnswer(sentences=sentences, verbatim=verbatim, notes=notes, labels=labels)
