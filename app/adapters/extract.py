"""Medication extraction behind a Protocol, with a deterministic implementation.

``MedicationExtractor`` is the seam a structured-output language model plugs
into. The implementation shipped here, :class:`RuleBasedExtractor`, is not a
stand-in for one: it is the safer of the two options and is what runs.

It can only *select* token ranges, never emit characters. Every string it
produces is the verbatim text of tokens it names in a span, so an extraction can
always be replayed against the OCR document and checked character for character.
A generative extractor cannot offer that, which is why the downstream verifier
re-checks every field against its span regardless of which extractor produced it.

No cloud language-model implementation is included: this deployment has no
credentials for one, and an adapter that cannot be executed has no place on the
safety path.

Layout assumptions
------------------
Tokens are grouped into visual lines by vertical overlap. A line is treated as a
*drug line* when it contains a strength (a number immediately followed by a unit)
and begins with a word. The text before the strength is the drug name; the
strength is the dose. A following *sig line* - one opening with a known
administration verb - supplies the frequency, matched as the longest known
frequency phrase plus an optional trailing modifier.

A line with no strength is never guessed at. It yields no medication, which
surfaces as nothing rather than as an invented dose.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.domain.ingest import OcrDocument, OcrPage, OcrToken, span_from_tokens
from app.domain.medication import MedicationLine

# A strength value: 500, 0.5, 20-25, 1/2.
_NUMBER = re.compile(r"^\d+(?:[./-]\d+)*$")

_DOSE_UNITS = frozenset(
    {"mg", "mcg", "ug", "g", "gm", "ml", "l", "unit", "units", "iu", "meq", "mmol"}
)

# Words that open an administration instruction.
_SIG_OPENERS = frozenset(
    {"take", "apply", "inject", "use", "instill", "inhale", "place", "insert", "swallow"}
)

# Frequency phrases, matched longest-first. Recognising a phrase only locates a
# token range; the text returned is always the verbatim tokens.
_FREQUENCY_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("three", "times", "daily"),
    ("four", "times", "daily"),
    ("two", "times", "daily"),
    ("every", "#", "hours"),
    ("every", "#", "hrs"),
    ("every", "other", "day"),
    ("once", "daily"),
    ("twice", "daily"),
    ("once", "a", "day"),
    ("twice", "a", "day"),
    ("every", "morning"),
    ("every", "evening"),
    ("every", "night"),
    ("at", "bedtime"),
    ("daily",),
    ("nightly",),
    ("bid",),
    ("tid",),
    ("qid",),
    ("qhs",),
    ("qam",),
    ("prn",),
)

# Optional trailing qualifiers appended to a matched frequency phrase.
_FREQUENCY_MODIFIERS: tuple[tuple[str, ...], ...] = (
    ("as", "needed"),
    ("with", "meals"),
    ("with", "food"),
    ("before", "meals"),
    ("after", "meals"),
    ("on", "an", "empty", "stomach"),
)


@runtime_checkable
class MedicationExtractor(Protocol):
    """Finds candidate medication lines in an OCR document."""

    async def extract(self, document: OcrDocument) -> tuple[MedicationLine, ...]:
        """Return candidate medications, each carrying its source spans."""
        ...


def _is_number(text: str) -> bool:
    """True when the token looks like a strength value."""
    return _NUMBER.match(text) is not None


def group_lines(page: OcrPage) -> tuple[tuple[OcrToken, ...], ...]:
    """Group a page's tokens into visual lines.

    Tokens are ordered top-to-bottom then left-to-right, and a new line starts
    when a token's vertical midpoint leaves the current line's band. The band is
    half the current line's height, which tolerates the baseline jitter typical
    of a scan without merging adjacent lines.

    Args:
        page: The page to group.

    Returns:
        Lines of tokens, in reading order.
    """
    if not page.tokens:
        return ()

    ordered = sorted(page.tokens, key=lambda token: (token.box.top, token.box.left))
    lines: list[list[OcrToken]] = [[ordered[0]]]
    for token in ordered[1:]:
        current = lines[-1]
        reference = current[0].box
        band = (reference.bottom - reference.top) / 2
        midpoint = (token.box.top + token.box.bottom) / 2
        if abs(midpoint - (reference.top + reference.bottom) / 2) <= band:
            current.append(token)
        else:
            lines.append([token])

    return tuple(tuple(sorted(line, key=lambda token: token.box.left)) for line in lines)


def _find_strength(tokens: Sequence[OcrToken]) -> tuple[int, int] | None:
    """Return the ``(start, end)`` slice of a number-then-unit pair, if any."""
    for index in range(len(tokens) - 1):
        if _is_number(tokens[index].text) and tokens[index + 1].text.casefold() in _DOSE_UNITS:
            return index, index + 2
    return None


def _match_at(tokens: Sequence[OcrToken], start: int, pattern: tuple[str, ...]) -> int | None:
    """Return the end index if ``pattern`` matches at ``start``, else ``None``."""
    if start + len(pattern) > len(tokens):
        return None
    for offset, expected in enumerate(pattern):
        actual = tokens[start + offset].text.casefold().strip(".,;:")
        if expected == "#":
            if not _is_number(actual):
                return None
        elif actual != expected:
            return None
    return start + len(pattern)


def _find_frequency(tokens: Sequence[OcrToken]) -> tuple[int, int] | None:
    """Return the ``(start, end)`` slice of the frequency phrase on a sig line."""
    for start in range(len(tokens)):
        for pattern in _FREQUENCY_PATTERNS:
            end = _match_at(tokens, start, pattern)
            if end is None:
                continue
            for modifier in _FREQUENCY_MODIFIERS:
                extended = _match_at(tokens, end, modifier)
                if extended is not None:
                    return start, extended
            return start, end
    return None


def _is_sig_line(tokens: Sequence[OcrToken]) -> bool:
    """True when the line opens with an administration verb."""
    return bool(tokens) and tokens[0].text.casefold().strip(".,;:") in _SIG_OPENERS


class RuleBasedExtractor:
    """Selects medication fields from token ranges. Emits no new text."""

    async def extract(self, document: OcrDocument) -> tuple[MedicationLine, ...]:
        """Extract every medication line the layout rules can justify.

        Args:
            document: The normalised OCR document.

        Returns:
            Candidate medication lines in reading order. Empty when no line
            carries a strength.
        """
        results: list[MedicationLine] = []
        for page in document.pages:
            lines = group_lines(page)
            for index, tokens in enumerate(lines):
                if _is_sig_line(tokens):
                    continue
                strength = _find_strength(tokens)
                if strength is None:
                    continue
                dose_start, dose_end = strength
                drug_tokens = tokens[:dose_start]
                if not drug_tokens:
                    continue

                following = lines[index + 1] if index + 1 < len(lines) else ()
                frequency_slice = _find_frequency(following) if _is_sig_line(following) else None

                page_number = page.page_number
                dose_tokens = tokens[dose_start:dose_end]
                frequency_tokens = (
                    following[frequency_slice[0] : frequency_slice[1]]
                    if frequency_slice is not None
                    else ()
                )

                drug_span = span_from_tokens(page_number, drug_tokens)
                dose_span = span_from_tokens(page_number, dose_tokens)
                frequency_span = (
                    span_from_tokens(page_number, frequency_tokens) if frequency_tokens else None
                )
                covered = (*tokens, *frequency_tokens)

                results.append(
                    MedicationLine(
                        page_number=page_number,
                        drug_text=drug_span.text,
                        drug_span=drug_span,
                        dose_text=dose_span.text,
                        dose_span=dose_span,
                        frequency_text=frequency_span.text if frequency_span else None,
                        frequency_span=frequency_span,
                        line_span=span_from_tokens(page_number, covered),
                    )
                )
        return tuple(results)
