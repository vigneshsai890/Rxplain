"""Tests for the NVIDIA vision transcription adapter.

No network. The transport is exercised through the response parser, and the
tokeniser is tested directly, so the safety-relevant behaviour - unsure words
being flagged, shorthand being left alone - is covered without a live model.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from app.adapters.ocr import OcrClient
from app.adapters.vision import (
    TRANSCRIPTION_PROMPT,
    NotAPrescriptionError,
    NvidiaVisionOcr,
    VisionTranscriptionError,
    _extract_content,
    _tokenise,
)
from app.config import Settings
from app.domain.ingest import UploadedFile

PAYLOAD = b"\x89PNG\r\n\x1a\n" + b"fixture"


def _settings(**overrides: object) -> Settings:
    """Build settings with a dummy key so the adapter can be constructed."""
    return Settings(app_env="test", nvidia_api_key="nvapi-test-not-a-real-key", **overrides)  # type: ignore[arg-type]


def _uploaded() -> UploadedFile:
    """Build an uploaded file matching ``PAYLOAD``."""
    return UploadedFile(
        document_id=hashlib.sha256(PAYLOAD).hexdigest(),
        media_type="image/png",
        size_bytes=len(PAYLOAD),
    )


def _completion(content: str) -> str:
    """Build a chat-completions response body."""
    return json.dumps({"choices": [{"message": {"content": content}}]})


# --------------------------------------------------------------------------- #
# construction and the Protocol seam
# --------------------------------------------------------------------------- #
def test_the_adapter_satisfies_the_ocr_protocol() -> None:
    """It drops into the same seam as the fixture replayer."""
    assert isinstance(NvidiaVisionOcr(_settings()), OcrClient)


def test_construction_requires_a_key() -> None:
    """Without a key the app must fall back, not build a broken client."""
    with pytest.raises(ValueError, match="nvidia_api_key"):
        NvidiaVisionOcr(Settings(app_env="test"))


def test_no_key_is_configured_by_default() -> None:
    """The offline fixture path stays the default."""
    assert Settings(app_env="test").nvidia_api_key is None


# --------------------------------------------------------------------------- #
# the transcription prompt
# --------------------------------------------------------------------------- #
def test_the_prompt_forbids_expanding_shorthand() -> None:
    """Expanding OD or HS is an interpretation, so the model is told not to."""
    lowered = TRANSCRIPTION_PROMPT.lower()
    assert "do not expand abbreviations" in lowered
    for shorthand in ("od", "bd", "tds", "hs", "pc", "ac", "sos", "prn"):
        assert shorthand.upper() in TRANSCRIPTION_PROMPT


def test_the_prompt_forbids_correcting_and_adding() -> None:
    """The model transcribes; it does not improve the prescription."""
    lowered = TRANSCRIPTION_PROMPT.lower()
    assert "do not correct spelling" in lowered
    assert "not visible in the image" in lowered


def test_the_prompt_requests_word_level_uncertainty() -> None:
    """Word-level marking is what drives the per-token gate."""
    assert "??word??" in TRANSCRIPTION_PROMPT


# --------------------------------------------------------------------------- #
# tokenisation
# --------------------------------------------------------------------------- #
def test_words_become_tokens_in_reading_order() -> None:
    """Token ids run in reading order across lines."""
    tokens = _tokenise(["METFORMIN 500 MG", "Take 1 tablet"], confident=0.9, unsure=0.45)
    assert [token.text for token in tokens] == [
        "METFORMIN",
        "500",
        "MG",
        "Take",
        "1",
        "tablet",
    ]
    assert [token.token_id for token in tokens][:2] == ["p1-t001", "p1-t002"]


def test_unsure_markers_are_stripped_and_scored_low() -> None:
    """A marked word keeps its text but is scored below the token gate."""
    tokens = _tokenise(["METFORMIN ??500?? MG"], confident=0.9, unsure=0.45)
    assert [token.text for token in tokens] == ["METFORMIN", "500", "MG"]
    assert [token.confidence for token in tokens] == [0.9, 0.45, 0.9]


def test_the_unsure_score_is_below_the_token_gate() -> None:
    """The whole mechanism depends on this inequality holding."""
    settings = Settings(app_env="test")
    assert settings.vision_unsure_token_confidence < settings.ocr_min_token_confidence


def test_blank_lines_are_dropped() -> None:
    """Empty lines contribute no tokens rather than empty ones."""
    assert _tokenise(["", "   ", "METFORMIN"], confident=0.9, unsure=0.45) != ()
    assert len(_tokenise(["", "METFORMIN"], confident=0.9, unsure=0.45)) == 1


def test_boxes_are_well_formed_and_ordered_down_the_page() -> None:
    """Spans need valid boxes, and later lines must sit lower."""
    tokens = _tokenise(["line one here", "line two here", "line three"], confident=0.9, unsure=0.4)
    for token in tokens:
        assert 0.0 <= token.box.left < token.box.right <= 1.0
        assert 0.0 <= token.box.top < token.box.bottom <= 1.0
    assert tokens[0].box.top < tokens[-1].box.top


def test_many_lines_stay_within_the_page() -> None:
    """A long prescription must not produce out-of-range boxes."""
    tokens = _tokenise([f"line {index} of text" for index in range(60)], confident=0.9, unsure=0.4)
    assert all(token.box.bottom <= 1.0 for token in tokens)


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #
def test_content_is_extracted_from_a_valid_response() -> None:
    """The happy path returns the assistant message."""
    assert _extract_content(_completion("METFORMIN 500 MG")) == "METFORMIN 500 MG"


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "[]",
        "{}",
        '{"choices": []}',
        '{"choices": [null]}',
        '{"choices": [{}]}',
        '{"choices": [{"message": {}}]}',
        '{"choices": [{"message": {"content": ""}}]}',
        '{"choices": [{"message": {"content": "   "}}]}',
        '{"choices": [{"message": {"content": 42}}]}',
    ],
)
def test_a_malformed_response_is_rejected(raw: str) -> None:
    """A response that is not shaped as expected fails loudly."""
    with pytest.raises(VisionTranscriptionError):
        _extract_content(raw)


# --------------------------------------------------------------------------- #
# end-to-end through a stubbed transport
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_a_transcription_becomes_a_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter produces a document keyed to the uploaded bytes."""
    adapter = NvidiaVisionOcr(_settings())

    async def _fake(self: NvidiaVisionOcr, media_type: str, payload: bytes) -> str:
        del self, media_type, payload
        return "METFORMIN HCL 500 MG TABLET\nTake 1 tablet by mouth twice daily"

    monkeypatch.setattr(NvidiaVisionOcr, "_transcribe", _fake)
    document = await adapter.analyze(_uploaded(), PAYLOAD)
    assert document.document_id == hashlib.sha256(PAYLOAD).hexdigest()
    assert document.media_type == "image/png"
    assert len(document.pages) == 1
    assert [token.text for token in document.tokens][:3] == ["METFORMIN", "HCL", "500"]


@pytest.mark.anyio
async def test_shorthand_survives_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorthand reaches the document unexpanded, as the prompt demands."""
    adapter = NvidiaVisionOcr(_settings())

    async def _fake(self: NvidiaVisionOcr, media_type: str, payload: bytes) -> str:
        del self, media_type, payload
        return "AMOXICILLIN 500 MG\nSig: 1 cap TDS PC"

    monkeypatch.setattr(NvidiaVisionOcr, "_transcribe", _fake)
    document = await adapter.analyze(_uploaded(), PAYLOAD)
    texts = [token.text for token in document.tokens]
    assert "TDS" in texts
    assert "PC" in texts


@pytest.mark.anyio
async def test_no_text_found_is_reported_as_not_a_prescription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A photo of something else is refused, not transcribed into nothing."""
    adapter = NvidiaVisionOcr(_settings())

    async def _fake(self: NvidiaVisionOcr, media_type: str, payload: bytes) -> str:
        del self, media_type, payload
        return "NO_TEXT_FOUND"

    monkeypatch.setattr(NvidiaVisionOcr, "_transcribe", _fake)
    with pytest.raises(NotAPrescriptionError):
        await adapter.analyze(_uploaded(), PAYLOAD)


@pytest.mark.anyio
async def test_a_whitespace_only_transcription_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty transcription must not become an empty guide."""
    adapter = NvidiaVisionOcr(_settings())

    async def _fake(self: NvidiaVisionOcr, media_type: str, payload: bytes) -> str:
        del self, media_type, payload
        return "\n   \n\n"

    monkeypatch.setattr(NvidiaVisionOcr, "_transcribe", _fake)
    with pytest.raises(VisionTranscriptionError):
        await adapter.analyze(_uploaded(), PAYLOAD)
