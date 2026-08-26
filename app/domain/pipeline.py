"""The end-to-end pipeline, assembled once and shared by every caller.

Having a single ``run`` means the web layer, the golden tests and the scripts all
exercise the same ordering of gates. A safety check added here cannot be missed
by one caller and applied by another.

Order is deliberate:

1. Validate the upload by content (:mod:`app.domain.ingest`).
2. Recognise text with the OCR client.
3. Apply the confidence gates. A rejected document stops here: no extraction is
   attempted on a scan we cannot read.
4. Extract candidate medication lines.
5. Verify each line against RxNorm and its own OCR spans
   (:mod:`app.safety.verifier`).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.adapters.extract import MedicationExtractor
from app.adapters.ocr import OcrClient
from app.adapters.rxnorm import RxNormResolver
from app.config import Settings
from app.domain.ingest import OcrDocument, ingest
from app.safety.gates import GateReport, evaluate_document
from app.safety.verifier import PatientGuide, verify_lines

_LOGGER = structlog.stdlib.get_logger("app.domain.pipeline")


@dataclass(frozen=True)
class PipelineResult:
    """Everything the interface needs to render one submitted document."""

    document: OcrDocument
    gate_report: GateReport
    guide: PatientGuide | None
    """``None`` when the confidence gates rejected the document."""

    @property
    def rejected(self) -> bool:
        """True when the document did not clear the confidence gates."""
        return self.guide is None


async def run(
    payload: bytes,
    *,
    ocr: OcrClient,
    extractor: MedicationExtractor,
    resolver: RxNormResolver,
    settings: Settings,
) -> PipelineResult:
    """Run one document through ingestion, gating, extraction and verification.

    Args:
        payload: Raw uploaded bytes.
        ocr: OCR client.
        extractor: Medication extractor.
        resolver: RxNorm resolver.
        settings: Thresholds and limits.

    Returns:
        The document, its gate report, and a guide unless the document was
        rejected.

    Raises:
        app.domain.ingest.IngestError: If the upload itself is unacceptable.
    """
    document = await ingest(payload, ocr=ocr, max_bytes=settings.max_upload_bytes)
    gate_report = evaluate_document(document, settings)
    if not gate_report.accepted:
        _LOGGER.info(
            "document_rejected",
            document_id=document.document_id,
            reason=gate_report.rejection_reason,
        )
        return PipelineResult(document=document, gate_report=gate_report, guide=None)

    lines = await extractor.extract(document)
    guide = await verify_lines(lines, resolver=resolver, gate_report=gate_report, settings=settings)
    return PipelineResult(document=document, gate_report=gate_report, guide=guide)
