"""Server-rendered routes.

The upload endpoint is unauthenticated by design for this prototype: there are
no accounts and nothing is persisted from an upload. That is a deliberate
trade-off worth stating plainly - anyone who can reach the service can submit a
document, so it must not be exposed publicly without a rate limit in front of it.
Uploads are held in memory for the duration of the request and never written to
disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from app.adapters.extract import MedicationExtractor
from app.adapters.ocr import FixtureNotFoundError, OcrClient
from app.adapters.openfda import OpenFdaClient
from app.adapters.rxnorm import RxNormResolver
from app.config import Settings
from app.domain import guide as guide_module
from app.domain import pipeline
from app.domain.ingest import (
    EmptyUploadError,
    FileTooLargeError,
    IngestError,
    UnsupportedMediaTypeError,
)

_LOGGER = structlog.stdlib.get_logger("app.web.routes")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()

_HTTP_BAD_REQUEST = 400
_HTTP_NOT_FOUND = 404
_HTTP_PAYLOAD_TOO_LARGE = 413
_HTTP_UNSUPPORTED_MEDIA_TYPE = 415


@dataclass(frozen=True)
class Services:
    """Collaborators built once at startup and shared by every request."""

    settings: Settings
    ocr: OcrClient
    extractor: MedicationExtractor
    resolver: RxNormResolver
    openfda: OpenFdaClient


def get_services(request: Request) -> Services:
    """Return the application's services.

    Raises:
        RuntimeError: If the lifespan has not populated application state.
    """
    services = request.app.state.services
    if not isinstance(services, Services):
        # A RuntimeError, not a TypeError: the caller passed nothing wrong, the
        # application simply has not completed startup.
        raise RuntimeError("application services are not initialised")  # noqa: TRY004
    return services


def _sample_files(settings: Settings) -> list[dict[str, str]]:
    """List the committed sample prescriptions offered on the upload page."""
    return [
        {"stem": path.stem, "name": path.name, "label": path.stem.replace("_", " ")}
        for path in sorted(settings.sample_prescription_dir.iterdir())
        if path.is_file()
    ]


@router.get("/", response_class=HTMLResponse)
async def upload_form(request: Request) -> HTMLResponse:
    """Render the upload page."""
    services = get_services(request)
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={
            "samples": _sample_files(services.settings),
            "max_mb": services.settings.max_upload_bytes // (1024 * 1024),
        },
    )


@router.get("/samples/{stem}", response_model=None)
async def download_sample(request: Request, stem: str) -> FileResponse | HTMLResponse:
    """Serve a committed sample prescription for trying the pipeline.

    ``stem`` is matched against the known sample list rather than joined onto a
    path, so a traversal attempt cannot reach another file.
    """
    services = get_services(request)
    for candidate in sorted(services.settings.sample_prescription_dir.iterdir()):
        if candidate.is_file() and candidate.stem == stem:
            return FileResponse(candidate, filename=candidate.name)
    return _error(request, "That sample does not exist.", _HTTP_NOT_FOUND)


def _error(request: Request, message: str, status_code: int) -> HTMLResponse:
    """Render the upload page with an error message attached to the field."""
    services = get_services(request)
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={
            "samples": _sample_files(services.settings),
            "max_mb": services.settings.max_upload_bytes // (1024 * 1024),
            "error": message,
        },
        status_code=status_code,
    )


def _rejection_for(exc: Exception, settings: Settings) -> tuple[str, int]:
    """Return the patient-facing message and status for a rejected upload.

    Kept as a mapping rather than a chain of handlers so every rejection path is
    visible in one place and worded consistently.
    """
    limit_mb = settings.max_upload_bytes // (1024 * 1024)
    messages: dict[type[Exception], tuple[str, int]] = {
        EmptyUploadError: (
            "That file was empty. Choose a photo or PDF.",
            _HTTP_BAD_REQUEST,
        ),
        FileTooLargeError: (
            f"That file is larger than {limit_mb} MB.",
            _HTTP_PAYLOAD_TOO_LARGE,
        ),
        UnsupportedMediaTypeError: (
            "That file is not a PNG, JPEG or PDF. The contents are checked, not the file name.",
            _HTTP_UNSUPPORTED_MEDIA_TYPE,
        ),
        FixtureNotFoundError: (
            (
                "This build reads prescriptions from its committed samples only. "
                "Download a sample below and upload it unchanged."
            ),
            _HTTP_BAD_REQUEST,
        ),
    }
    return messages.get(type(exc), ("That document could not be read.", _HTTP_BAD_REQUEST))


@router.post("/guide", response_class=HTMLResponse)
async def create_guide(request: Request, prescription: UploadFile) -> HTMLResponse:
    """Run one uploaded document through the pipeline and render the result."""
    services = get_services(request)
    settings = services.settings

    payload = await prescription.read()
    try:
        result = await pipeline.run(
            payload,
            ocr=services.ocr,
            extractor=services.extractor,
            resolver=services.resolver,
            settings=settings,
        )
    except (IngestError, FixtureNotFoundError) as exc:
        message, status_code = _rejection_for(exc, settings)
        _LOGGER.info("upload_rejected", failure=type(exc).__name__, status_code=status_code)
        return _error(request, message, status_code)

    if result.guide is None:
        return templates.TemplateResponse(
            request=request,
            name="rejected.html",
            context={
                "report": result.gate_report,
                "mean_percent": round(result.gate_report.mean_confidence * 100),
                "threshold_percent": round(result.gate_report.mean_confidence_threshold * 100),
            },
        )

    composed = await guide_module.compose(result.guide, openfda=services.openfda)
    if isinstance(composed, guide_module.GuideRejected):
        return templates.TemplateResponse(
            request=request,
            name="failed.html",
            context={"rejection": composed},
        )

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={"guide": composed, "report": result.gate_report},
    )
