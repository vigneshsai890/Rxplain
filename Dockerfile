# syntax=docker/dockerfile:1

# --------------------------------------------------------------------------- #
# Build stage: install pinned dependencies into a self-contained virtualenv.
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /src
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pyproject.toml is the single source of dependency pins; there is no second
# requirements file to drift from it.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

# --------------------------------------------------------------------------- #
# Runtime stage: slim base, no compilers or pip cache, unprivileged user.
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

RUN groupadd --system --gid 10001 rxplain \
 && useradd --system --uid 10001 --gid 10001 \
        --no-create-home --shell /usr/sbin/nologin rxplain

COPY --from=builder /opt/venv /opt/venv

# /srv is empty on purpose: keeping the working directory clear of a directory
# named `app` removes any chance of shadowing the installed package.
WORKDIR /srv
USER 10001:10001
EXPOSE 8080

# $PORT is supplied by the platform (Cloud Run injects it); 0.0.0.0 is required
# for the container to be reachable. `exec` keeps uvicorn as PID 1 so SIGTERM
# reaches it directly. The access log is disabled in app/web/logging.py rather
# than with a CLI flag, so the guarantee survives any entrypoint.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT}\""]
