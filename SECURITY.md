# Security

## Reporting

Open a private security advisory on the repository. Please do not open a public
issue for an unpatched vulnerability.

## What is enforced, and where it is tested

| Control | Where | Test |
|---|---|---|
| Content type from magic bytes, never the filename | `app/domain/ingest.py` | `test_ingest.py::test_spoofed_extension_is_rejected` |
| Request size refused **before** the body is buffered | `app/web/limits.py` | `test_limits.py::test_an_oversized_body_is_refused_with_413` |
| Per-IP rate limiting, tighter on the transcription endpoint | `app/web/ratelimit.py` | `test_limits.py::test_repeated_uploads_are_eventually_refused` |
| CSP, HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy | `app/web/middleware.py` | `test_security_headers.py` |
| Path traversal on the sample route | `app/web/routes.py` | matched against a known list, never path-joined |
| No PII in logs; route templates, not raw paths | `app/web/logging.py` | `test_logging.py::test_request_log_omits_raw_paths` |
| Credentials as `SecretStr`, redacted from reprs | `app/config.py` | `test_config.py::test_database_url_is_not_printable` |
| Parameterised SQL throughout | `app/adapters/*` | psycopg placeholders only; no string interpolation |
| Dependency audit | CI | `pip-audit --strict` |
| Output escaping | Jinja2 autoescape | `test_limits.py::test_refusals_never_echo_the_request` |

## Threat model

Uploads are held in memory for one request and never written to disk. Postgres
holds only public reference data — RxNorm concepts and cached openFDA labels — so
the default deployment stores nothing about a patient. Prescription history is a
separate, opt-in MongoDB store that is disabled unless a URI is configured.

## Known gaps

Stated rather than hidden.

- **`POST /guide` is unauthenticated.** Rate limiting is the only control in front
  of it. Do not expose it publicly without also putting a WAF or auth in front.
- **Rate-limit counters are in-process.** On a horizontally scaled deployment the
  effective ceiling is the limit times the instance count. A shared Redis backend
  is required to make it global.
- **No CSRF tokens.** Not currently exploitable because no cookie or session
  exists. This must be added in the same change that introduces sign-in.
- **Prescription history is not wired to any route.** Keying it on a
  user-supplied profile name would be an IDOR; it waits for authenticated
  subjects.
- **Vision transcription confidence is self-reported** by the model, not measured.
  It is labelled as such, and words the model flags are scored below the gate.
