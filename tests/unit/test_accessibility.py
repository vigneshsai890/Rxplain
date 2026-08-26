"""WCAG 2.1 AA checks on the rendered pages and the stylesheet.

These are the same structural rules an automated auditor applies to static
markup - landmarks, programmatic labels, live regions, language, heading order,
and text contrast. They run in-process, so no browser is needed and they cannot
be skipped in an environment without one.

Contrast is computed from the stylesheet itself, using the WCAG relative
luminance formula, so a colour cannot be changed without the ratio being
rechecked.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from itertools import pairwise
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

STYLESHEET = Path(__file__).resolve().parents[2] / "app" / "web" / "static" / "styles.css"

MINIMUM_CONTRAST = 4.5

_HEX = re.compile(r"^#([0-9a-fA-F]{6})$")
_VAR_DECLARATION = re.compile(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6});")
_CONTRAST_PAIR = re.compile(r"/\*\s*contrast-pair:\s*--([a-z-]+)\s+on\s+--([a-z-]+)\s*\*/")


class _Collector(HTMLParser):
    """Gathers the structural facts the assertions below need."""

    def __init__(self) -> None:
        """Initialise empty collections."""
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.headings: list[int] = []
        self.labels_for: list[str] = []
        self.input_ids: list[str] = []
        self.live_regions: list[str] = []
        self.html_lang: str | None = None
        self.images_without_alt = 0
        self.roles: list[str] = []
        self.labelled_by: list[str] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record one element."""
        attributes = {name: (value or "") for name, value in attrs}
        self.tags.append(tag)
        if identifier := attributes.get("id"):
            self.ids.append(identifier)
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.headings.append(int(tag[1]))
        if tag == "label" and "for" in attributes:
            self.labels_for.append(attributes["for"])
        if tag in {"input", "select", "textarea"}:
            self.input_ids.append(attributes.get("id", ""))
        if "aria-live" in attributes:
            self.live_regions.append(attributes["aria-live"])
        if tag == "html":
            self.html_lang = attributes.get("lang")
        if tag == "img" and not attributes.get("alt"):
            self.images_without_alt += 1
        if role := attributes.get("role"):
            self.roles.append(role)
        if labelled := attributes.get("aria-labelledby"):
            self.labelled_by.extend(labelled.split())


def _parse(markup: str) -> _Collector:
    """Parse markup and return the collected structure."""
    collector = _Collector()
    collector.feed(markup)
    return collector


def _channel(value: float) -> float:
    """Linearise one sRGB channel per the WCAG definition."""
    if value <= 0.03928:
        return value / 12.92
    return float(((value + 0.055) / 1.055) ** 2.4)


def relative_luminance(colour: str) -> float:
    """Return the WCAG relative luminance of a ``#rrggbb`` colour.

    Raises:
        ValueError: If ``colour`` is not a six-digit hex colour.
    """
    match = _HEX.match(colour)
    if match is None:
        raise ValueError(f"not a six-digit hex colour: {colour!r}")
    digits = match.group(1)
    red, green, blue = (int(digits[index : index + 2], 16) / 255 for index in (0, 2, 4))
    return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio between two colours."""
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _declared_colours() -> dict[str, str]:
    """Return every custom property declared in the stylesheet."""
    return dict(_VAR_DECLARATION.findall(STYLESHEET.read_text(encoding="utf-8")))


def _declared_pairs() -> list[tuple[str, str]]:
    """Return every ``contrast-pair`` annotation in the stylesheet."""
    return _CONTRAST_PAIR.findall(STYLESHEET.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# contrast
# --------------------------------------------------------------------------- #
def test_the_stylesheet_declares_contrast_pairs() -> None:
    """The annotations the contrast test relies on are present."""
    assert len(_declared_pairs()) >= 6


def test_known_reference_ratios_are_computed_correctly() -> None:
    """Validates the formula against values that are fixed by the spec."""
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
    assert contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0, abs=0.01)
    assert contrast_ratio("#767676", "#ffffff") == pytest.approx(4.54, abs=0.02)


@pytest.mark.parametrize(("foreground", "background"), _declared_pairs())
def test_every_declared_colour_pair_meets_aa(foreground: str, background: str) -> None:
    """Every foreground/background pair used in the design meets 4.5:1."""
    colours = _declared_colours()
    ratio = contrast_ratio(colours[foreground], colours[background])
    assert ratio >= MINIMUM_CONTRAST, (
        f"--{foreground} on --{background} is {ratio:.2f}:1, below {MINIMUM_CONTRAST}:1"
    )


def test_focus_indicator_is_visible_and_offset() -> None:
    """A visible focus ring is required, not merely the browser default.

    Asserted by shape rather than by variable name, so renaming a colour token is
    a refactor and not a failing test, while removing the ring still fails.
    """
    css = STYLESHEET.read_text(encoding="utf-8")
    assert ":focus-visible" in css
    ring = re.search(r":focus-visible\s*\{[^}]*outline:\s*(\d+)px solid var\(--[a-z-]+\)", css)
    assert ring is not None, "no themed outline declared on :focus-visible"
    assert int(ring.group(1)) >= 2, "a 1px ring is too easy to miss"
    assert "outline-offset" in css


def test_no_outline_is_removed_without_replacement() -> None:
    """Guards against a future edit suppressing focus visibility."""
    css = STYLESHEET.read_text(encoding="utf-8")
    for forbidden in ("outline: none;", "outline: 0;"):
        occurrences = css.count(forbidden)
        # `main:focus` legitimately drops the ring: it is a programmatic target
        # for the skip link and shows :focus-visible instead.
        assert occurrences <= 1


# --------------------------------------------------------------------------- #
# structure, on every page the app can render
# --------------------------------------------------------------------------- #
@pytest.fixture
def pages(client: TestClient) -> dict[str, str]:
    """Render every page: upload, result, gate rejection, and form error."""
    clean = client.get("/samples/clean_three_drugs").content
    illegible = client.get("/samples/illegible_scan").content
    return {
        "upload": client.get("/").text,
        "result": client.post("/guide", files={"prescription": ("a.png", clean, "image/png")}).text,
        "rejected": client.post(
            "/guide", files={"prescription": ("a.png", illegible, "image/png")}
        ).text,
        "error": client.post(
            "/guide", files={"prescription": ("a.png", b"PK\x03\x04" + b"0" * 64, "image/png")}
        ).text,
    }


def test_every_page_declares_a_language(pages: dict[str, str]) -> None:
    """Screen readers need a language to choose a voice."""
    for name, markup in pages.items():
        assert _parse(markup).html_lang == "en", name


def test_every_page_has_the_required_landmarks(pages: dict[str, str]) -> None:
    """A keyboard or screen-reader user needs landmarks to navigate."""
    for name, markup in pages.items():
        tags = _parse(markup).tags
        assert "header" in tags, name
        assert "main" in tags, name
        assert "footer" in tags, name


def test_every_page_offers_a_skip_link_to_main(pages: dict[str, str]) -> None:
    """The first focusable element bypasses the banner."""
    for name, markup in pages.items():
        assert 'class="skip-link" href="#main"' in markup, name
        assert 'id="main"' in markup, name


def test_every_page_has_exactly_one_h1(pages: dict[str, str]) -> None:
    """One page title, so the document outline is unambiguous."""
    for name, markup in pages.items():
        assert _parse(markup).headings.count(1) == 1, name


def test_heading_levels_never_skip(pages: dict[str, str]) -> None:
    """An h2 is never followed directly by an h4."""
    for name, markup in pages.items():
        levels = _parse(markup).headings
        for previous, current in pairwise(levels):
            assert current <= previous + 1, f"{name}: h{previous} followed by h{current}"


def test_every_form_control_has_a_label(pages: dict[str, str]) -> None:
    """No control is left to be identified by placement alone."""
    parsed = _parse(pages["upload"])
    assert parsed.input_ids
    for input_id in parsed.input_ids:
        assert input_id, "every control needs an id to be labelled"
        assert input_id in parsed.labels_for


def test_results_are_announced_through_a_live_region(pages: dict[str, str]) -> None:
    """A result arriving after submission must be announced, not silent."""
    for name in ("result", "rejected"):
        parsed = _parse(pages[name])
        assert "polite" in parsed.live_regions, name
        assert "status" in parsed.roles, name


def test_every_aria_labelledby_target_exists(pages: dict[str, str]) -> None:
    """A dangling reference leaves a section unnamed."""
    for name, markup in pages.items():
        parsed = _parse(markup)
        for target in parsed.labelled_by:
            assert target in parsed.ids, f"{name}: aria-labelledby={target} has no element"


def test_no_image_lacks_alternative_text(pages: dict[str, str]) -> None:
    """Any image must be described or explicitly marked decorative."""
    for name, markup in pages.items():
        assert _parse(markup).images_without_alt == 0, name


def test_the_error_message_is_associated_with_the_field(pages: dict[str, str]) -> None:
    """An error a screen reader cannot reach is not an error message."""
    markup = pages["error"]
    assert 'id="prescription-error"' in markup
    assert 'aria-describedby="prescription-hint prescription-error"' in markup
    assert 'aria-invalid="true"' in markup


def test_state_is_not_conveyed_by_colour_alone(pages: dict[str, str], client: TestClient) -> None:
    """Unverified and stopped states carry a word as well as a colour."""
    assert "Stopped" in pages["rejected"]
    unknown = client.get("/samples/unknown_drug_and_blank_page").content
    with_unverified = client.post(
        "/guide", files={"prescription": ("a.png", unknown, "image/png")}
    ).text
    assert "Not verified" in with_unverified
    assert "ask your pharmacist" in with_unverified


def test_the_disclaimer_appears_on_every_page(pages: dict[str, str]) -> None:
    """The "not medical advice" notice is never a page a user can miss."""
    for name, markup in pages.items():
        assert "Not medical advice" in markup, name
        assert "does not give medical advice" in markup, name
