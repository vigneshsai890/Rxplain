/**
 * axe-core accessibility audit against a running RxPlain instance.
 *
 * Drives a real browser through the real form, so the pages produced by a POST -
 * the guide, and the two refusal pages - are audited as a user receives them, not
 * as a saved snapshot.
 *
 * Usage:
 *   npm --prefix tools/axe ci
 *   RXPLAIN_URL=http://127.0.0.1:8080 node tools/axe/audit.mjs
 *
 * Exits non-zero on any WCAG 2.1 A or AA violation.
 */

import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import process from "node:process";

// axe-core ships as CommonJS, so its named exports arrive on the default object.
import axeCore from "axe-core";
import puppeteer from "puppeteer";

const axeSource = axeCore.source;

const BASE_URL = process.env.RXPLAIN_URL ?? "http://127.0.0.1:8080";
const PRESCRIPTIONS = resolve(
  import.meta.dirname,
  "..",
  "..",
  "tests",
  "golden",
  "prescriptions",
);

// WCAG 2.1 Level A and AA only. Best-practice rules are excluded so the gate
// tracks the conformance target rather than shifting advice.
const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"];

// 320px is the width WCAG 2.1 "Reflow" (1.4.10) names explicitly. The rest cover
// a common phone, a tablet and a laptop.
const VIEWPORTS = [
  { name: "320px", width: 320, height: 720, mobile: true },
  { name: "375px", width: 375, height: 812, mobile: true },
  { name: "768px", width: 768, height: 1024, mobile: true },
  { name: "1440px", width: 1440, height: 900, mobile: false },
];

/** A file whose bytes are a ZIP, to exercise the upload rejection page. */
function spoofedFile() {
  const directory = mkdtempSync(join(tmpdir(), "rxplain-axe-"));
  const path = join(directory, "not-really-a-png.png");
  writeFileSync(path, Buffer.concat([Buffer.from("PK\x03\x04"), Buffer.alloc(64)]));
  return path;
}

/**
 * Horizontal scrolling at the current viewport, in pixels.
 *
 * WCAG 1.4.10 requires none: a reader who has zoomed in must not have to scroll
 * sideways to finish a sentence.
 */
async function horizontalOverflow(page) {
  return page.evaluate(
    () =>
      Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) -
      window.innerWidth,
  );
}

/**
 * Block until the stylesheet is loaded and applied.
 *
 * `domcontentloaded` fires before stylesheets finish. Against a local server the
 * CSS arrives in about a millisecond so this never mattered; against a real
 * deployment axe ran on an unstyled page and reported the browser's default link
 * colour as a contrast failure. Auditing before the design is applied tests
 * nothing, so every navigation waits for this.
 */
async function waitForStyles(page) {
  await page.waitForFunction(
    () =>
      document.styleSheets.length > 0 &&
      getComputedStyle(document.documentElement).getPropertyValue("--link").trim() !== "",
    { timeout: 15000 },
  );
}

/** Run axe against the page's current document. */
async function audit(page, label) {
  const overflow = await horizontalOverflow(page);
  if (overflow > 1) {
    console.log(`${label.padEnd(30)} horizontal overflow of ${overflow}px (WCAG 1.4.10)`);
  }
  await page.evaluate(axeSource);
  const results = await page.evaluate(
    async (tags) => window.axe.run(document, { runOnly: { type: "tag", values: tags } }),
    TAGS,
  );
  const { violations, passes, incomplete } = results;
  console.log(
    `${label.padEnd(30)} violations=${violations.length} passes=${passes.length} ` +
      `incomplete=${incomplete.length} overflow=${overflow}px`,
  );
  for (const violation of violations) {
    console.log(`  [${violation.impact}] ${violation.id}: ${violation.help}`);
    for (const node of violation.nodes.slice(0, 3)) {
      console.log(`      ${node.html.slice(0, 140)}`);
    }
    console.log(`      ${violation.helpUrl}`);
  }
  return violations.length + (overflow > 1 ? 1 : 0);
}

/** Submit a file through the upload form and audit the resulting page. */
async function submitAndAudit(page, filePath, label) {
  await page.goto(`${BASE_URL}/`, { waitUntil: "load" });
  const input = await page.waitForSelector("#prescription");
  await input.uploadFile(filePath);
  await Promise.all([
    page.waitForNavigation({ waitUntil: "load" }),
    page.click('button[type="submit"]'),
  ]);
  await waitForStyles(page);
  return audit(page, label);
}

const browser = await puppeteer.launch({
  headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});

let total = 0;
const spoofed = spoofedFile();
try {
  for (const viewport of VIEWPORTS) {
    const page = await browser.newPage();
    await page.setViewport({
      width: viewport.width,
      height: viewport.height,
      deviceScaleFactor: viewport.mobile ? 2 : 1,
      isMobile: viewport.mobile,
      hasTouch: viewport.mobile,
    });
    console.log(`\n--- ${viewport.name} ---`);

    await page.goto(`${BASE_URL}/`, { waitUntil: "load" });
    await waitForStyles(page);
    total += await audit(page, "upload page");

    total += await submitAndAudit(
      page,
      join(PRESCRIPTIONS, "clean_three_drugs.png"),
      "guide (3 verified)",
    );
    total += await submitAndAudit(
      page,
      join(PRESCRIPTIONS, "unknown_drug_and_blank_page.png"),
      "guide (unverified)",
    );
    total += await submitAndAudit(
      page,
      join(PRESCRIPTIONS, "illegible_scan.png"),
      "gate refusal",
    );
    total += await submitAndAudit(page, spoofed, "upload refusal");
    await page.close();
  }
} finally {
  await browser.close();
}

if (total > 0) {
  console.error(`\n${total} WCAG 2.1 AA failure(s) across ${VIEWPORTS.length} viewports`);
  process.exit(1);
}
console.log(
  `\naxe-core: no WCAG 2.1 AA violations and no horizontal overflow ` +
    `at ${VIEWPORTS.map((v) => v.name).join(", ")}`,
);
