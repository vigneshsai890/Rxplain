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

/** A file whose bytes are a ZIP, to exercise the upload rejection page. */
function spoofedFile() {
  const directory = mkdtempSync(join(tmpdir(), "rxplain-axe-"));
  const path = join(directory, "not-really-a-png.png");
  writeFileSync(path, Buffer.concat([Buffer.from("PK\x03\x04"), Buffer.alloc(64)]));
  return path;
}

/** Run axe against the page's current document. */
async function audit(page, label) {
  await page.evaluate(axeSource);
  const results = await page.evaluate(
    async (tags) => window.axe.run(document, { runOnly: { type: "tag", values: tags } }),
    TAGS,
  );
  const { violations, passes, incomplete } = results;
  console.log(
    `${label.padEnd(22)} violations=${violations.length} passes=${passes.length} ` +
      `incomplete=${incomplete.length}`,
  );
  for (const violation of violations) {
    console.log(`  [${violation.impact}] ${violation.id}: ${violation.help}`);
    for (const node of violation.nodes.slice(0, 3)) {
      console.log(`      ${node.html.slice(0, 140)}`);
    }
    console.log(`      ${violation.helpUrl}`);
  }
  return violations.length;
}

/** Submit a file through the upload form and audit the resulting page. */
async function submitAndAudit(page, filePath, label) {
  await page.goto(`${BASE_URL}/`, { waitUntil: "domcontentloaded" });
  const input = await page.waitForSelector("#prescription");
  await input.uploadFile(filePath);
  await Promise.all([
    page.waitForNavigation({ waitUntil: "domcontentloaded" }),
    page.click('button[type="submit"]'),
  ]);
  return audit(page, label);
}

const browser = await puppeteer.launch({
  headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});

let total = 0;
try {
  const page = await browser.newPage();

  await page.goto(`${BASE_URL}/`, { waitUntil: "domcontentloaded" });
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
  total += await submitAndAudit(page, spoofedFile(), "upload refusal");
} finally {
  await browser.close();
}

if (total > 0) {
  console.error(`\naxe-core found ${total} WCAG 2.1 AA violation(s)`);
  process.exit(1);
}
console.log("\naxe-core: no WCAG 2.1 AA violations");
