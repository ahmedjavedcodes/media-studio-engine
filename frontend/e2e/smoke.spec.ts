import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const shots = process.env.E2E_SHOTS ?? path.join(__dirname, "screenshots");
fs.mkdirSync(shots, { recursive: true });

async function shot(page: Page, name: string) {
  await page.screenshot({ path: path.join(shots, `${name}.png`), fullPage: true });
}

const prompt = (page: Page) => page.getByPlaceholder("Describe the image you want");
const createButton = (page: Page) => page.getByRole("button", { name: /^Create/ });

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Image Create" })).toBeVisible();
  // the model list comes from the API
  await expect(page.getByRole("button", { name: /FLUX\.2 klein 4B/ })).toBeVisible();
});

test("create -> progress -> ready", async ({ page }) => {
  await prompt(page).fill("a red fox in the snow");
  await page.getByLabel("Variations").selectOption("2");
  await shot(page, "01-form");
  await createButton(page).click();

  // progress is friendly words only, and no image is shown before it is released
  await expect(page.getByRole("status")).toContainText(/Understanding|Thinking|Compiling|Creating/);
  await shot(page, "02-progress");
  await expect(page.getByRole("img", { name: /Result/ })).toHaveCount(2, { timeout: 40_000 });
  await expect(page.getByText(/matches your prompt/)).toBeVisible();
  // The dev trace panel is on in this test, and it names models and steps on purpose. The rest of
  // the page must not.
  const body = (
    await page.evaluate(() => {
      const copy = document.body.cloneNode(true) as HTMLElement;
      copy.querySelector('[data-testid="trace-panel"]')?.remove();
      return copy.textContent ?? "";
    })
  ).toLowerCase();
  for (const hidden of ["enhancer", "critic", "cloudflare", "flux-2"]) {
    expect(body.includes(hidden), hidden).toBe(false);
  }
  expect(/\bseed\b/.test(body), "seed").toBe(false); // "Seedream" is a model name, a seed is not

  const img = page.getByRole("img", { name: "Result 1" });
  expect(await img.getAttribute("src")).toMatch(/\/v1\/files\/.+\?t=/); // signed url
  await shot(page, "03-ready");
});

test("question -> answer -> ready", async ({ page }) => {
  await prompt(page).fill("make it nice");
  await createButton(page).click();
  await expect(page.getByText("Need a quick detail").first()).toBeVisible({ timeout: 20_000 });
  await expect(page.locator("legend", { hasText: "What should the image show?" })).toBeVisible();
  await expect(page.getByLabel(/A product on a table/)).toBeChecked(); // the recommended one
  await expect(page.getByText("Something else")).toBeVisible();
  await shot(page, "04-question");
  await page.getByLabel("A landscape").check();
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("img", { name: /Result/ })).toHaveCount(1, { timeout: 40_000 });
  await shot(page, "05-question-answered");
});

test("question -> expiry -> notice", async ({ page }) => {
  await prompt(page).fill("make it nice");
  await createButton(page).click();
  await expect(page.locator("legend", { hasText: "What should the image show?" })).toBeVisible({ timeout: 20_000 });
  await page.waitForTimeout(4_500); // the fake backend expires a question after 3 seconds
  await page.getByRole("button", { name: "Continue" }).click();
  // (Next adds its own role=alert route announcer, so match our notice by its text)
  await expect(page.locator('div[role="alert"]', { hasText: /timed out/i })).toBeVisible();
  await expect(page.getByRole("img", { name: /Result/ })).toHaveCount(0);
  await shot(page, "06-expired");
});

test("a model above the plan shows a lock, and only the usable model is enabled", async ({ page }) => {
  // Basic plan gets models up to Pro. Recraft V4.1 Vector is an Ultra model, so it is locked.
  // It is also "coming soon" (OpenRouter is off)
  const recraft = page.getByRole("button", { name: /Recraft V4\.1 Vector/ });
  await expect(recraft).toBeDisabled();
  await expect(recraft).toContainText("🔒");
  await expect(recraft).toContainText("Soon");
  await expect(recraft).toHaveAttribute("title", /higher plan/);
  // a Basic model that is "coming soon" has no lock
  const flare = page.getByRole("button", { name: /GPT Image 2\.5 Flare/ });
  await expect(flare).toBeDisabled();
  await expect(flare).not.toContainText("🔒");
  await expect(page.getByRole("button", { name: /FLUX\.2 klein 4B/ })).toBeEnabled();
  await shot(page, "07-locked-model");
});

test("changing the plan updates the locks and the limits", async ({ page }) => {
  const variations = page.getByLabel("Variations");
  const seedream = page.getByRole("button", { name: /Seedream 5\.0 Flash/ });
  const vector = page.getByRole("button", { name: /Recraft V4\.1 Vector/ });
  const nanoPro = page.getByRole("button", { name: /Nano Banana Pro/ });
  // Basic (free): up to 2 variations, models up to Pro are open, Ultra and Max are locked,
  // 2K is "coming soon"
  await expect(variations.locator("option")).toHaveCount(2);
  await expect(page.getByRole("option", { name: "2K (coming soon)" })).toBeDisabled();
  await expect(seedream).not.toContainText("🔒");
  await expect(vector).toContainText("🔒");
  await expect(nanoPro).toContainText("🔒");

  // Pro and above can use any model
  await page.getByLabel("Plan").selectOption("pro");
  await expect(variations.locator("option")).toHaveCount(4);
  await expect(vector).not.toContainText("🔒");
  await expect(nanoPro).not.toContainText("🔒");
  await shot(page, "08-pro-plan");

  for (const plan of ["ultra", "max"]) {
    await page.getByLabel("Plan").selectOption(plan);
    await expect(vector).not.toContainText("🔒");
    await expect(page.getByRole("button", { name: /Recraft V4\.1 Pro/ })).not.toContainText("🔒");
  }

  // back to Basic: the limit and the locks return
  await page.getByLabel("Plan").selectOption("basic");
  await expect(variations.locator("option")).toHaveCount(2);
  await expect(vector).toContainText("🔒");
  await expect(page.getByRole("button", { name: /Auto \(Smart Router\)/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
});

test("the curated 14 are listed, in both views, and nothing else", async ({ page }) => {
  // the models the product decided (GPT, Nano Banana, Seedream, Recraft, FLUX) are visible
  // but cannot be picked yet: OpenRouter is off
  for (const name of [/GPT Image 2\.5 Flare/, /Nano Banana 2\.1/, /Seedream 5\.0 Flash/, /FLUX 3 Klein/]) {
    const row = page.getByRole("button", { name });
    await expect(row).toBeDisabled();
    await expect(row).toContainText("Soon");
  }
  // the older and overlapping models, and the free engines, are not in the picker
  for (const name of [
    /Nano Banana 2 /, /GPT Image 1/, /FLUX\.1 schnell/, /Phoenix/, /Krea/, /Muse/, /Seedream 5\.5/,
    /Recraft V4 /, /Qwen/, /Riverflow/, /MAI-Image/, /FLUX\.2 Flex/, /FLUX\.2 klein 9B/,
  ]) {
    await expect(page.getByRole("button", { name })).toHaveCount(0);
  }
  // exactly the 14, plus Auto
  await expect(page.locator("ul button")).toHaveCount(15);
  // usable models come before the others
  const names = await page.locator("ul button").allInnerTexts();
  expect(names[1]).toMatch(/FLUX\.2 klein 4B/);
  await shot(page, "09-list-view");

  // Family view groups by display name
  await page.getByRole("tab", { name: "Family" }).click();
  for (const family of [
    "FLUX",
    "OpenAI GPT Image",
    "ByteDance Seedream",
    "Recraft Design & Vector",
    "Google Nano Banana & Gemini",
  ]) {
    await expect(page.getByText(family, { exact: true })).toBeVisible();
  }
  await expect(page.getByText("Qwen", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Meta Muse", { exact: true })).toHaveCount(0);
  await shot(page, "10-family-view");

  // search works across the list
  await page.getByLabel("Search models").fill("banana");
  await expect(page.getByRole("button", { name: /Nano Banana Pro/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /FLUX\.2 klein 4B/ })).toHaveCount(0);
});

test("the picker shows credits, reference limits and capability pills", async ({ page }) => {
  // Auto is first and called the Smart Router
  await expect(page.locator("ul button").first()).toContainText("Auto (Smart Router)");

  const vector = page.getByRole("button", { name: /Recraft V4\.1 Vector/ });
  await expect(vector).toContainText("7 cr");
  await expect(vector).toContainText("refs ≤ 1");
  await expect(vector).toContainText("SVG Vector");
  await expect(page.getByRole("button", { name: /Recraft V4\.1 Flash/ })).toContainText("text only");
  await expect(page.getByRole("button", { name: /Seedream 5\.0 Flash/ })).toContainText("Multi-Ref");
  await expect(page.getByRole("button", { name: /Seedream 5\.0 Flash/ })).toContainText("refs ≤ 14");
  await expect(page.getByRole("button", { name: /Nano Banana Pro/ })).toContainText("12 cr");
  await expect(page.getByRole("button", { name: /Nano Banana Pro/ })).toContainText("4K Studio");
  await expect(page.getByRole("button", { name: /GPT Image 2\.5 Flare/ })).toContainText("Transparent BG");
  await expect(page.getByRole("button", { name: /GPT Image 2\.5 Flare/ })).toContainText("Text in images");
  await expect(page.getByRole("button", { name: /FLUX\.2 klein 4B/ })).toContainText("Fast");
  await shot(page, "11-pills");
});

test("attached images disable models that cannot take them, and the credit estimate follows", async ({
  page,
}) => {
  const klein = page.getByRole("button", { name: /FLUX\.2 klein 4B/ });
  await expect(klein).toBeEnabled();
  await page.locator('input[type="file"]').setInputFiles({
    name: "ref.png",
    mimeType: "image/png",
    buffer: Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
      "base64",
    ),
  });
  await expect(page.getByText(/1 image attached/)).toBeVisible();
  // klein is text only, so with a reference attached it cannot be picked
  await expect(klein).toBeDisabled();
  await expect(klein).toHaveAttribute("title", /does not take reference images/);
  await shot(page, "12-attached-image");

  // a picked model that cannot take the images goes back to Auto
  await page.locator('input[type="file"]').setInputFiles([]);
  await klein.click();
  await expect(klein).toHaveAttribute("aria-pressed", "true");
  await page.locator('input[type="file"]').setInputFiles({
    name: "ref.png",
    mimeType: "image/png",
    buffer: Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
      "base64",
    ),
  });
  await expect(page.getByRole("button", { name: /Auto \(Smart Router\)/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
});

test("the credit estimate counts variations, and Fast draft is available", async ({ page }) => {
  await expect(createButton(page)).toContainText("up to");
  await page.getByRole("button", { name: /FLUX\.2 klein 4B/ }).click();
  await expect(createButton(page)).toContainText("1 credit");
  await page.getByLabel("Variations").selectOption("2");
  await expect(createButton(page)).toContainText("2 credits");
  await expect(page.getByLabel("Fast draft")).toBeVisible();
});

test("the dev trace panel shows every step with time, model and cost", async ({ page }) => {
  await prompt(page).fill("a red fox in the snow");
  await page.getByLabel("Variations").selectOption("2");
  await createButton(page).click();
  const panel = page.getByTestId("trace-panel");
  await expect(panel).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole("img", { name: /Result/ })).toHaveCount(2, { timeout: 40_000 });
  await expect(panel).toContainText("done", { timeout: 20_000 }); // it follows the run to the end

  // every step, in order: from the guardrail in front, to settle
  for (const step of ["guardrail", "code_checks", "router", "enhancer", "code_guard", "generate", "critic", "settle"]) {
    await expect(panel.locator(`[data-span="${step}"]`).first()).toBeVisible();
  }
  await expect(panel.locator('[data-span="generate_image"]')).toHaveCount(2);
  await expect(panel.locator('[data-span="safety"]')).toHaveCount(2);

  // time, model and provider, cost, tokens and neurons
  const totals = page.getByTestId("trace-totals");
  await expect(totals).toContainText("Time");
  await expect(totals).toContainText("Cost");
  await expect(totals).toContainText("Credits");
  await expect(totals).toContainText("Delivered 2 / 2");
  await expect(totals).toContainText("Neurons");
  await expect(panel.locator('[data-span="enhancer"]')).toContainText("fake-enhancer");
  await expect(panel.locator('[data-span="enhancer"]')).toContainText("neurons");
  await expect(panel.locator('[data-span="generate_image"]').first()).toContainText("cloudflare");
  await expect(panel.locator('[data-span="generate_image"]').first()).toContainText("$0.0011");
  await expect(panel.locator('[data-span="router"]')).toContainText("held");
  await expect(panel.locator('[data-span="settle"]')).toContainText("charged");
  await expect(panel.getByText("slowest")).toHaveCount(1);
  await shot(page, "13-trace");

  // a step opens to show what went in and out
  await panel.locator('[data-span="code_guard"] summary').click();
  await expect(panel.locator('[data-span="code_guard"]')).toContainText("final_prompt");
  await shot(page, "14-trace-step-open");

  // none of this is in the normal page: the images and the recommendation stay clean
  const results = await page.getByRole("img", { name: /Result/ }).count();
  expect(results).toBe(2);
});

test("the trace follows a run that asks a question", async ({ page }) => {
  await prompt(page).fill("make it nice");
  await createButton(page).click();
  const panel = page.getByTestId("trace-panel");
  await expect(page.locator("legend", { hasText: "What should the image show?" })).toBeVisible({ timeout: 20_000 });
  await expect(panel.locator('[data-span="interrupt"]')).toBeVisible();
  await expect(panel.locator('[data-span="interrupt"]')).toHaveAttribute("data-status", "running");
  await page.getByLabel("A landscape").check();
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(panel).toContainText("done", { timeout: 30_000 });
  await expect(panel.locator('[data-span="enhancer"]')).toHaveCount(2); // before and after the answer
  await expect(panel.locator('[data-span="interrupt"]')).toHaveAttribute("data-status", "ok");
});
