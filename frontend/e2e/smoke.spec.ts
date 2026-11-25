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
  const body = (await page.locator("body").innerText()).toLowerCase();
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
  await expect(page.getByText("What should the image show?")).toBeVisible();
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
  await expect(page.getByText("What should the image show?")).toBeVisible({ timeout: 20_000 });
  await page.waitForTimeout(4_500); // the fake backend expires a question after 3 seconds
  await page.getByRole("button", { name: "Continue" }).click();
  // (Next adds its own role=alert route announcer, so match our notice by its text)
  await expect(page.locator('div[role="alert"]', { hasText: /timed out/i })).toBeVisible();
  await expect(page.getByRole("img", { name: /Result/ })).toHaveCount(0);
  await shot(page, "06-expired");
});

test("a locked model shows the upgrade option", async ({ page }) => {
  // Basic plan: Phoenix is a Pro-tier model
  const phoenix = page.getByRole("button", { name: /Phoenix 1\.0/ });
  await expect(phoenix).toBeDisabled();
  await expect(phoenix).toContainText("Upgrade");
  await expect(phoenix).toHaveAttribute("title", /Upgrade your plan/);
  await expect(page.getByRole("button", { name: /FLUX\.2 klein 4B/ })).toBeEnabled();
  await shot(page, "07-locked-model");
});

test("changing the plan updates the locks and the limits", async ({ page }) => {
  const variations = page.getByLabel("Variations");
  // Basic: up to 2 variations, Phoenix locked, 2K is "coming soon"
  await expect(variations.locator("option")).toHaveCount(2);
  await expect(page.getByRole("option", { name: "2K (coming soon)" })).toBeDisabled();
  await expect(page.getByRole("button", { name: /Phoenix 1\.0/ })).toBeDisabled();

  await page.getByLabel("Plan").selectOption("pro");
  await expect(variations.locator("option")).toHaveCount(4);
  await expect(page.getByRole("button", { name: /Phoenix 1\.0/ })).toBeEnabled();
  await shot(page, "08-pro-plan");

  // back to Basic: the limit and the lock return, and a locked pick goes back to Auto
  await page.getByRole("button", { name: /Phoenix 1\.0/ }).click();
  await page.getByLabel("Plan").selectOption("basic");
  await expect(variations.locator("option")).toHaveCount(2);
  await expect(page.getByRole("button", { name: /Phoenix 1\.0/ })).toBeDisabled();
  await expect(page.getByRole("button", { name: /Auto \(Smart Router\)/ })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
});

test("decided OpenRouter models are listed as coming soon, in both views", async ({ page }) => {
  // The models the spec decided (GPT, Nano Banana, Seedream ...) are visible but cannot be picked
  for (const name of [/GPT Image 2\.5 Flare/, /Nano Banana 2\.1/, /Seedream 5\.0 Flash/, /Krea 2 Large/]) {
    const row = page.getByRole("button", { name });
    await expect(row).toBeDisabled();
    await expect(row).toContainText("Soon");
    await expect(row).toHaveAttribute("title", "Coming soon");
  }
  // usable models come before them
  const names = await page.locator("ul button").allInnerTexts();
  expect(names[1]).toMatch(/FLUX\.1 schnell|FLUX\.2 klein/);
  await shot(page, "09-list-view");

  // Family view groups by display name
  await page.getByRole("tab", { name: "Family" }).click();
  for (const family of [
    "Google Nano Banana & Gemini",
    "OpenAI GPT Image",
    "FLUX",
    "ByteDance Seedream",
    "Recraft Design & Vector",
    "Qwen",
    "Meta Muse",
  ]) {
    await expect(page.getByText(family, { exact: true })).toBeVisible();
  }
  await shot(page, "10-family-view");

  // search works across the list
  await page.getByLabel("Search models").fill("banana");
  await expect(page.getByRole("button", { name: /Nano Banana Pro/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /FLUX\.1 schnell/ })).toHaveCount(0);
});

test("the picker shows credits, reference limits and capability pills", async ({ page }) => {
  // Auto is first and called the Smart Router
  const first = page.locator("ul button").first();
  await expect(first).toContainText("Auto (Smart Router)");

  const vector = page.getByRole("button", { name: /Recraft V4\.1 Vector/ });
  await expect(vector).toContainText("7 cr");
  await expect(vector).toContainText("refs ≤ 1");
  await expect(vector).toContainText("SVG Vector");
  await expect(page.getByRole("button", { name: /Recraft V4\.1 Flash \(free\)/ })).toContainText("text only");
  await expect(page.getByRole("button", { name: /Seedream 5\.5/ })).toContainText("Multi-Ref");
  await expect(page.getByRole("button", { name: /Nano Banana Pro/ })).toContainText("12 cr");
  await expect(page.getByRole("button", { name: /Nano Banana Pro/ })).toContainText("4K Studio");
  await expect(page.getByRole("button", { name: /GPT Image 2 / })).toContainText("Transparent BG");
  await expect(page.getByRole("button", { name: /FLUX\.1 schnell/ })).toContainText("Fast");
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
