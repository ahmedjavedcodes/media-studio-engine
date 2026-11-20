/**
 * Scripted stand-in for the backend, used when NEXT_PUBLIC_MOCK=1. It replays the same events and
 * results the real API sends, with placeholder images from /public/mock.
 *
 * Pick the outcome with a word in the prompt:
 *   "reject" -> rejected       "fail" -> failed          "issues" -> done_with_issues
 *   "question" (manual mode) -> needs_input, then done once an answer is in settings
 * anything else -> done. An operation other than create (edit / merge / restyle) -> escalate,
 * with that operation's own "not available yet" message, as the real API returns.
 * A "complex" prompt is NOT special any more: the API makes one image for it.
 *
 * Images stream in as they pass the check (image_ready), one at a time, before the result.
 */
import type {
  Catalog,
  CatalogModel,
  CreateRequest,
  Handlers,
  Operation,
  Plan,
  PublicLabel,
  RunEvent,
  RunResult,
  Status,
} from "./api";

type Scenario = Status | "running";
interface MockRun {
  req: CreateRequest;
  scenario: Scenario;
  cancelled: boolean;
  result?: RunResult;
}

const runs = new Map<string, MockRun>();
const LABELS: PublicLabel[] = ["Understanding your request", "Creating your image", "Finishing up"];
const STEP_MS = 1100;
const BRANCH_MS = 350;

// The real API's per-operation messages (agents/image_generation/agent.py).
const STUB_SUMMARY: Record<Exclude<Operation, "create">, string> = {
  edit: "Editing an image is not available yet.",
  merge: "Merging images is not available yet.",
  restyle: "Restyling an image is not available yet.",
};

const PLACEHOLDERS = 4;

/** Mock images live in /public/mock: img_mock_<k> maps onto placeholder <k>. */
export function mockImageUrl(imageId: string): string {
  const k = Number(imageId.replace(/\D+/g, "")) || 1;
  return `/mock/placeholder-${((k - 1) % PLACEHOLDERS) + 1}.svg`;
}

const RANK: Record<Plan, number> = { free: 0, pro: 1, max: 2, ultra: 3 };
// A few options from the real catalogue (services/catalog.py), one per plan and a batch-only one.
const MOCK_MODELS: Omit<CatalogModel, "locked" | "upgrade_to">[] = [
  { id: "openai/gpt-image-2@low", slug: "openai/gpt-image-2", name: "GPT Image 2 (low)", plan: "free", settings: { quality: "low" }, price_usd: 0.006, batch_only: false },
  { id: "black-forest-labs/flux.2-klein-4b", slug: "black-forest-labs/flux.2-klein-4b", name: "FLUX.2 Klein 4B", plan: "free", settings: {}, price_usd: 0.014, batch_only: false },
  { id: "black-forest-labs/flux.2-pro", slug: "black-forest-labs/flux.2-pro", name: "FLUX.2 Pro", plan: "pro", settings: {}, price_usd: 0.03, batch_only: false },
  { id: "recraft/recraft-v4.1", slug: "recraft/recraft-v4.1", name: "Recraft v4.1", plan: "max", settings: {}, price_usd: 0.035, batch_only: false },
  { id: "google/gemini-3.1-flash-image", slug: "google/gemini-3.1-flash-image", name: "Nano Banana 2", plan: "ultra", settings: {}, price_usd: 0.067, batch_only: false },
  { id: "openai/gpt-image-2@high", slug: "openai/gpt-image-2", name: "GPT Image 2 (high)", plan: "ultra", settings: { quality: "high" }, price_usd: 0.211, batch_only: true },
];

export async function mockCatalog(plan: Plan): Promise<Catalog> {
  return {
    plan,
    default_model: "auto",
    models: MOCK_MODELS.map((m) => {
      const locked = RANK[m.plan] > RANK[plan];
      return { ...m, locked, upgrade_to: locked ? m.plan : null };
    }),
    max_images: plan === "free" ? 2 : 4,
    aspect_ratios: ["1:1", "4:3", "3:4", "16:9", "9:16", "4:5", "3:2"],
    default_aspect_ratio: "1:1",
    max_resolution: plan === "ultra" ? "4K" : "2K",
    resolutions: plan === "ultra" ? ["1K", "2K", "4K"] : ["1K", "2K"],
    modes: ["auto", "manual"],
    operations: { create: "live", edit: "coming_soon", merge: "coming_soon", restyle: "coming_soon" },
  };
}

function scenarioFor(req: CreateRequest): Scenario {
  const text = req.prompt.toLowerCase();
  if (req.operation !== "create") return "escalate";
  if (text.includes("reject")) return "rejected";
  if (text.includes("fail")) return "failed";
  if (text.includes("issues")) return "done_with_issues";
  const answered = Object.keys(req.settings).some((key) => key !== "aspect_ratio");
  if (text.includes("question") && req.mode === "manual" && !answered) return "needs_input";
  return "done";
}

function images(count: number): RunResult["images"] {
  return Array.from({ length: count }, (_, i) => ({
    image_id: `img_mock_${i + 1}`,
    url: mockImageUrl(`img_mock_${i + 1}`),
    quality_score: Math.max(3, 9 - i),
  }));
}

function resultFor(runId: string, run: MockRun, checked: number): RunResult {
  const base = { run_id: runId, reason: null, images: [], issues: [], questions: [], assumptions: [] };
  switch (run.scenario) {
    case "done":
      return { ...base, status: "done", summary: "Your image is ready.", images: images(run.req.n) };
    case "done_with_issues":
      return {
        ...base,
        status: "done_with_issues",
        summary: "Your image is ready, with some issues noted.",
        images: images(run.req.n),
        issues: ["The lamps are slightly blown out.", "1 of the requested images could not be created."],
        assumptions: ["aspect_ratio: 1:1"],
      };
    case "needs_input":
      return {
        ...base,
        status: "needs_input",
        summary: "A little more information is needed before the image can be made.",
        reason: "the request needs details only the user can give",
        questions: ["Which city should the street be in?", "Is it day or night?"],
      };
    case "rejected":
      return {
        ...base,
        status: "rejected",
        summary: "This image could not be delivered because it did not pass the safety review.",
      };
    case "failed":
      return { ...base, status: "failed", summary: "The image could not be created.", reason: "provider unavailable" };
    case "escalate": {
      const op = run.req.operation as Exclude<Operation, "create">;
      return { ...base, status: "escalate", summary: STUB_SUMMARY[op], reason: `${op} not implemented` };
    }
    default:
      return { ...base, status: "cancelled", summary: "Image creation was cancelled.", images: images(checked) };
  }
}

export async function mockCreateRun(req: CreateRequest): Promise<{ run_id: string }> {
  const run_id = `run_mock_${Math.random().toString(36).slice(2, 10)}`;
  runs.set(run_id, { req, scenario: scenarioFor(req), cancelled: false });
  return { run_id };
}

export async function mockCancel(runId: string): Promise<void> {
  const run = runs.get(runId);
  if (run) run.cancelled = true;
}

export async function mockGetRun(runId: string): Promise<RunResult> {
  const run = runs.get(runId);
  if (!run) throw new Error("404 Not Found");
  return run.result ?? { ...resultFor(runId, run, 0), status: "failed", summary: "Still running." };
}

/** Replays the script. Returns a function that stops listening. */
export function mockSubscribe(runId: string, handlers: Handlers): () => void {
  const run = runs.get(runId);
  if (!run) {
    handlers.onError("Unknown run.");
    return () => undefined;
  }
  const timers: ReturnType<typeof setTimeout>[] = [];
  let stopped = false;
  let at = 200;
  let checked = 0;

  const event = (step: number, phase: RunEvent["phase"]): RunEvent => ({
    run_id: runId,
    step,
    public_label: LABELS[step - 1],
    phase,
    ts: new Date().toISOString(),
  });
  // A checked image can be shown at once (image_ready), before the run's result.
  const streamImage = (k: number) => {
    if (run.scenario !== "done" && run.scenario !== "done_with_issues") return;
    handlers.onEvent({ ...event(3, "progress"), phase: "image_ready", image_id: `img_mock_${k}` });
  };
  const later = (ms: number, fn: () => void) => {
    timers.push(
      setTimeout(() => {
        if (!stopped) fn();
      }, ms),
    );
  };
  const finish = (result: RunResult) => {
    run.result = result;
    handlers.onResult(result);
  };

  // A scenario ends early at a step: escalate/needs_input never start generating, failed stops
  // while creating, rejected is decided at the critique.
  const lastStep: Record<string, number> = { escalate: 0, needs_input: 1, failed: 2, rejected: 3 };
  const stopAfter = lastStep[run.scenario] ?? 3;
  const branches = run.req.n;

  if (stopAfter === 0) {
    later(at + 300, () => finish(resultFor(runId, run, 0)));
    return () => {
      stopped = true;
      timers.forEach(clearTimeout);
    };
  }

  for (let step = 1; step <= stopAfter; step++) {
    const duration = step === 1 ? STEP_MS : STEP_MS + (branches > 1 ? branches * BRANCH_MS : 0);
    later(at, () => {
      if (run.cancelled) return;
      handlers.onEvent(event(step, "started"));
    });
    if (step > 1 && branches > 1) {
      for (let b = 1; b <= branches; b++) {
        later(at + b * BRANCH_MS, () => {
          if (run.cancelled) return;
          handlers.onEvent(event(step, "progress"));
          if (step === 2) checked = 0;
          if (step === 3) {
            checked = b;
            streamImage(b);
          }
        });
      }
    }
    later(at + duration, () => {
      if (run.cancelled) return;
      if (step === 3 && branches === 1) streamImage(1);
      handlers.onEvent(event(step, "finished"));
    });
    at += duration;
  }
  later(at + 200, () => {
    if (!run.cancelled) finish(resultFor(runId, run, 0));
  });

  // A cancel stops the script: the in-flight step finishes, and images already checked are kept.
  const watcher = setInterval(() => {
    if (run.cancelled && !stopped) {
      stopped = true;
      timers.forEach(clearTimeout);
      clearInterval(watcher);
      const step = Math.min(3, Math.max(1, Math.ceil((Date.now() - startedAt) / STEP_MS)));
      handlers.onEvent(event(step, "finished"));
      finish({ ...resultFor(runId, { ...run, scenario: "running" }, checked), status: "cancelled" });
    }
  }, 100);
  const startedAt = Date.now();

  return () => {
    stopped = true;
    timers.forEach(clearTimeout);
    clearInterval(watcher);
  };
}
