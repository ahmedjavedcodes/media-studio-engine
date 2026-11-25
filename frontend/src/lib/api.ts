// Types and the calls for the image create engine. Mirrors the bodies of the /v1 API.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Plan = "basic" | "pro" | "ultra" | "max";
export type Tier = "basic" | "pro" | "ultra" | "max";

export type Blocked = {
  code: "content_blocked";
  category: string;
  blocked_by: "prompt" | "image" | "both";
  message: string;
  tip: string | null;
  warning: boolean;
  // Instant reject (strike)
  strike_count?: number;
  strike_limit?: number;
  strike_text?: string;
  banned?: boolean;
  // Standard reject (cooldown)
  retry_after?: number;
  retry_text?: string;
};

export type Settings = {
  model: string; // "auto" or a model id
  aspect_ratio: string;
  resolution: "1K" | "2K";
  n: number;
  draft?: boolean; // fast, low-credit mode: the router picks the cheapest model
};

export type PlanInfo = { id: Plan; max_n: number; max_resolution: string; tier: Tier };
export type PlansResponse = {
  trace_ui?: boolean; // the dev trace panel is on (TRACE_UI=true on the server, never in prod)
  plans: PlanInfo[];
  aspect_ratios: string[];
  resolutions: { id: string; available: boolean }[];
};

export type ModelInfo = {
  id: string;
  name: string;
  family: string;
  family_label: string; // display name, for the Family view
  tier: Tier;
  tag: string;
  featured: boolean;
  credits_per_image: number;
  ref_surcharge: number; // extra credits for each reference image after the first
  resolution_multipliers: Record<string, number>;
  max_refs: number; // reference images the model takes (0 = text only)
  capabilities: string[];
  locked: boolean; // the plan is too low
  available: boolean; // false = "coming soon"
  aspect_ratios: string[];
};

export type Option = { id: string; label: string; recommended?: boolean };
export type Question = { id: string; text: string; options: Option[] };

export type ImageOut = { id: string; url: string; thumb_url: string; width: number; height: number };

// One SSE event. `type` is the SSE event name.
// Images arrive only in `ready`, after the safety check. While they are made you get progress.
export type RunEvent = {
  type: "status" | "question" | "ready" | "error" | "expired";
  run_id: string;
  label?: string;
  progress?: string;
  images?: ImageOut[];
  recommendation?: string;
  questions?: Question[];
  code?: string;
  message?: string;
};

export type Draft = { prompt: string; plan: Plan; settings: Settings };

export type StartResult =
  | { kind: "started"; runId: string; token: string }
  | { kind: "blocked"; body: Blocked }
  | { kind: "cooldown"; message: string; retryAfter: number }
  | { kind: "error"; code: string; message: string };

// The stream URL. EventSource cannot send headers, so access is the signed token in the URL.
export function eventsUrl(runId: string, token: string): string {
  return `${API_URL}/v1/runs/${runId}/events?t=${encodeURIComponent(token)}`;
}

// Image paths from the server already carry `?t=<token>`.
export function imageUrl(path: string): string {
  return `${API_URL}${path}`;
}

async function readFailure(res: Response): Promise<StartResult> {
  const body = await res.json().catch(() => ({}));
  if (res.status === 422 && body.code === "content_blocked") return { kind: "blocked", body };
  if (res.status === 429) {
    return { kind: "cooldown", message: body.message ?? "Please wait.", retryAfter: body.retry_after ?? 60 };
  }
  return {
    kind: "error",
    code: body.code ?? "error",
    message: body.message ?? body.detail ?? "Something went wrong. Please try again.",
  };
}

const NETWORK: StartResult = {
  kind: "error",
  code: "network",
  message: "Could not reach the server. Please try again.",
};

export async function startRun(
  prompt: string,
  images: string[],
  plan: Plan,
  settings: Settings,
): Promise<StartResult> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/v1/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, images, plan, settings }),
    });
  } catch {
    return NETWORK;
  }
  if (res.ok) {
    const body = await res.json();
    return { kind: "started", runId: body.run_id, token: body.token };
  }
  return readFailure(res);
}

// Answers: {question id: option id}. For "Something else": {q1: "other", q1_text: "..."}.
export async function resumeRun(runId: string, answers: Record<string, string>): Promise<StartResult> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/v1/runs/${runId}/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answers }),
    });
  } catch {
    return NETWORK;
  }
  if (res.ok) {
    const body = await res.json();
    return { kind: "started", runId, token: body.token };
  }
  return readFailure(res);
}

export async function fetchPlans(): Promise<PlansResponse> {
  const res = await fetch(`${API_URL}/v1/plans`);
  return res.json();
}

// Fill in anything an older backend does not send, so a version mismatch cannot crash the picker.
function withDefaults(m: Partial<ModelInfo> & { id: string; name: string }): ModelInfo {
  return {
    family: "",
    family_label: m.family ?? "",
    tier: "basic",
    tag: "",
    featured: false,
    credits_per_image: 1,
    ref_surcharge: 0,
    resolution_multipliers: {},
    max_refs: 99, // unknown: do not disable anything
    capabilities: [],
    locked: false,
    available: true,
    aspect_ratios: ["1:1"],
    ...m,
  } as ModelInfo;
}

// What Auto may cost on this plan: prices of every model it can pick, shown in the picker or not.
export type AutoPoolEntry = {
  max_refs: number;
  credits_per_image: number;
  ref_surcharge: number;
  resolution_multipliers: Record<string, number>;
};

export async function fetchModels(
  plan: Plan,
): Promise<{ models: ModelInfo[]; autoPool: AutoPoolEntry[] }> {
  const res = await fetch(`${API_URL}/v1/models?plan=${plan}`);
  const body = await res.json();
  return {
    models: (body.models as (Partial<ModelInfo> & { id: string; name: string })[]).map(withDefaults),
    autoPool: (body.auto_pool ?? []) as AutoPoolEntry[], // an older backend sends none
  };
}

export async function fetchRun(
  runId: string,
): Promise<{ status: string; result?: { images: ImageOut[]; recommendation?: string }; draft?: Draft } | null> {
  const res = await fetch(`${API_URL}/v1/runs/${runId}`).catch(() => null);
  return res && res.ok ? res.json() : null;
}

// ---- the dev trace panel (GET /v1/runs/{id}/trace). Dev only: the server must have TRACE_UI on.

export type TraceSpan = {
  id: number;
  parent: number | null;
  name: string;
  kind: string;
  status: "ok" | "error" | "running";
  start_ms: number; // relative to the first span. The guardrail starts before the run: negative
  duration_ms: number;
  model: string | null;
  provider: string | null;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  metadata: Record<string, unknown>;
  error: string | null;
};

export type TraceTotals = {
  duration_ms: number;
  images_usd: number;
  images_cost_estimated: boolean; // true when a provider billed nothing we could read
  llm_usd_estimate: number;
  total_usd: number;
  neurons: number;
  tokens: { prompt: number; completion: number };
  credits_held: number;
  credits_charged: number;
  images_requested: number;
  images_delivered: number;
};

export type Trace = {
  run_id: string;
  status: string; // running | waiting | resuming | done | failed | expired
  input: { prompt?: string; plan?: string; settings?: Record<string, unknown>; n_refs?: number };
  metadata: Record<string, unknown>;
  totals: TraceTotals;
  spans: TraceSpan[];
};

export async function fetchTrace(runId: string, token: string): Promise<Trace | null> {
  const res = await fetch(`${API_URL}/v1/runs/${runId}/trace?t=${encodeURIComponent(token)}`, {
    cache: "no-store",
  }).catch(() => null);
  return res && res.ok ? ((await res.json()) as Trace) : null;
}
