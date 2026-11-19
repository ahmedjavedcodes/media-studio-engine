/**
 * Typed client for the contract in project-status/ARCHITECTURE.md, section 8.
 *
 *   POST /runs               -> { run_id }
 *   GET  /runs/:id/events    -> SSE: "step" (RunEvent, many) then "result" (RunResult, once)
 *   POST /runs/:id/cancel
 *   GET  /runs/:id           -> RunResult
 *   GET  /images/:image_id   -> the image file (a URL for <img>; never bytes or base64 in JSON)
 *
 * NEXT_PUBLIC_MOCK=1 (the default) replays scripted events and placeholder images instead, so
 * the UI works without a backend. Set NEXT_PUBLIC_MOCK=0 to use the real API.
 */
import { mockCancel, mockCreateRun, mockGetRun, mockSubscribe } from "./mock";

export type Plan = "free" | "pro" | "max" | "ultra";
export type Mode = "auto" | "manual";
export type AspectRatio = "1:1" | "4:3" | "3:4" | "16:9" | "9:16";

/** Envelope statuses, plus "running" while a run is in progress (RunState only). */
export type Status =
  | "done"
  | "done_with_issues"
  | "needs_input"
  | "rejected"
  | "cancelled"
  | "failed"
  | "escalate";

/** The ONLY strings a user may see. Internal agent names never reach the UI. */
export const PUBLIC_LABELS = [
  "Understanding your request",
  "Creating your image",
  "Finishing up",
] as const;
export type PublicLabel = (typeof PUBLIC_LABELS)[number];

export type Phase = "started" | "progress" | "finished";

export interface CreateRequest {
  prompt: string;
  plan: Plan;
  mode: Mode;
  n: number; // 1..4
  /** aspect_ratio, plus one entry per answered question (keyed by the question text). */
  settings: Record<string, string>;
}

export interface RunEvent {
  run_id: string;
  step: number; // 1..3
  public_label: PublicLabel;
  phase: Phase;
  ts: string;
}

export interface ImageResult {
  image_id: string;
  url: string; // always a URL: the API never sends bytes or base64
  quality_score: number | null;
}

export interface RunResult {
  run_id: string;
  status: Status;
  summary: string;
  reason: string | null;
  images: ImageResult[]; // best first
  issues: string[]; // done_with_issues
  questions: string[]; // needs_input
  assumptions: string[];
}

export interface Handlers {
  onEvent: (event: RunEvent) => void;
  onResult: (result: RunResult) => void;
  onError: (message: string) => void;
}

export const MOCK = process.env.NEXT_PUBLIC_MOCK !== "0";
const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

function isPublicLabel(value: unknown): value is PublicLabel {
  return PUBLIC_LABELS.some((label) => label === value);
}

function absolute(url: string): string {
  return url.startsWith("/") && !MOCK ? `${BASE}${url}` : url;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return (await response.json()) as T;
}

function normalise(result: RunResult): RunResult {
  return { ...result, images: result.images.map((i) => ({ ...i, url: absolute(i.url) })) };
}

export async function createRun(req: CreateRequest): Promise<{ run_id: string }> {
  return MOCK ? mockCreateRun(req) : request("/runs", { method: "POST", body: JSON.stringify(req) });
}

export async function cancelRun(runId: string): Promise<void> {
  if (MOCK) return mockCancel(runId);
  await request(`/runs/${runId}/cancel`, { method: "POST" });
}

export async function getRun(runId: string): Promise<RunResult> {
  return normalise(MOCK ? await mockGetRun(runId) : await request<RunResult>(`/runs/${runId}`));
}

/** Follow a run. Returns a function that stops following (it does not cancel the run). */
export function subscribe(runId: string, handlers: Handlers): () => void {
  const onEvent = (event: RunEvent) => {
    if (isPublicLabel(event.public_label)) handlers.onEvent(event); // never show anything else
  };
  const onResult = (result: RunResult) => handlers.onResult(normalise(result));

  if (MOCK) return mockSubscribe(runId, { ...handlers, onEvent, onResult });

  const source = new EventSource(`${BASE}/runs/${runId}/events`);
  source.addEventListener("step", (e) => onEvent(JSON.parse((e as MessageEvent).data)));
  source.addEventListener("result", (e) => {
    onResult(JSON.parse((e as MessageEvent).data));
    source.close();
  });
  source.onerror = () => {
    // EventSource retries by itself; only report a connection that is gone for good.
    if (source.readyState === EventSource.CLOSED) handlers.onError("Lost the connection.");
  };
  return () => source.close();
}

// ---------------------------------------------------------------------------------------------
// Developer inspector: GET /runs/:id/trace. A SEPARATE channel from the events and the result
// (the backend serves it only when DEBUG_TRACE=1; the page shows it only when
// NEXT_PUBLIC_INSPECTOR=1). It is the ONE place internal agent names and model ids appear.
// ---------------------------------------------------------------------------------------------

export const INSPECTOR = process.env.NEXT_PUBLIC_INSPECTOR === "1";

export type AttemptOutcome = "ok" | "skipped" | "rate_limited" | "timeout" | "refused" | "error";
export interface GatewayAttempt {
  provider: string;
  model: string;
  outcome: AttemptOutcome;
  reason: string | null;
}

export interface ImageDetail {
  width_requested: number;
  height_requested: number;
  width: number | null;
  height: number | null;
  seed: number | null;
  seed_sent: boolean;
  seed_honoured: boolean | null;
  mime: string | null;
  bytes: number | null;
  steps: number | null;
}

export interface CallRecord {
  seq: number;
  ts: string; // when the call started
  step: "route" | "enhance" | "generate" | "critique" | "supervise";
  role: string; // internal agent name
  attempt: number;
  branch: number;
  kind: "llm" | "image";
  chain: string | null;
  provider: string | null;
  model: string | null;
  was_fallback: boolean;
  attempts: GatewayAttempt[];
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens: number | null;
  total_tokens: number;
  tokens_estimated: boolean;
  reasoning_effort: string | null;
  neurons: number;
  cost_usd: number;
  duration_ms: number;
  was_repair: boolean;
  failure: string | null;
  request_text: string;
  reply_text: string | null;
  reasoning_text: string | null;
  image: ImageDetail | null;
}

export interface TraceDecision {
  seq: number;
  ts: string;
  step: string;
  text: string;
}

export interface TraceOrphan {
  image_id: string;
  reason: string;
  ts: string;
}

export interface TraceSpend {
  tokens: number;
  cost_usd: number;
  duration_ms: number;
  neurons: number;
  llm_calls: number;
}

export interface TraceStep {
  step: number;
  agent: string; // internal role
  public_label: PublicLabel;
  status: "pending" | "running" | "done" | "failed" | "cancelled" | "skipped";
  branches_total: number | null;
  branches_done: number | null;
  branches_aborted: number | null;
  partial_outputs: string[];
  spend: TraceSpend;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number;
}

export interface TraceAttemptRecord {
  number: number;
  branch: number;
  lever: string | null;
  seed: number;
  tier: string;
  chain_offset: number;
  feedback: string | null;
  image_id: string | null;
  provider: string | null;
  verdict: "allow" | "block" | null;
  quality_score: number | null;
  critique: string | null;
  failure: string | null;
}

export interface TraceBudget {
  tokens_total: number;
  cost_cap_usd: number;
  neurons_total: number;
  reserve_fraction: number;
  tokens_spent: number;
  cost_spent_usd: number;
  neurons_spent: number;
}

export interface Trace {
  calls: CallRecord[];
  decisions: TraceDecision[];
  steps: TraceStep[];
  attempts: TraceAttemptRecord[];
  budget: TraceBudget | null;
  supervisor_spend: TraceSpend | null;
  retries_used: number;
  critic_reasks_used: number;
  orphaned: TraceOrphan[];
  status: string; // "running" or the final envelope status
  totals: {
    calls: number;
    llm_calls: number;
    image_calls: number;
    prompt_tokens: number;
    completion_tokens: number;
    reasoning_tokens: number;
    total_tokens: number;
    neurons: number;
    cost_usd: number;
  };
  meta: {
    plan?: string;
    n?: number;
    retry_cap?: number;
    critic_reask_cap?: number;
    quality_threshold?: number;
    reserve_fraction?: number;
    max_parallel_branches?: number;
  };
  created_at: string | null;
  finished_at: string | null;
  wall_ms: number | null;
}

/** The run's trace so far (live while it runs). null when tracing is off or in mock mode. */
export async function getTrace(runId: string): Promise<Trace | null> {
  if (MOCK) return null;
  const response = await fetch(`${BASE}/runs/${runId}/trace`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return (await response.json()) as Trace;
}
