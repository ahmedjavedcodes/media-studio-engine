// Helpers for the dev trace panel. Pure functions, so they are easy to test.
import type { TraceSpan } from "@/lib/api";

export const TERMINAL = ["done", "failed", "expired"];

export function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

export function fmtUsd(usd: number, estimated = false): string {
  const text = usd === 0 ? "$0" : usd < 0.001 ? `$${usd.toFixed(5)}` : `$${usd.toFixed(4)}`;
  return estimated ? `≈ ${text}` : text;
}

// How deep a span sits in the tree (generate > generate_image > safety).
export function depthOf(span: TraceSpan, byId: Map<number, TraceSpan>): number {
  let depth = 0;
  let cur = span.parent !== null ? byId.get(span.parent) : undefined;
  while (cur && depth < 6) {
    depth += 1;
    cur = cur.parent !== null ? byId.get(cur.parent) : undefined;
  }
  return depth;
}

// The numbers worth showing on a span's row, by what the span is.
export function summaryOf(span: TraceSpan): string[] {
  const o = span.outputs as Record<string, unknown>;
  const out: string[] = [];
  const num = (v: unknown) => (typeof v === "number" ? v : undefined);
  if (span.kind === "image") {
    const cost = num(o.cost_usd);
    const est = num(o.cost_estimate_usd);
    out.push(cost !== undefined ? fmtUsd(cost) : est !== undefined ? fmtUsd(est, true) : "");
    if (o.size) out.push(String(o.size));
    if (num(o.bytes)) out.push(`${Math.round((num(o.bytes) as number) / 1024)} KB`);
    if (o.provider_fallback) out.push("fallback");
  } else if (span.kind === "llm" || span.kind === "safety") {
    if (num(o.prompt_tokens) !== undefined || num(o.completion_tokens) !== undefined) {
      out.push(`${num(o.prompt_tokens) ?? 0}+${num(o.completion_tokens) ?? 0} tok`);
    }
    if (num(o.neurons) !== undefined) out.push(`${num(o.neurons)} neurons`);
    if (o.verdict) out.push(String(o.verdict));
    if (o.skipped) out.push("skipped");
  } else if (span.kind === "router") {
    if (o.category) out.push(String(o.category));
    if (num(o.held) !== undefined) out.push(`held ${o.held} cr`);
  } else if (span.kind === "settle") {
    if (num(o.delivered) !== undefined) out.push(`${o.delivered} delivered`);
    if (num(o.charged) !== undefined) out.push(`charged ${o.charged} cr`);
  } else if (span.kind === "guardrail") {
    if (o.decision) out.push(String(o.decision));
  } else if (span.kind === "check" && span.name === "code_guard") {
    out.push(o.prompt_changed ? "prompt changed" : "prompt kept");
    if (num(o.words) !== undefined) out.push(`${o.words} words`);
  } else if (span.kind === "wait") {
    out.push("waiting for the user");
  }
  return out.filter(Boolean);
}

export const KIND_COLOR: Record<string, string> = {
  guardrail: "bg-rose-400",
  check: "bg-zinc-400",
  router: "bg-violet-400",
  llm: "bg-sky-400",
  safety: "bg-amber-400",
  wait: "bg-zinc-300",
  generate: "bg-emerald-300",
  image: "bg-emerald-500",
  settle: "bg-teal-400",
};
