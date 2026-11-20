"use client";

import { useState } from "react";
import type { CallRecord, Trace, TraceBudget, TraceStep } from "@/lib/api";
import { ms, num, since, usd } from "@/lib/format";

/**
 * Developer inspector. Everything about a run: steps, every model call and which model served
 * it, tokens, reasoning, retries, budget and decisions. It reads GET /runs/:id/trace, a channel
 * apart from the public events, and is the ONE place internal names and model ids are shown.
 */
export default function Inspector({ trace, running }: { trace: Trace | null; running: boolean }) {
  if (!trace) {
    return (
      <p className="rounded border border-neutral-300 bg-white p-4 text-sm text-neutral-600">
        No trace yet. It needs the API running with <code>DEBUG_TRACE=1</code> and{" "}
        <code>NEXT_PUBLIC_MOCK=0</code> (mock mode has no trace).
      </p>
    );
  }
  return (
    <div className="space-y-6 text-sm">
      <SummaryBar trace={trace} running={running} />
      <Budget trace={trace} />
      <Providers calls={trace.calls} />
      <Timeline trace={trace} />
      <StepCards steps={trace.steps} />
      <CallsTable calls={trace.calls} />
      <ImageDetails calls={trace.calls} />
      <Attempts trace={trace} />
      <Decisions trace={trace} />
    </div>
  );
}

const panel = "rounded-lg border border-neutral-200 bg-white p-4";
const h = "mb-3 text-base font-semibold text-neutral-900";
const th = "px-2 py-1 text-left text-xs font-medium uppercase text-neutral-500";
const td = "px-2 py-1 align-top";

function Badge({ children, tone = "gray" }: { children: React.ReactNode; tone?: "gray" | "green" | "red" | "amber" | "blue" }) {
  const tones = {
    gray: "bg-neutral-100 text-neutral-700",
    green: "bg-emerald-100 text-emerald-800",
    red: "bg-red-100 text-red-800",
    amber: "bg-amber-100 text-amber-900",
    blue: "bg-indigo-100 text-indigo-800",
  };
  return <span className={`inline-block rounded px-1.5 py-0.5 text-xs font-medium ${tones[tone]}`}>{children}</span>;
}

function statusTone(status: string): "gray" | "green" | "red" | "amber" | "blue" {
  if (status === "done") return "green";
  if (status === "running") return "blue";
  if (status === "failed" || status === "rejected") return "red";
  if (status === "cancelled" || status === "done_with_issues" || status === "skipped") return "amber";
  return "gray";
}

// ------------------------------------------------------------------------------ summary ------

function SummaryBar({ trace, running }: { trace: Trace; running: boolean }) {
  const t = trace.totals;
  const m = trace.meta;
  const items: [string, React.ReactNode][] = [
    ["Status", <Badge key="s" tone={statusTone(trace.status)}>{trace.status}</Badge>],
    ["Wall time", <span key="w">{ms(trace.wall_ms)}{running ? " …" : ""}</span>],
    ["LLM calls", `${t.llm_calls} (+${t.image_calls} image)`],
    ["Tokens", `${num(t.prompt_tokens)} in / ${num(t.completion_tokens)} out / ${num(t.reasoning_tokens)} reasoning / ${num(t.total_tokens)} total`],
    ["Neurons", num(t.neurons, 1)],
    ["USD", usd(t.cost_usd)],
    ["Retries", `${trace.retries_used} of ${m.retry_cap ?? "?"}`],
    ["Critic re-asks", `${trace.critic_reasks_used} of ${m.critic_reask_cap ?? "?"}`],
  ];
  return (
    <section className={panel}>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
        {items.map(([label, value]) => (
          <div key={label}>
            <dt className="text-xs uppercase text-neutral-500">{label}</dt>
            <dd className="font-medium">{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

// ------------------------------------------------------------------------------- budget ------

function Meter({ label, spent, total, reserve, fmt }: { label: string; spent: number; total: number; reserve: number; fmt: (v: number) => string }) {
  if (total <= 0) {
    return (
      <div>
        <div className="mb-1 flex justify-between"><span className="font-medium">{label}</span><span>{fmt(spent)} spent, no cap</span></div>
      </div>
    );
  }
  const usable = total * (1 - reserve);
  const touched = spent > usable;
  const pct = Math.min(100, (spent / total) * 100);
  return (
    <div>
      <div className="mb-1 flex flex-wrap justify-between gap-2">
        <span className="font-medium">{label}</span>
        <span>
          {fmt(spent)} of {fmt(total)} <Badge tone={touched ? "red" : "green"}>{touched ? "reserve touched" : "reserve untouched"}</Badge>
        </span>
      </div>
      <div className="relative h-3 rounded bg-neutral-200">
        <div className={`h-3 rounded ${touched ? "bg-red-500" : "bg-indigo-500"}`} style={{ width: `${pct}%` }} />
        <div
          className="absolute top-[-3px] h-[18px] w-0.5 bg-neutral-900"
          style={{ left: `${(1 - reserve) * 100}%` }}
          title={`${Math.round(reserve * 100)}% reserve starts here (kept for retries)`}
        />
      </div>
      <div className="mt-0.5 text-xs text-neutral-500">
        marker: where the {Math.round(reserve * 100)}% reserve for retries begins ({fmt(usable)})
      </div>
    </div>
  );
}

function Budget({ trace }: { trace: Trace }) {
  const b: TraceBudget | null = trace.budget;
  if (!b) return null;
  return (
    <section className={panel}>
      <h3 className={h}>Budget</h3>
      <div className="space-y-4">
        <Meter label="Tokens" spent={b.tokens_spent} total={b.tokens_total} reserve={b.reserve_fraction} fmt={(v) => num(v)} />
        <Meter label="Neurons" spent={b.neurons_spent} total={b.neurons_total} reserve={b.reserve_fraction} fmt={(v) => num(v, 1)} />
        <Meter label="USD" spent={b.cost_spent_usd} total={b.cost_cap_usd} reserve={b.reserve_fraction} fmt={usd} />
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------- providers ------

// Free-tier limits the provider split is about (CLAUDE.md, "Model provider strategy"):
// Cloudflare's neurons are ONE daily pool shared by LLM and image calls; Groq limits are PER MODEL.
const CF_DAILY_NEURONS = 10_000;
const GROQ_TOKENS_PER_MINUTE = 8_000;

// The dev fake backend reports e.g. "groq (fake)": group it with the real provider.
function providerOf(c: CallRecord): string {
  return (c.provider ?? "none").replace(/ \(fake\)$/, "");
}

function Providers({ calls }: { calls: CallRecord[] }) {
  const served = calls.filter((c) => c.model);
  if (served.length === 0) return null;

  type Row = { provider: string; model: string; kinds: Set<string>; roles: Set<string>; calls: number; tokens: number; neurons: number; usd: number };
  const rows = new Map<string, Row>();
  for (const c of served) {
    const key = `${providerOf(c)}/${c.model}`;
    const r = rows.get(key) ?? { provider: providerOf(c), model: c.model ?? "", kinds: new Set(), roles: new Set(), calls: 0, tokens: 0, neurons: 0, usd: 0 };
    r.kinds.add(c.kind);
    r.roles.add(c.role);
    r.calls += 1;
    r.tokens += c.kind === "llm" ? c.total_tokens : 0;
    r.neurons += c.neurons;
    r.usd += c.cost_usd;
    rows.set(key, r);
  }

  const cf = served.filter((c) => providerOf(c) === "cloudflare");
  const cfAgents = cf.filter((c) => c.kind === "llm").reduce((s, c) => s + c.neurons, 0);
  const cfImages = cf.filter((c) => c.kind === "image").reduce((s, c) => s + c.neurons, 0);
  const cfTotal = cfAgents + cfImages;
  const groq = [...rows.values()].filter((r) => r.provider === "groq");

  return (
    <section className={panel}>
      <h3 className={h}>Providers</h3>
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <div className="mb-1 font-medium">Cloudflare neuron pool (shared by agents and images)</div>
          {cfTotal === 0 ? (
            <p className="text-neutral-500">No Cloudflare neurons used this run.</p>
          ) : (
            <>
              <div className="flex h-3 overflow-hidden rounded bg-neutral-200" title="agents vs images">
                <div className="h-3 bg-amber-500" style={{ width: `${(cfAgents / cfTotal) * 100}%` }} />
                <div className="h-3 bg-indigo-500" style={{ width: `${(cfImages / cfTotal) * 100}%` }} />
              </div>
              <div className="mt-1 flex flex-wrap gap-x-4 text-xs">
                <span><span className="text-amber-700">■</span> agents {num(cfAgents, 1)} ({Math.round((cfAgents / cfTotal) * 100)}%)</span>
                <span><span className="text-indigo-700">■</span> images {num(cfImages, 1)} ({Math.round((cfImages / cfTotal) * 100)}%)</span>
              </div>
            </>
          )}
          <p className="mt-1 text-xs text-neutral-500">
            {num(cfTotal, 1)} of {num(CF_DAILY_NEURONS)} neurons/day ({((cfTotal / CF_DAILY_NEURONS) * 100).toFixed(2)}%).
            {cfAgents > 0 && " Agent calls on Cloudflare spend image budget: a Groq agent was unavailable (see FALLBACK)."}
          </p>
        </div>
        <div>
          <div className="mb-1 font-medium">Groq tokens per model (limit {num(GROQ_TOKENS_PER_MINUTE)}/min each)</div>
          {groq.length === 0 ? (
            <p className="text-neutral-500">No Groq calls this run.</p>
          ) : (
            <ul className="space-y-2">
              {groq.map((r) => {
                const pct = Math.min(100, (r.tokens / GROQ_TOKENS_PER_MINUTE) * 100);
                return (
                  <li key={r.model}>
                    <div className="flex justify-between gap-2 text-xs">
                      <span>{r.model} <span className="text-neutral-500">({[...r.roles].join(", ")})</span></span>
                      <span>{num(r.tokens)} ({pct.toFixed(0)}%)</span>
                    </div>
                    <div className="h-2 rounded bg-neutral-200">
                      <div className={`h-2 rounded ${pct > 50 ? "bg-red-500" : "bg-emerald-500"}`} style={{ width: `${pct}%` }} />
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
          <p className="mt-1 text-xs text-neutral-500">
            Share of one minute&apos;s budget used by this run alone. Red above 50%: two such runs in a minute hit the limit.
          </p>
        </div>
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[640px] text-xs">
          <thead>
            <tr>{["Provider", "Model", "Used by", "Calls", "LLM tokens", "Neurons", "USD"].map((c) => <th key={c} className={th}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {[...rows.values()].map((r) => (
              <tr key={`${r.provider}/${r.model}`} className="border-t border-neutral-200">
                <td className={td}>
                  <Badge tone={r.provider === "groq" ? "green" : r.provider === "cloudflare" ? "blue" : r.usd > 0 ? "red" : "gray"}>{r.provider}</Badge>
                </td>
                <td className={td}>{r.model}</td>
                <td className={td}>{[...r.roles].join(", ")}</td>
                <td className={td}>{r.calls}</td>
                <td className={td}>{r.kinds.has("llm") ? num(r.tokens) : "–"}</td>
                <td className={td}>{r.provider === "cloudflare" ? num(r.neurons, 1) : "–"}</td>
                <td className={td}>{r.usd > 0 ? usd(r.usd) : r.provider === "openrouter" ? usd(0) : "free"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ----------------------------------------------------------------------------- timeline ------

type Entry = { ts: string; sort: number; kind: "call" | "decision" | "step"; text: React.ReactNode; tone?: string };

function Timeline({ trace }: { trace: Trace }) {
  const start = trace.created_at;
  const entries: Entry[] = [];
  for (const s of trace.steps) {
    if (s.started_at) {
      entries.push({
        ts: s.started_at,
        sort: 0,
        kind: "step",
        text: (
          <span>
            <b>Step {s.step}</b> {s.public_label} <Badge>{s.agent}</Badge> {s.status}
            {s.finished_at ? ` · took ${ms(s.duration_ms)} in total` : " · running"}
          </span>
        ),
      });
    }
  }
  for (const c of trace.calls) {
    entries.push({
      ts: c.ts,
      sort: 1,
      kind: "call",
      text: (
        <span>
          call #{c.seq} <Badge tone="blue">{c.step}</Badge> {c.role} →{" "}
          {c.model ? `${c.provider} / ${c.model}` : "no model served it"}
          {c.was_fallback && <> <Badge tone="amber">FALLBACK</Badge></>}
          {c.failure && <> <Badge tone="red">{c.failure}</Badge></>} · {ms(c.duration_ms)}
          {c.kind === "llm" && ` · ${num(c.total_tokens)} tokens`}
        </span>
      ),
    });
  }
  for (const d of trace.decisions) {
    const retry = d.step === "supervise" || d.text.includes("reused") || d.text.includes("re-ask");
    entries.push({ ts: d.ts, sort: 2, kind: "decision", tone: retry ? "retry" : undefined, text: <span><Badge tone={retry ? "amber" : "gray"}>{d.step}</Badge> {d.text}</span> });
  }
  entries.sort((a, b) => a.ts.localeCompare(b.ts) || a.sort - b.sort);
  return (
    <section className={panel}>
      <h3 className={h}>Timeline {trace.wall_ms !== null && <span className="font-normal text-neutral-500">· {ms(trace.wall_ms)} overall</span>}</h3>
      <ol className="max-h-96 space-y-1 overflow-auto">
        {entries.map((e, i) => (
          <li key={i} className={`flex gap-3 rounded px-1 py-0.5 ${e.tone === "retry" ? "bg-amber-50" : ""}`}>
            <span className="w-16 shrink-0 font-mono text-xs text-neutral-500">{since(start, e.ts)}</span>
            <span className="w-4 shrink-0 text-neutral-400">{e.kind === "call" ? "●" : e.kind === "step" ? "▸" : "·"}</span>
            <span className="min-w-0 break-words">{e.text}</span>
          </li>
        ))}
      </ol>
      <p className="mt-2 text-xs text-neutral-500">Amber rows are retries and decisions: the lever chosen and why, a fallback, and steps reused (done, no cost).</p>
    </section>
  );
}

// ------------------------------------------------------------------------------ steps -------

function StepCards({ steps }: { steps: TraceStep[] }) {
  return (
    <section className={panel}>
      <h3 className={h}>Steps</h3>
      <div className="grid gap-3 sm:grid-cols-3">
        {steps.map((s) => (
          <div key={s.step} className="rounded border border-neutral-200 p-3">
            <div className="mb-1 flex items-center justify-between">
              <b>{s.step}. {s.public_label}</b>
              <Badge tone={statusTone(s.status)}>{s.status}</Badge>
            </div>
            <div className="text-xs text-neutral-500">role: {s.agent}</div>
            <dl className="mt-2 space-y-0.5">
              <Row k="Time" v={`${ms(s.duration_ms)}${s.started_at ? "" : " (not started)"}`} />
              <Row k="LLM calls" v={String(s.spend.llm_calls)} />
              <Row k="Tokens" v={num(s.spend.tokens)} />
              <Row k="Neurons" v={num(s.spend.neurons, 1)} />
              <Row k="USD" v={usd(s.spend.cost_usd)} />
              {s.branches_total !== null && (
                <Row k="Branches" v={`${s.branches_total} total / ${s.branches_done ?? 0} done / ${s.branches_aborted ?? 0} aborted`} />
              )}
              {s.partial_outputs.length > 0 && <Row k="Partial outputs" v={s.partial_outputs.join(", ")} />}
            </dl>
          </div>
        ))}
      </div>
    </section>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-neutral-500">{k}</dt>
      <dd className="text-right">{v}</dd>
    </div>
  );
}

// ------------------------------------------------------------------------------- calls ------

function CallsTable({ calls }: { calls: CallRecord[] }) {
  const [open, setOpen] = useState<Set<number>>(new Set());
  const toggle = (seq: number) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(seq)) next.delete(seq);
      else next.add(seq);
      return next;
    });
  return (
    <section className={panel}>
      <h3 className={h}>Model calls ({calls.length})</h3>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-xs">
          <thead>
            <tr>
              {["#", "Step", "Role", "Chain", "Served by", "Tokens in / out / reasoning / total", "Effort", "Neurons", "USD", "Time", "Flags"].map((c) => (
                <th key={c} className={th}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {calls.map((c) => {
              const est = c.tokens_estimated ? "~" : "";
              return (
                <FragmentRow key={c.seq}>
                  <tr className="cursor-pointer border-t border-neutral-200 hover:bg-neutral-50" onClick={() => toggle(c.seq)}>
                    <td className={td}>{open.has(c.seq) ? "▾" : "▸"} {c.seq}</td>
                    <td className={td}>{c.step}{c.attempt > 0 ? ` r${c.attempt}${c.branch ? `/b${c.branch}` : ""}` : ""}</td>
                    <td className={td}>{c.role}</td>
                    <td className={td}>{c.chain ?? "–"}</td>
                    <td className={td}>
                      {c.model ? `${c.provider} / ${c.model}` : "–"} {c.was_fallback && <Badge tone="amber">FALLBACK</Badge>}
                    </td>
                    <td className={td}>
                      {c.kind === "image" ? "–" : `${est}${c.prompt_tokens} / ${est}${c.completion_tokens} / ${c.reasoning_tokens ?? "n/r"} / ${est}${c.total_tokens}`}
                      {c.tokens_estimated && <span className="ml-1 text-amber-700">(estimated)</span>}
                    </td>
                    <td className={td}>{c.reasoning_effort ?? "–"}</td>
                    <td className={td}>{providerOf(c) === "cloudflare" ? num(c.neurons, 1) : "–"}</td>
                    <td className={td}>{usd(c.cost_usd)}</td>
                    <td className={td}>{ms(c.duration_ms)}</td>
                    <td className={td}>
                      {c.was_repair && <Badge tone="amber">repair</Badge>} {c.failure && <Badge tone="red">{c.failure}</Badge>}
                    </td>
                  </tr>
                  {open.has(c.seq) && (
                    <tr className="bg-neutral-50">
                      <td colSpan={11} className="px-3 py-3">
                        <CallDetail call={c} />
                      </td>
                    </tr>
                  )}
                </FragmentRow>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function FragmentRow({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

function CallDetail({ call }: { call: CallRecord }) {
  const box = "max-h-64 overflow-auto whitespace-pre-wrap rounded border border-neutral-200 bg-white p-2 font-mono text-xs";
  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1 font-medium">Gateway attempts (who was tried, in order)</div>
        {call.attempts.length === 0 ? (
          <span className="text-neutral-500">none recorded</span>
        ) : (
          <ol className="list-inside list-decimal">
            {call.attempts.map((a, i) => (
              <li key={i}>
                {a.provider} / {a.model}{" "}
                <Badge tone={a.outcome === "ok" ? "green" : a.outcome === "skipped" ? "gray" : "red"}>{a.outcome}</Badge>
                {a.reason ? ` ${a.reason}` : ""}
              </li>
            ))}
          </ol>
        )}
      </div>
      <div>
        <div className="mb-1 font-medium">Request</div>
        <pre className={box}>{call.request_text || "(empty)"}</pre>
      </div>
      <div>
        <div className="mb-1 font-medium">Reply</div>
        <pre className={box}>{call.reply_text ?? "(no reply)"}</pre>
      </div>
      <div>
        <div className="mb-1 font-medium">Reasoning</div>
        {call.reasoning_text ? <pre className={box}>{call.reasoning_text}</pre> : <span className="text-neutral-500">not reported by the provider</span>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------- images ------

function ImageDetails({ calls }: { calls: CallRecord[] }) {
  const images = calls.filter((c) => c.kind === "image");
  if (images.length === 0) return null;
  return (
    <section className={panel}>
      <h3 className={h}>Image details</h3>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[700px] text-xs">
          <thead>
            <tr>{["Call", "Round / branch", "Provider / model", "Seed sent", "Seed honoured", "Requested", "Returned", "Mime", "Bytes", "Steps"].map((c) => <th key={c} className={th}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {images.map((c) => {
              const i = c.image;
              return (
                <tr key={c.seq} className="border-t border-neutral-200">
                  <td className={td}>#{c.seq}</td>
                  <td className={td}>r{c.attempt} / b{c.branch}</td>
                  <td className={td}>{c.model ? `${c.provider} / ${c.model}` : "–"} {c.failure && <Badge tone="red">{c.failure}</Badge>}</td>
                  <td className={td}>{i ? (i.seed_sent ? String(i.seed) : `not sent (${i.seed} asked)`) : "–"}</td>
                  <td className={td}>{i ? (i.seed_honoured === null ? "unverified" : i.seed_honoured ? "yes" : "no") : "–"}</td>
                  <td className={td}>{i ? `${i.width_requested}×${i.height_requested}` : "–"}</td>
                  <td className={td}>{i && i.width ? `${i.width}×${i.height}` : "–"}</td>
                  <td className={td}>{i?.mime ?? "–"}</td>
                  <td className={td}>{i?.bytes ? num(i.bytes) : "–"}</td>
                  <td className={td}>{i?.steps ?? "provider default"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ----------------------------------------------------------------------------- attempts ------

function Attempts({ trace }: { trace: Trace }) {
  return (
    <section className={panel}>
      <h3 className={h}>Attempts</h3>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-xs">
          <thead>
            <tr>{["Round", "Branch", "Lever", "Seed", "Tier", "Chain offset", "Feedback sent to the Enhancer", "Score", "Verdict", "Critique", "Failure"].map((c) => <th key={c} className={th}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {trace.attempts.map((a, i) => (
              <tr key={i} className="border-t border-neutral-200">
                <td className={td}>{a.number}</td>
                <td className={td}>{a.branch}</td>
                <td className={td}>{a.lever ?? "first"}</td>
                <td className={td}>{a.seed}</td>
                <td className={td}>{a.tier}</td>
                <td className={td}>{a.chain_offset}</td>
                <td className={`${td} max-w-xs`}>{a.feedback ?? "–"}</td>
                <td className={td}>{a.quality_score ?? "–"}</td>
                <td className={td}>{a.verdict ? <Badge tone={a.verdict === "allow" ? "green" : "red"}>{a.verdict}</Badge> : "–"}</td>
                <td className={`${td} max-w-xs`}>{a.critique ?? "–"}</td>
                <td className={td}>{a.failure ? <Badge tone="red">{a.failure}</Badge> : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ----------------------------------------------------------------------------- decisions -----

function Decisions({ trace }: { trace: Trace }) {
  return (
    <section className={panel}>
      <h3 className={h}>Decisions</h3>
      <ol className="max-h-96 space-y-1 overflow-auto">
        {trace.decisions.map((d) => (
          <li key={d.seq} className="flex gap-2">
            <span className="w-8 shrink-0 text-right font-mono text-xs text-neutral-400">{d.seq}</span>
            <span className="w-20 shrink-0"><Badge>{d.step}</Badge></span>
            <span className="min-w-0 break-words">{d.text}</span>
          </li>
        ))}
      </ol>
      <h4 className="mb-1 mt-4 font-medium">Orphaned images</h4>
      {trace.orphaned.length === 0 ? (
        <p className="text-neutral-500">none</p>
      ) : (
        <ul className="list-inside list-disc">
          {trace.orphaned.map((o) => (
            <li key={o.image_id + o.reason}>{o.image_id} <Badge tone="amber">{o.reason}</Badge></li>
          ))}
        </ul>
      )}
    </section>
  );
}
