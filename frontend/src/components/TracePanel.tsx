"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { fetchTrace, type Trace, type TraceSpan } from "@/lib/api";
import {
  depthOf,
  fmtMs,
  fmtUsd,
  KIND_COLOR,
  summaryOf,
  TERMINAL,
} from "@/lib/trace";

type Props = { runId: string; token: string };

// DEV ONLY. Every step of a run, from the input to the result: time, model, provider, cost and
// what went in and out. It polls its own endpoint (the user stream carries none of this), and it
// only renders when the server has TRACE_UI on.
export default function TracePanel({ runId, token }: Props) {
  const [trace, setTrace] = useState<Trace | null>(null);
  const [open, setOpen] = useState(true);
  const stop = useRef(false);

  useEffect(() => {
    stop.current = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function tick() {
      const next = await fetchTrace(runId, token);
      if (stop.current) return;
      if (next) setTrace(next);
      const terminal = next && TERMINAL.includes(next.status);
      if (!terminal) timer = setTimeout(tick, next?.status === "waiting" ? 2000 : 700);
    }
    tick();
    return () => {
      stop.current = true;
      if (timer) clearTimeout(timer);
    };
  }, [runId, token]);

  const byId = useMemo(() => new Map((trace?.spans ?? []).map((s) => [s.id, s])), [trace]);
  const { offset, total } = useMemo(() => {
    const spans = trace?.spans ?? [];
    const lo = Math.min(0, ...spans.map((s) => s.start_ms));
    const hi = Math.max(1, ...spans.map((s) => s.start_ms + s.duration_ms));
    return { offset: lo, total: Math.max(1, hi - lo) };
  }, [trace]);
  const slowest = useMemo(() => {
    const leafy = (trace?.spans ?? []).filter((s) => s.kind !== "generate" && s.kind !== "wait");
    return leafy.reduce<TraceSpan | null>((a, s) => (!a || s.duration_ms > a.duration_ms ? s : a), null);
  }, [trace]);

  if (!trace) {
    return <p className="text-xs text-zinc-500">Trace: waiting for the first step…</p>;
  }
  const t = trace.totals;
  const running = !TERMINAL.includes(trace.status);

  return (
    <section
      aria-label="Trace"
      data-testid="trace-panel"
      className="flex flex-col gap-3 rounded-lg border border-zinc-300 bg-zinc-50 p-3 text-xs dark:border-zinc-700 dark:bg-zinc-900"
    >
      <header className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => setOpen((o) => !o)} className="font-semibold">
          {open ? "▾" : "▸"} Trace
        </button>
        <span className="rounded bg-amber-200 px-1.5 py-0.5 text-[10px] font-medium text-amber-900">dev</span>
        <span className="rounded bg-zinc-200 px-1.5 py-0.5 dark:bg-zinc-800">
          {running ? `● ${trace.status}` : trace.status}
        </span>
        <span className="ml-auto flex gap-2">
          <button
            type="button"
            className="underline"
            onClick={() => navigator.clipboard?.writeText(JSON.stringify(trace, null, 2))}
          >
            Copy JSON
          </button>
        </span>
      </header>

      <div className="flex flex-wrap gap-2" data-testid="trace-totals">
        <Chip label="Time" value={fmtMs(t.duration_ms)} />
        <Chip label="Cost" value={fmtUsd(t.total_usd, t.images_cost_estimated)} />
        <Chip label="Images" value={fmtUsd(t.images_usd, t.images_cost_estimated)} />
        <Chip label="LLM (est.)" value={fmtUsd(t.llm_usd_estimate)} />
        <Chip label="Credits" value={`${t.credits_charged} / ${t.credits_held} held`} />
        <Chip label="Neurons" value={String(t.neurons)} />
        <Chip label="Tokens" value={`${t.tokens.prompt} in, ${t.tokens.completion} out`} />
        <Chip label="Delivered" value={`${t.images_delivered} / ${t.images_requested}`} />
      </div>

      {open && (
        <>
          <details className="rounded border border-zinc-200 p-2 dark:border-zinc-800">
            <summary className="cursor-pointer font-medium">Input</summary>
            <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap">
              {JSON.stringify(trace.input, null, 2)}
            </pre>
          </details>

          <ol className="flex flex-col gap-1" data-testid="trace-spans">
            {trace.spans.map((s) => (
              <SpanRow
                key={s.id}
                span={s}
                depth={depthOf(s, byId)}
                offset={offset}
                total={total}
                slowest={slowest?.id === s.id}
              />
            ))}
          </ol>
        </>
      )}
    </section>
  );
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded border border-zinc-200 px-2 py-0.5 dark:border-zinc-700">
      <span className="text-zinc-500">{label} </span>
      <span className="font-medium">{value}</span>
    </span>
  );
}

function SpanRow(props: {
  span: TraceSpan;
  depth: number;
  offset: number;
  total: number;
  slowest: boolean;
}) {
  const { span, depth, offset, total, slowest } = props;
  const left = ((span.start_ms - offset) / total) * 100;
  const width = Math.max(0.6, (span.duration_ms / total) * 100);
  const who = [span.provider, span.model].filter(Boolean).join(" · ");
  const bits = summaryOf(span);
  return (
    <li data-span={span.name} data-status={span.status}>
      <details className="rounded hover:bg-zinc-100 dark:hover:bg-zinc-800">
        <summary className="flex cursor-pointer flex-col gap-1 px-1 py-1">
          <span className="flex items-center gap-2" style={{ paddingLeft: depth * 14 }}>
            <span
              className={`inline-block size-2 rounded-full ${
                span.status === "error"
                  ? "bg-red-500"
                  : span.status === "running"
                    ? "animate-pulse bg-sky-500"
                    : "bg-emerald-500"
              }`}
              title={span.status}
            />
            <span className="font-medium">{span.name}</span>
            {who && <span className="text-zinc-500">{who}</span>}
            <span className="ml-auto tabular-nums">
              {fmtMs(span.duration_ms)}
              {slowest && <span className="ml-1 rounded bg-orange-200 px-1 text-[10px] text-orange-900">slowest</span>}
            </span>
          </span>
          <span className="relative h-1.5 rounded bg-zinc-200 dark:bg-zinc-800" aria-hidden>
            <span
              className={`absolute top-0 h-1.5 rounded ${KIND_COLOR[span.kind] ?? "bg-zinc-400"}`}
              style={{ left: `${left}%`, width: `${width}%` }}
            />
          </span>
          {bits.length > 0 && (
            <span className="text-zinc-500" style={{ paddingLeft: depth * 14 + 16 }}>
              {bits.join("  ·  ")}
            </span>
          )}
          {span.error && (
            <span className="text-red-600" style={{ paddingLeft: depth * 14 + 16 }}>
              {span.error}
            </span>
          )}
        </summary>
        <div className="grid gap-2 px-2 pb-2 md:grid-cols-2">
          <Json title="Input" value={span.inputs} />
          <Json title="Output" value={span.outputs} />
          {Object.keys(span.metadata).length > 0 && <Json title="Metadata" value={span.metadata} />}
        </div>
      </details>
    </li>
  );
}

function Json({ title, value }: { title: string; value: unknown }) {
  return (
    <div>
      <p className="mb-1 font-medium text-zinc-500">{title}</p>
      <pre className="max-h-56 overflow-auto whitespace-pre-wrap rounded bg-zinc-100 p-2 dark:bg-zinc-800">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}
