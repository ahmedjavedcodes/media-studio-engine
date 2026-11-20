"use client";

/**
 * Run history: DEVELOPER VIEW ONLY (NEXT_PUBLIC_INSPECTOR=1, API with DEBUG_TRACE=1). Recent
 * finished runs from GET /history, and one run in full from GET /history/:id: the input, the
 * route decisions, every call (agent, model, tokens/neurons, cost, latency) through the existing
 * Inspector, and the output (images, scores and ranking, the documents). Reasoning text stays in
 * this view; it is never part of the product UI.
 */
import { useEffect, useState } from "react";
import Inspector from "@/components/Inspector";
import { getHistory, getHistoryRun, imageUrl, type HistoryRecord, type HistoryRow } from "@/lib/api";
import { ms, usd } from "@/lib/format";

const cell = "px-2 py-1.5 text-left align-top";

export default function RunHistory({ refreshKey }: { refreshKey: string | null }) {
  const [rows, setRows] = useState<HistoryRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<HistoryRecord | null>(null);
  const [visible, setVisible] = useState(false);

  // Reload when shown, and after each run finishes (refreshKey changes).
  useEffect(() => {
    if (!visible) return;
    let alive = true;
    getHistory()
      .then((next) => alive && (setRows(next), setError(null)))
      .catch((e) => alive && setError(e instanceof Error ? e.message : "Could not load the history."));
    return () => {
      alive = false;
    };
  }, [visible, refreshKey]);

  async function show(runId: string) {
    try {
      setOpen(await getHistoryRun(runId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load that run.");
    }
  }

  return (
    <section className="space-y-3">
      <button
        onClick={() => setVisible((v) => !v)}
        className="rounded-md border border-neutral-300 bg-white px-3 py-1.5 text-sm font-medium"
      >
        {visible ? "Hide run history" : "Show run history"}
      </button>
      {visible && (
        <div className="space-y-4 rounded-lg border border-neutral-200 bg-white p-4">
          <h2 className="text-lg font-semibold">
            Run history <span className="text-sm font-normal text-neutral-500">(developer view)</span>
          </h2>
          {error && <p className="text-sm text-red-700">{error}</p>}
          {rows === null && !error && (
            <p className="text-sm text-neutral-600">
              No history. It needs the API with <code>DEBUG_TRACE=1</code> and <code>NEXT_PUBLIC_MOCK=0</code>.
            </p>
          )}
          {rows && rows.length === 0 && <p className="text-sm text-neutral-600">No finished runs yet.</p>}
          {rows && rows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-neutral-200 text-neutral-500">
                  <tr>
                    <th className={cell}>When</th>
                    <th className={cell}>Prompt</th>
                    <th className={cell}>Status</th>
                    <th className={cell}>Plan · op · n</th>
                    <th className={cell}>Model</th>
                    <th className={cell}>Cost</th>
                    <th className={cell}>Calls</th>
                    <th className={cell}>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr
                      key={r.run_id}
                      onClick={() => void show(r.run_id)}
                      className={
                        "cursor-pointer border-b border-neutral-100 hover:bg-neutral-50 " +
                        (open?.run_id === r.run_id ? "bg-indigo-50" : "")
                      }
                    >
                      <td className={cell}>{r.created_at ? new Date(r.created_at).toLocaleString() : "—"}</td>
                      <td className={cell + " max-w-xs truncate"} title={r.prompt}>
                        {r.prompt}
                      </td>
                      <td className={cell}>{r.status}</td>
                      <td className={cell}>
                        {r.plan} · {r.operation} · {r.n}
                      </td>
                      <td className={cell}>
                        <div>{r.model_pick}</div>
                        {r.models.length > 0 && <div className="text-xs text-neutral-500">{r.models.join(", ")}</div>}
                      </td>
                      <td className={cell}>
                        {usd(r.cost_usd)}
                        {r.neurons > 0 && <div className="text-xs text-neutral-500">{r.neurons.toFixed(1)} neurons</div>}
                      </td>
                      <td className={cell}>{r.llm_calls} LLM</td>
                      <td className={cell}>{r.wall_ms !== null ? ms(r.wall_ms) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {open && <RunDetail record={open} />}
        </div>
      )}
    </section>
  );
}

function RunDetail({ record }: { record: HistoryRecord }) {
  const { request, result, output, trace } = record;
  const documents = output.documents ?? [];
  const route = trace.decisions.filter((d) => d.step === "route");
  return (
    <div className="space-y-4 border-t border-neutral-200 pt-4">
      <h3 className="font-semibold">
        Run <code className="text-sm">{record.run_id}</code>
      </h3>
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <h4 className="mb-1 text-sm font-medium text-neutral-500">Input</h4>
          <dl className="grid grid-cols-[8rem_1fr] gap-x-2 text-sm">
            <dt>Prompt</dt>
            <dd className="break-words">{request.prompt}</dd>
            <dt>Plan · mode</dt>
            <dd>
              {request.plan} · {request.mode}
            </dd>
            <dt>Operation · n</dt>
            <dd>
              {request.operation ?? "create"} · {request.n}
            </dd>
            <dt>Model</dt>
            <dd>{request.model ?? "auto"}</dd>
            <dt>Settings</dt>
            <dd className="break-words">{JSON.stringify(request.settings)}</dd>
          </dl>
        </div>
        <div>
          <h4 className="mb-1 text-sm font-medium text-neutral-500">Route</h4>
          <ul className="space-y-1 text-sm">
            {route.map((d) => (
              <li key={d.seq}>{d.text}</li>
            ))}
            {route.length === 0 && <li className="text-neutral-500">No route decisions recorded.</li>}
          </ul>
        </div>
      </div>
      <div>
        <h4 className="mb-1 text-sm font-medium text-neutral-500">
          Output · {result.status}
          {result.reason ? ` (${result.reason})` : ""}
        </h4>
        {documents.length > 0 ? (
          <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {documents.map((doc) => {
              const base = doc.layers.find((l) => l.type === "image");
              return (
                <li key={doc.document_id} className="overflow-hidden rounded border border-neutral-200 text-xs">
                  {base?.image_id && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={imageUrl(base.image_id)} alt={`Rank ${doc.rank}`} className="aspect-square w-full object-cover" />
                  )}
                  <div className="space-y-0.5 p-2">
                    <div className="font-medium">Rank {doc.rank} · {doc.layers.length} layer(s)</div>
                    <div className="text-neutral-600">
                      {Object.entries(doc.scores)
                        .map(([k, v]) => `${k} ${v}`)
                        .join(" · ")}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="text-sm text-neutral-600">{result.summary}</p>
        )}
      </div>
      <Inspector trace={trace} running={false} />
    </div>
  );
}
