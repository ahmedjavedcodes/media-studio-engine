"use client";

import { useEffect, useRef, useState } from "react";
import Inspector from "@/components/Inspector";
import ResultView from "@/components/ResultView";
import Stepper, { type StepState } from "@/components/Stepper";
import {
  cancelRun,
  createRun,
  getTrace,
  imageUrl,
  INSPECTOR,
  MOCK,
  subscribe,
  type AspectRatio,
  type CreateRequest,
  type Mode,
  type Plan,
  type RunEvent,
  type RunResult,
  type Trace,
} from "@/lib/api";

const PLANS: Plan[] = ["free", "pro", "max", "ultra"];
const MODES: Mode[] = ["auto", "manual"];
const RATIOS: AspectRatio[] = ["1:1", "4:3", "3:4", "16:9", "9:16"];
const select = "w-full rounded-md border border-neutral-300 bg-white px-3 py-2";

type Phase = "idle" | "running" | "finished";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [plan, setPlan] = useState<Plan>("free");
  const [mode, setMode] = useState<Mode>("auto");
  const [n, setN] = useState(1);
  const [ratio, setRatio] = useState<AspectRatio>("1:1");

  const [phase, setPhase] = useState<Phase>("idle");
  const [steps, setSteps] = useState<StepState[]>(["pending", "pending", "pending"]);
  const [progress, setProgress] = useState<number[]>([0, 0, 0]);
  const [result, setResult] = useState<RunResult | null>(null);
  // Images streamed while the run is still going (image_ready events), in arrival order.
  const [streamed, setStreamed] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [currentRun, setCurrentRun] = useState<string | null>(null);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [showInspector, setShowInspector] = useState(false);

  const runId = useRef<string | null>(null);
  const stop = useRef<(() => void) | null>(null);
  const last = useRef<CreateRequest | null>(null);

  useEffect(() => () => stop.current?.(), []);

  // Inspector (NEXT_PUBLIC_INSPECTOR=1): poll the run's trace every 500 ms while it runs, then
  // once more when it ends. A separate channel from the events: the public UI never reads it.
  useEffect(() => {
    if (!INSPECTOR || !currentRun) return;
    let alive = true;
    const poll = async () => {
      try {
        const next = await getTrace(currentRun);
        if (alive) setTrace(next);
      } catch {
        /* the inspector is best effort */
      }
    };
    void poll();
    if (phase !== "running") return () => { alive = false; };
    const timer = setInterval(poll, 500);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [currentRun, phase]);

  function onEvent(event: RunEvent) {
    const i = event.step - 1;
    if (i < 0 || i > 2) return;
    if (event.phase === "image_ready") {
      const id = event.image_id;
      if (id) setStreamed((s) => (s.includes(id) ? s : [...s, id]));
      return;
    }
    if (event.phase === "progress") {
      setProgress((p) => p.map((v, k) => (k === i ? v + 1 : v)));
    } else {
      setSteps((s) => s.map((v, k) => (k === i ? (event.phase === "started" ? "active" : "done") : v)));
    }
  }

  async function start(req: CreateRequest) {
    stop.current?.();
    last.current = req;
    setPhase("running");
    setSteps(["pending", "pending", "pending"]);
    setProgress([0, 0, 0]);
    setResult(null);
    setStreamed([]);
    setError(null);
    setCancelling(false);
    setTrace(null);
    try {
      const { run_id } = await createRun(req);
      runId.current = run_id;
      setCurrentRun(run_id);
      stop.current = subscribe(run_id, {
        onEvent,
        onResult: (r) => {
          stop.current?.();
          setResult(r);
          setPhase("finished");
        },
        onError: (message) => {
          setError(message);
          setPhase("finished");
        },
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start the run.");
      setPhase("finished");
    }
  }

  function generate() {
    if (!prompt.trim()) return;
    void start({ prompt: prompt.trim(), plan, mode, n, settings: { aspect_ratio: ratio } });
  }

  function answer(answers: Record<string, string>) {
    if (last.current) void start({ ...last.current, settings: { ...last.current.settings, ...answers } });
  }

  async function cancel() {
    if (!runId.current) return;
    setCancelling(true);
    try {
      await cancelRun(runId.current);
    } catch {
      setCancelling(false);
    }
  }

  const running = phase === "running";

  return (
    <>
    <main className="mx-auto max-w-3xl space-y-6 px-4 py-10">
      <header>
        <h1 className="text-2xl font-semibold">Media Studio</h1>
        <p className="text-sm text-neutral-600">Describe an image and we will make it.</p>
        {MOCK && (
          <p className="mt-2 rounded bg-neutral-200 px-2 py-1 text-xs text-neutral-700">
            Mock mode: no backend. Words in the prompt pick the outcome: question (manual mode), issues, reject,
            fail, complex.
          </p>
        )}
      </header>

      <section className="space-y-4 rounded-lg border border-neutral-200 bg-white p-4">
        <textarea
          className="h-28 w-full resize-none rounded-md border border-neutral-300 p-3"
          placeholder="A cozy coffee shop at dusk, warm lamps, wooden tables"
          value={prompt}
          disabled={running}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <label className="text-sm">
            Plan
            <select className={select} value={plan} disabled={running} onChange={(e) => setPlan(e.target.value as Plan)}>
              {PLANS.map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            Mode
            <select className={select} value={mode} disabled={running} onChange={(e) => setMode(e.target.value as Mode)}>
              {MODES.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            Images
            <select className={select} value={n} disabled={running} onChange={(e) => setN(Number(e.target.value))}>
              {[1, 2, 3, 4].map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            Aspect ratio
            <select
              className={select}
              value={ratio}
              disabled={running}
              onChange={(e) => setRatio(e.target.value as AspectRatio)}
            >
              {RATIOS.map((r) => (
                <option key={r}>{r}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={generate}
            disabled={running || !prompt.trim()}
            className="rounded-md bg-indigo-600 px-5 py-2 font-medium text-white disabled:opacity-40"
          >
            Generate
          </button>
          {running && (
            <button
              onClick={cancel}
              disabled={cancelling}
              className="rounded-md border border-neutral-300 px-4 py-2 font-medium disabled:opacity-50"
            >
              {cancelling ? "Cancelling…" : "Cancel"}
            </button>
          )}
        </div>
      </section>

      {INSPECTOR && currentRun && (
        <button
          onClick={() => setShowInspector((v) => !v)}
          className="rounded-md border border-neutral-300 bg-white px-3 py-1.5 text-sm font-medium"
        >
          {showInspector ? "Hide inspector" : "Show inspector"}
        </button>
      )}

      {running && (
        <section className="rounded-lg border border-neutral-200 bg-white p-4">
          <Stepper states={steps} progress={progress} total={n} />
          {streamed.length > 0 && (
            <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {streamed.map((id, i) => (
                <li key={id} className="overflow-hidden rounded border border-neutral-200">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={imageUrl(id)} alt={`Image ${i + 1}, ready`} className="aspect-square w-full object-cover" />
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {phase === "finished" && (
        <section>
          {result ? (
            <ResultView result={result} onAnswers={answer} onRetry={() => last.current && void start(last.current)} />
          ) : (
            <div className="rounded-lg border border-red-200 bg-red-50 p-4">
              <p className="mb-3 font-medium text-red-900">{error ?? "Something went wrong."}</p>
              <button
                onClick={() => last.current && void start(last.current)}
                className="rounded-md bg-red-600 px-4 py-2 font-medium text-white"
              >
                Try again
              </button>
            </div>
          )}
        </section>
      )}
    </main>
    {INSPECTOR && showInspector && (
      <div className="mx-auto max-w-6xl px-4 pb-16">
        <h2 className="mb-3 text-lg font-semibold">Inspector <span className="text-sm font-normal text-neutral-500">(developer view: not part of the product UI)</span></h2>
        <Inspector trace={trace} running={running} />
      </div>
    )}
    </>
  );
}
