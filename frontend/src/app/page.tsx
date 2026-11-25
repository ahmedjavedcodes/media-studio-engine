"use client";

import { useEffect, useRef, useState } from "react";
import ImageGrid from "@/components/ImageGrid";
import TracePanel from "@/components/TracePanel";
import { creditEstimate, takesImages } from "@/lib/credits";
import ModelPicker from "@/components/ModelPicker";
import QuestionCard from "@/components/QuestionCard";
import {
  eventsUrl,
  fetchModels,
  fetchPlans,
  resumeRun,
  startRun,
  type Blocked,
  type ImageOut,
  type AutoPoolEntry,
  type ModelInfo,
  type Plan,
  type PlansResponse,
  type Question,
  type RunEvent,
  type Settings,
  type StartResult,
} from "@/lib/api";

const REPORT_URL = process.env.NEXT_PUBLIC_REPORT_URL ?? "#report-wrong-block";

type Notice =
  | { kind: "blocked"; body: Blocked }
  | { kind: "cooldown"; message: string }
  | { kind: "error"; message: string };

function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] ?? "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function formatWait(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

const STREAM_EVENTS = ["status", "question", "ready", "error", "expired"] as const;
const MAX_RECONNECTS = 3;

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [images, setImages] = useState<string[]>([]);
  // Build phase: the plan is picked by hand. It will come from the user's account later.
  const [plan, setPlan] = useState<Plan>("basic");
  const [plans, setPlans] = useState<PlansResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [autoPool, setAutoPool] = useState<AutoPoolEntry[]>([]);
  const [settings, setSettings] = useState<Settings>({
    model: "auto",
    aspect_ratio: "1:1",
    resolution: "1K",
    n: 1,
  });
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [status, setStatus] = useState<string>("");
  const [runId, setRunId] = useState<string | null>(null);
  const token = useRef(""); // signed by the server for this run. It goes in the stream URL
  const [traceToken, setTraceToken] = useState(""); // the same token, for the dev trace panel
  const [questions, setQuestions] = useState<Question[] | null>(null);
  const [shown, setShown] = useState<ImageOut[]>([]);
  const [recommendation, setRecommendation] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  // Cooldown: the time (ms) when the user may try again, and a ticking clock for the countdown.
  const [retryAt, setRetryAt] = useState(0);
  const [now, setNow] = useState(0);
  const source = useRef<EventSource | null>(null);
  const reconnect = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnects = useRef(0); // reset whenever an event arrives, so a long run can drop often

  useEffect(() => {
    fetchPlans().then(setPlans).catch(() => setNotice({ kind: "error", message: "Could not reach the server." }));
  }, []);

  // The plan decides which models are locked.
  useEffect(() => {
    fetchModels(plan)
      .then(({ models: list, autoPool: pool }) => {
        setModels(list);
        setAutoPool(pool);
        // A model the new plan does not allow goes back to Auto.
        setSettings((s) => (list.some((m) => m.id === s.model && m.locked) ? { ...s, model: "auto" } : s));
      })
      .catch(() => {
        setModels([]);
        setAutoPool([]);
      });
  }, [plan]);

  // ...and the variation limit.
  function changePlan(next: Plan) {
    setPlan(next);
    const limit = plans?.plans.find((p) => p.id === next)?.max_n ?? 1;
    setSettings((s) => ({ ...s, n: Math.min(s.n, limit) }));
  }

  useEffect(() => {
    if (retryAt <= Date.now()) return;
    const timer = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(timer);
  }, [retryAt]);

  useEffect(
    () => () => {
      source.current?.close();
      if (reconnect.current) clearTimeout(reconnect.current);
    },
    [],
  );

  const secondsLeft = Math.max(0, Math.ceil((retryAt - now) / 1000));
  const coolingDown = secondsLeft > 0;
  const maxN = plans?.plans.find((p) => p.id === plan)?.max_n ?? 1;
  const chosen = models.find((m) => m.id === settings.model);
  const estimate = creditEstimate(
    models,
    settings.model,
    settings.n,
    images.length,
    settings.resolution,
    autoPool,
  );
  const creditText = `${estimate.upTo ? "up to " : ""}${estimate.credits} credit${estimate.credits === 1 ? "" : "s"}`;

  function startCooldown(seconds: number) {
    setNow(Date.now());
    setRetryAt(Date.now() + seconds * 1000);
  }

  function pickModel(id: string) {
    const m = models.find((x) => x.id === id);
    setSettings((s) => ({
      ...s,
      model: id,
      aspect_ratio: m && !m.aspect_ratios.includes(s.aspect_ratio) ? m.aspect_ratios[0] : s.aspect_ratio,
    }));
  }

  function onEvent(ev: RunEvent) {
    if (ev.type === "status") {
      setStatus(ev.progress ? `${ev.label} (${ev.progress})` : (ev.label ?? ""));
    } else if (ev.type === "question") {
      setQuestions(ev.questions ?? []);
      setStatus("Need a quick detail");
    } else if (ev.type === "ready") {
      setShown(ev.images ?? []);
      setRecommendation(ev.recommendation ?? null);
      setStatus("Ready");
      setDone(true);
    } else if (ev.type === "expired") {
      setQuestions(null);
      setStatus("");
      setNotice({ kind: "error", message: ev.message ?? "This request timed out. Tap Create to start again." });
    } else if (ev.type === "error") {
      setStatus("");
      setShown([]);
      setNotice({ kind: "error", message: ev.message ?? "Something went wrong." });
    }
  }

  // Reads the run's events. The server keeps them, so after a network drop we can connect again
  // and get the current phase from the start.
  function watchRun(id: string, signed: string) {
    source.current?.close();
    const es = new EventSource(eventsUrl(id, signed));
    source.current = es;
    let finished = false;
    for (const name of STREAM_EVENTS) {
      es.addEventListener(name, (e) => {
        reconnects.current = 0;
        onEvent(JSON.parse((e as MessageEvent).data) as RunEvent);
        // The server closes the stream after these. Close here too, so it does not reconnect.
        if (name === "question" || name === "ready" || name === "error" || name === "expired") {
          finished = true;
          es.close();
        }
      });
    }
    es.onerror = () => {
      es.close();
      if (finished) return;
      if (reconnects.current < MAX_RECONNECTS) {
        const delay = 500 * 2 ** reconnects.current;
        reconnects.current += 1;
        reconnect.current = setTimeout(() => watchRun(id, signed), delay);
      } else {
        setStatus("");
        setNotice({ kind: "error", message: "Lost the connection. Please try again." });
      }
    };
  }

  async function onFiles(files: FileList | null) {
    const next = files ? await Promise.all(Array.from(files).map(toBase64)) : [];
    setImages(next);
    // A picked model that cannot take this many images goes back to Auto.
    const picked = models.find((m) => m.id === settings.model);
    if (picked && !takesImages(picked, next.length)) setSettings((s) => ({ ...s, model: "auto" }));
  }

  function handleFailure(out: Exclude<StartResult, { kind: "started" }>) {
    if (out.kind === "blocked") {
      setNotice({ kind: "blocked", body: out.body });
      if (out.body.retry_after) startCooldown(out.body.retry_after);
    } else if (out.kind === "cooldown") {
      setNotice({ kind: "cooldown", message: out.message });
      startCooldown(out.retryAfter);
    } else {
      setNotice({ kind: "error", message: out.message });
    }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy || coolingDown) return;
    source.current?.close();
    if (reconnect.current) clearTimeout(reconnect.current);
    reconnects.current = 0;
    setBusy(true);
    setNotice(null);
    setStatus("Understanding your request");
    setShown([]);
    setRecommendation(null);
    setQuestions(null);
    setDone(false);
    const out = await startRun(prompt, images, plan, settings);
    setBusy(false);
    if (out.kind === "started") {
      setRunId(out.runId);
      token.current = out.token;
      setTraceToken(out.token);
      watchRun(out.runId, out.token);
    } else {
      setStatus("");
      handleFailure(out);
    }
  }

  async function answer(answers: Record<string, string>) {
    if (!runId) return;
    setBusy(true);
    setNotice(null);
    const out = await resumeRun(runId, answers);
    setBusy(false);
    if (out.kind === "started") {
      setQuestions(null);
      setStatus("Thinking");
      token.current = out.token;
      setTraceToken(out.token);
      watchRun(runId, out.token);
    } else {
      handleFailure(out);
    }
  }

  const select = "rounded border border-zinc-300 bg-transparent px-2 py-1 text-sm dark:border-zinc-700";

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-6 px-6 py-12">
      <h1 className="text-2xl font-semibold">Image Create</h1>

      <form onSubmit={submit} className="flex flex-col gap-4">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          placeholder="Describe the image you want"
          className="w-full rounded-lg border border-zinc-300 p-3 dark:border-zinc-700 dark:bg-zinc-900"
        />

        <div className="flex flex-wrap items-center gap-4 text-sm">
          <label className="flex items-center gap-2">
            Plan
            <select value={plan} onChange={(e) => changePlan(e.target.value as Plan)} className={select}>
              {["basic", "pro", "ultra", "max"].map((p) => (
                <option key={p} value={p}>
                  {p[0].toUpperCase() + p.slice(1)}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            Aspect ratio
            <select
              value={settings.aspect_ratio}
              onChange={(e) => setSettings((s) => ({ ...s, aspect_ratio: e.target.value }))}
              className={select}
            >
              {(plans?.aspect_ratios ?? ["1:1"]).map((r) => (
                <option key={r} value={r} disabled={!!chosen && !chosen.aspect_ratios.includes(r)}>
                  {r}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            Resolution
            <select
              value={settings.resolution}
              onChange={(e) => setSettings((s) => ({ ...s, resolution: e.target.value as "1K" | "2K" }))}
              className={select}
            >
              <option value="1K">1K</option>
              <option value="2K" disabled>
                2K (coming soon)
              </option>
            </select>
          </label>
          <label className="flex items-center gap-2" title="Fast, low-credit mode: the cheapest model">
            <input
              type="checkbox"
              checked={!!settings.draft}
              onChange={(e) => setSettings((s) => ({ ...s, draft: e.target.checked }))}
            />
            Fast draft
          </label>
          <label className="flex items-center gap-2">
            Variations
            <select
              value={settings.n}
              onChange={(e) => setSettings((s) => ({ ...s, n: Number(e.target.value) }))}
              className={select}
            >
              {Array.from({ length: maxN }, (_, i) => i + 1).map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>

        <ModelPicker models={models} value={settings.model} onChange={pickModel} attached={images.length} />

        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          multiple
          onChange={(e) => onFiles(e.target.files)}
          className="text-sm"
        />
        {images.length > 0 && (
          <p className="text-sm text-zinc-500">
            {images.length} image{images.length > 1 ? "s" : ""} attached (photo editing is not available yet)
          </p>
        )}
        <button
          type="submit"
          disabled={busy || coolingDown || !!questions}
          className="rounded-lg bg-zinc-900 px-4 py-2 text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
        >
          {busy
            ? "Checking your request…"
            : coolingDown
              ? `Try again in ${formatWait(secondsLeft)}`
              : `Create · ${creditText}`}
        </button>
      </form>

      {notice?.kind === "blocked" && <BlockedCard body={notice.body} secondsLeft={secondsLeft} />}
      {notice?.kind === "cooldown" && (
        <div role="alert" className="rounded-lg border border-amber-400 bg-amber-50 p-4 text-amber-900">
          <p className="font-medium">{notice.message}</p>
          <p className="mt-1 text-sm">You can try again in {formatWait(secondsLeft)}.</p>
        </div>
      )}
      {notice?.kind === "error" && (
        <div role="alert" className="rounded-lg border border-red-400 bg-red-50 p-4 text-red-900">
          {notice.message}
        </div>
      )}

      {status && !done && !questions && (
        <p role="status" className="text-sm text-zinc-600 dark:text-zinc-400">
          {status}…
        </p>
      )}
      {questions && <QuestionCard questions={questions} busy={busy} onSubmit={answer} />}
      <ImageGrid images={shown} />
      {plans?.trace_ui && runId && traceToken && (
        <TracePanel key={runId} runId={runId} token={traceToken} />
      )}
      {done && recommendation && (
        <p className="rounded-lg border border-emerald-400 bg-emerald-50 p-4 text-emerald-900">{recommendation}</p>
      )}
    </main>
  );
}

function BlockedCard({ body, secondsLeft }: { body: Blocked; secondsLeft: number }) {
  const isStrike = body.strike_count !== undefined;
  return (
    <div role="alert" className="flex flex-col gap-3 rounded-lg border border-red-400 bg-red-50 p-4 text-red-900">
      {isStrike && body.warning && (
        <div className="rounded border border-red-600 bg-red-100 p-3 font-medium">
          {body.strike_text ?? `Strike ${body.strike_count} of ${body.strike_limit}`}
        </div>
      )}
      <p className="font-medium">{body.message}</p>
      {body.tip && <p className="text-sm">{body.tip}</p>}
      {!isStrike && body.retry_after !== undefined && (
        <p className="text-sm">
          {secondsLeft > 0
            ? `You can try again in ${formatWait(secondsLeft)}.`
            : "You can try again now."}
        </p>
      )}
      <a href={REPORT_URL} className="text-sm underline">
        Report a wrong block
      </a>
    </div>
  );
}
