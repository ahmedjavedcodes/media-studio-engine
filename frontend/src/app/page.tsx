"use client";

import { useEffect, useRef, useState } from "react";
import { API_URL, startRun, type Blocked } from "@/lib/api";

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

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [progress, setProgress] = useState<string[]>([]);
  const [result, setResult] = useState<string | null>(null);
  // Cooldown: the time (ms) when the user may try again, and a ticking clock for the countdown.
  const [retryAt, setRetryAt] = useState(0);
  const [now, setNow] = useState(0);
  const source = useRef<EventSource | null>(null);

  useEffect(() => {
    if (retryAt <= Date.now()) return;
    const timer = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(timer);
  }, [retryAt]);

  useEffect(() => () => source.current?.close(), []);

  const secondsLeft = Math.max(0, Math.ceil((retryAt - now) / 1000));
  const coolingDown = secondsLeft > 0;

  function startCooldown(seconds: number) {
    setNow(Date.now());
    setRetryAt(Date.now() + seconds * 1000);
  }

  function watchRun(runId: string) {
    source.current?.close();
    const es = new EventSource(`${API_URL}/v1/runs/${runId}/events`);
    source.current = es;
    es.addEventListener("progress", (e) => {
      const data = JSON.parse((e as MessageEvent).data);
      setProgress((p) => [...p, data.message]);
    });
    es.addEventListener("result", (e) => {
      const data = JSON.parse((e as MessageEvent).data);
      setResult(data.message);
      es.close();
    });
    es.onerror = () => es.close();
  }

  async function onFiles(files: FileList | null) {
    setImages(files ? await Promise.all(Array.from(files).map(toBase64)) : []);
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy || coolingDown) return;
    setBusy(true);
    setNotice(null);
    setProgress([]);
    setResult(null);
    const out = await startRun(prompt, images);
    setBusy(false);
    if (out.kind === "started") {
      watchRun(out.runId);
    } else if (out.kind === "blocked") {
      setNotice({ kind: "blocked", body: out.body });
      if (out.body.retry_after) startCooldown(out.body.retry_after);
    } else if (out.kind === "cooldown") {
      setNotice({ kind: "cooldown", message: out.message });
      startCooldown(out.retryAfter);
    } else {
      setNotice({ kind: "error", message: out.message });
    }
  }

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
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          multiple
          onChange={(e) => onFiles(e.target.files)}
          className="text-sm"
        />
        {images.length > 0 && (
          <p className="text-sm text-zinc-500">
            {images.length} image{images.length > 1 ? "s" : ""} attached
          </p>
        )}
        <button
          type="submit"
          disabled={busy || coolingDown}
          className="rounded-lg bg-zinc-900 px-4 py-2 text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
        >
          {busy
            ? "Checking your request…"
            : coolingDown
              ? `Try again in ${formatWait(secondsLeft)}`
              : "Create"}
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

      {progress.length > 0 && (
        <ul className="text-sm text-zinc-600 dark:text-zinc-400">
          {progress.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      )}
      {result && <p className="rounded-lg border p-4">{result}</p>}
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
