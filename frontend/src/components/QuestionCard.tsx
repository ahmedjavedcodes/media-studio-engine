"use client";

import { useState } from "react";
import type { Question } from "@/lib/api";

type Props = {
  questions: Question[];
  busy: boolean;
  onSubmit: (answers: Record<string, string>) => void;
};

// The enhancer needs one more detail. Each question has 3 choices (one recommended) and
// "Something else", which opens a text box.
export default function QuestionCard({ questions, busy, onSubmit }: Props) {
  const [picked, setPicked] = useState<Record<string, string>>(() =>
    Object.fromEntries(questions.map((q) => [q.id, q.options.find((o) => o.recommended)?.id ?? ""])),
  );
  const [typed, setTyped] = useState<Record<string, string>>({});

  const ready = questions.every((q) => picked[q.id] && (picked[q.id] !== "other" || typed[q.id]?.trim()));

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const answers: Record<string, string> = {};
    for (const q of questions) {
      answers[q.id] = picked[q.id];
      if (picked[q.id] === "other") answers[`${q.id}_text`] = typed[q.id].trim();
    }
    onSubmit(answers);
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-4 rounded-lg border border-sky-400 bg-sky-50 p-4 text-sky-950">
      <p className="font-medium">Need a quick detail</p>
      {questions.map((q) => (
        <fieldset key={q.id} className="flex flex-col gap-2">
          <legend className="mb-1 text-sm font-medium">{q.text}</legend>
          {q.options.map((o) => (
            <label key={o.id} className="flex items-center gap-2 text-sm">
              <input
                type="radio"
                name={q.id}
                checked={picked[q.id] === o.id}
                onChange={() => setPicked((p) => ({ ...p, [q.id]: o.id }))}
              />
              {o.label}
              {o.recommended && <span className="rounded bg-sky-200 px-1.5 text-xs">Recommended</span>}
            </label>
          ))}
          {picked[q.id] === "other" && (
            <input
              value={typed[q.id] ?? ""}
              onChange={(e) => setTyped((t) => ({ ...t, [q.id]: e.target.value }))}
              maxLength={200}
              placeholder="Type your answer"
              aria-label="Your answer"
              className="rounded border border-sky-300 bg-white px-2 py-1 text-sm"
            />
          )}
        </fieldset>
      ))}
      <button
        type="submit"
        disabled={busy || !ready}
        className="self-start rounded-lg bg-sky-900 px-4 py-2 text-white disabled:opacity-50"
      >
        {busy ? "Checking…" : "Continue"}
      </button>
    </form>
  );
}
