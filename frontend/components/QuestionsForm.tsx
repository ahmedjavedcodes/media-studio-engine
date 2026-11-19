"use client";

import { useState } from "react";

interface Props {
  questions: string[];
  onSubmit: (answers: Record<string, string>) => void;
}

/** needs_input: one field per question. Submitting re-runs with the answers added to settings. */
export default function QuestionsForm({ questions, onSubmit }: Props) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const complete = questions.every((q) => (answers[q] ?? "").trim() !== "");

  return (
    <form
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault();
        if (complete) onSubmit(answers);
      }}
    >
      {questions.map((question) => (
        <label key={question} className="block">
          <span className="mb-1 block text-sm font-medium text-neutral-800">{question}</span>
          <input
            className="w-full rounded-md border border-neutral-300 px-3 py-2"
            value={answers[question] ?? ""}
            onChange={(e) => setAnswers({ ...answers, [question]: e.target.value })}
          />
        </label>
      ))}
      <button
        type="submit"
        disabled={!complete}
        className="rounded-md bg-indigo-600 px-4 py-2 font-medium text-white disabled:opacity-40"
      >
        Continue
      </button>
    </form>
  );
}
