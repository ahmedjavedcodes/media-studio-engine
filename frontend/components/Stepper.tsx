import { PUBLIC_LABELS } from "@/lib/api";

export type StepState = "pending" | "active" | "done";

interface Props {
  states: StepState[]; // one per public label
  progress: number[]; // finished branches per step (only shown when more than one image)
  total: number; // images requested
}

/** The progress stepper. It can only ever show the three public labels. */
export default function Stepper({ states, progress, total }: Props) {
  return (
    <ol className="space-y-3" aria-label="Progress">
      {PUBLIC_LABELS.map((label, i) => {
        const state = states[i] ?? "pending";
        const done = progress[i] ?? 0;
        return (
          <li key={label} className="flex items-center gap-3">
            <span
              className={
                "flex h-7 w-7 items-center justify-center rounded-full text-sm font-medium " +
                (state === "done"
                  ? "bg-emerald-500 text-white"
                  : state === "active"
                    ? "animate-pulse bg-indigo-500 text-white"
                    : "bg-neutral-200 text-neutral-500")
              }
            >
              {state === "done" ? "✓" : i + 1}
            </span>
            <span className={state === "pending" ? "text-neutral-400" : "text-neutral-900"}>
              {label}
              {total > 1 && i > 0 && state !== "pending" ? (
                <span className="ml-2 text-sm text-neutral-500">
                  {Math.min(done, total)} of {total}
                </span>
              ) : null}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
