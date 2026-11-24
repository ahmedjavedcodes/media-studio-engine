"use client";

import { useMemo, useState } from "react";
import type { ModelInfo } from "@/lib/api";

const TIER_LABEL: Record<string, string> = { basic: "Basic", pro: "Pro", ultra: "Ultra", max: "Max" };

type Props = {
  models: ModelInfo[];
  value: string; // "auto" or a model id
  onChange: (id: string) => void;
};

// Two views: a list sorted by tier, and a view grouped by family. Auto is always first.
export default function ModelPicker({ models, value, onChange }: Props) {
  const [view, setView] = useState<"list" | "family">("list");
  const [query, setQuery] = useState("");

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return models.filter((m) => !q || m.name.toLowerCase().includes(q) || m.family.includes(q));
  }, [models, query]);

  const groups = useMemo(() => {
    const map = new Map<string, ModelInfo[]>();
    for (const m of shown) map.set(m.family, [...(map.get(m.family) ?? []), m]);
    return [...map.entries()];
  }, [shown]);

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-zinc-300 p-3 dark:border-zinc-700">
      <div className="flex items-center gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search models"
          aria-label="Search models"
          className="min-w-0 flex-1 rounded border border-zinc-300 px-2 py-1 text-sm dark:border-zinc-700 dark:bg-zinc-900"
        />
        <div role="tablist" className="flex overflow-hidden rounded border border-zinc-300 text-xs dark:border-zinc-700">
          {(["list", "family"] as const).map((v) => (
            <button
              key={v}
              type="button"
              role="tab"
              aria-selected={view === v}
              onClick={() => setView(v)}
              className={`px-2 py-1 ${view === v ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900" : ""}`}
            >
              {v === "list" ? "List" : "Family"}
            </button>
          ))}
        </div>
      </div>

      <ul className="flex max-h-56 flex-col gap-1 overflow-y-auto text-sm">
        <Row
          name="Auto"
          badge="Recommended"
          selected={value === "auto"}
          onPick={() => onChange("auto")}
        />
        {view === "list"
          ? shown.map((m) => <ModelRow key={m.id} m={m} value={value} onChange={onChange} />)
          : groups.map(([family, items]) => (
              <li key={family}>
                <p className="mt-1 text-xs font-semibold uppercase text-zinc-500">{family.replace("_", " ")}</p>
                <ul className="flex flex-col gap-1">
                  {items.map((m) => (
                    <ModelRow key={m.id} m={m} value={value} onChange={onChange} />
                  ))}
                </ul>
              </li>
            ))}
        {shown.length === 0 && <li className="px-2 py-1 text-zinc-500">No model matches.</li>}
      </ul>
    </div>
  );
}

function ModelRow({ m, value, onChange }: { m: ModelInfo; value: string; onChange: (id: string) => void }) {
  return (
    <Row
      name={m.name}
      badge={TIER_LABEL[m.tier]}
      tag={m.tag}
      credits={m.credits_per_image}
      locked={m.locked}
      selected={value === m.id}
      onPick={() => onChange(m.id)}
    />
  );
}

function Row(props: {
  name: string;
  badge: string;
  tag?: string;
  credits?: number;
  locked?: boolean;
  selected: boolean;
  onPick: () => void;
}) {
  const { name, badge, tag, credits, locked, selected, onPick } = props;
  return (
    <li>
      <button
        type="button"
        onClick={onPick}
        disabled={locked}
        aria-pressed={selected}
        title={locked ? "Upgrade your plan to use this model" : undefined}
        className={`flex w-full items-center justify-between gap-2 rounded px-2 py-1 text-left disabled:opacity-50 ${
          selected ? "bg-zinc-200 dark:bg-zinc-800" : "hover:bg-zinc-100 dark:hover:bg-zinc-900"
        }`}
      >
        <span className="flex min-w-0 flex-col">
          <span className="truncate">
            {locked && "🔒 "}
            {name}
          </span>
          {tag && <span className="truncate text-xs text-zinc-500">{tag}</span>}
        </span>
        <span className="flex shrink-0 items-center gap-2 text-xs">
          {credits !== undefined && <span className="text-zinc-500">{credits} cr</span>}
          <span className="rounded bg-zinc-200 px-1.5 py-0.5 dark:bg-zinc-700">{locked ? "Upgrade" : badge}</span>
        </span>
      </button>
    </li>
  );
}
