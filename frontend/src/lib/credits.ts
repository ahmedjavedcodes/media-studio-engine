// The same credit rule as the backend (gateway/pricing.py): the model's price, plus a surcharge for
// each reference image after the first, times the resolution multiplier, rounded up.
import type { AutoPoolEntry, ModelInfo } from "@/lib/api";

type Priced = Pick<
  ModelInfo,
  "credits_per_image" | "ref_surcharge" | "resolution_multipliers" | "max_refs"
>;

export function creditsPerImage(m: Priced, attached: number, resolution = "1K"): number {
  const base = m.credits_per_image + Math.max(0, attached - 1) * m.ref_surcharge;
  return Math.ceil(base * (m.resolution_multipliers[resolution] ?? 1));
}

// A model that cannot take the attached images cannot be picked.
export function takesImages(m: Pick<ModelInfo, "max_refs">, attached: number): boolean {
  return attached <= m.max_refs;
}

// Auto picks a model that fits, so the most it can cost is the dearest model that fits.
export function creditEstimate(
  models: ModelInfo[],
  choice: string,
  n: number,
  attached: number,
  resolution = "1K",
  autoPool: AutoPoolEntry[] = [],
): { credits: number; upTo: boolean } {
  if (choice !== "auto") {
    const m = models.find((x) => x.id === choice);
    return { credits: (m ? creditsPerImage(m, attached, resolution) : 1) * n, upTo: false };
  }
  // The pool the server says Auto can pick from. An older server sends none: use the picker list.
  const pool: Priced[] =
    autoPool.length > 0
      ? autoPool
      : models.filter((m) => m.available && !m.locked);
  const usable = pool.filter((m) => takesImages(m, attached));
  const dearest = Math.max(1, ...usable.map((m) => creditsPerImage(m, attached, resolution)));
  return { credits: dearest * n, upTo: true };
}

// Capability pills shown next to a model.
export const PILLS: { label: string; has: (c: string[]) => boolean }[] = [
  { label: "SVG Vector", has: (c) => c.includes("svg_vector") },
  { label: "Multi-Ref", has: (c) => c.includes("multi_ref_merge") },
  {
    label: "4K Studio",
    has: (c) => ["4k", "1k_to_4k", "studio_photoreal"].some((x) => c.includes(x)),
  },
  { label: "Transparent BG", has: (c) => c.includes("transparent_bg") },
  { label: "Fast", has: (c) => c.includes("fast") },
];
