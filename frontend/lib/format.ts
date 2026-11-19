/** Small formatting helpers for the inspector. */

export function ms(value: number | null | undefined): string {
  if (value === null || value === undefined) return "–";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

export function num(value: number, digits = 0): string {
  return value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function usd(value: number): string {
  return `$${value.toFixed(4)}`;
}

/** Milliseconds since the run started, for timestamps in the timeline. */
export function since(start: string | null, ts: string | null): string {
  if (!start || !ts) return "";
  return `+${((Date.parse(ts) - Date.parse(start)) / 1000).toFixed(2)}s`;
}
