// Types and the one call that starts a run. Mirrors the 422/429/403/400/503 bodies of POST /v1/runs.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Blocked = {
  code: "content_blocked";
  category: string;
  blocked_by: "prompt" | "image" | "both";
  message: string;
  tip: string | null;
  warning: boolean;
  // Instant reject (strike)
  strike_count?: number;
  strike_limit?: number;
  strike_text?: string;
  banned?: boolean;
  // Standard reject (cooldown)
  retry_after?: number;
  retry_text?: string;
};

export type StartResult =
  | { kind: "started"; runId: string }
  | { kind: "blocked"; body: Blocked }
  | { kind: "cooldown"; message: string; retryAfter: number }
  | { kind: "error"; code: string; message: string };

export async function startRun(prompt: string, images: string[]): Promise<StartResult> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/v1/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, images }),
    });
  } catch {
    return { kind: "error", code: "network", message: "Could not reach the server. Please try again." };
  }
  const body = await res.json().catch(() => ({}));
  if (res.ok) return { kind: "started", runId: body.run_id };
  if (res.status === 422 && body.code === "content_blocked") return { kind: "blocked", body };
  if (res.status === 429) {
    return { kind: "cooldown", message: body.message ?? "Please wait.", retryAfter: body.retry_after ?? 60 };
  }
  return {
    kind: "error",
    code: body.code ?? "error",
    message: body.message ?? "Something went wrong. Please try again.",
  };
}
