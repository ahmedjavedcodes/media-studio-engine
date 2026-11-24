import { defineConfig } from "@playwright/test";

// Smoke test of the real frontend against the real FastAPI app with FAKE providers
// (tests/e2e/fake_backend.py). Nothing calls Cloudflare or any paid service.
// (Next rewrites tsconfig.json on each build. Run `git checkout frontend/tsconfig.json` after.)
// Backend: port 8001. Frontend: a production build on port 3100, pointed at that backend.
const root = "..";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: "http://localhost:3100", screenshot: "off", trace: "off" },
  webServer: [
    {
      command: ".venv/bin/python -m uvicorn tests.e2e.fake_backend:app --port 8001", // runs in cwd
      cwd: root,
      url: "http://localhost:8001/health",
      env: {
        PYTHONPATH: "src",
        DRAFT_TIMEOUT_S: "3",
        CORS_ORIGINS: "http://localhost:3100",
        // the smoke test must never post traces to LangSmith
        LANGSMITH_TRACING: "false",
        LANGSMITH_API_KEY: "",
      },
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      // A production build in its own folder and on its own port: a `next dev` of yours on port
      // 3000 is never touched. The API address is baked in at build time.
      command: "npx next build && npx next start --port 3100",
      url: "http://localhost:3100",
      env: { NEXT_PUBLIC_API_URL: "http://localhost:8001", NEXT_DIST_DIR: ".next-e2e" },
      reuseExistingServer: false,
      timeout: 300_000,
    },
  ],
});
