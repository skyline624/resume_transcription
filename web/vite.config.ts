import preact from "@preact/preset-vite";
import { defineConfig } from "vitest/config";

const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [preact()],
  server: {
    proxy: {
      "/health": api,
      "/transcribe": api,
      "/summarize": api,
      "/v1": api,
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
