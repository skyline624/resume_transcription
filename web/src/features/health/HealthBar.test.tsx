import { fireEvent, render, screen } from "@testing-library/preact";
import { describe, expect, it } from "vitest";

import { fakeServices } from "../../test/fakes";
import { HealthBar } from "./HealthBar";

describe("HealthBar", () => {
  it("explains when the Qwen worker is unavailable", async () => {
    const services = fakeServices({
      health: {
        status: "degraded",
        device: "cuda",
        gpu: { name: "RTX 3090" },
        tts: {
          enabled: true,
          worker: false,
          state: "error",
          last_error: "worker_unreachable",
        },
      },
    });
    render(<HealthBar />, { wrapper: services.wrapper });

    expect(await screen.findByText("Synthèse indisponible")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Détails du serveur" }));
    expect(screen.getByText("worker_unreachable")).toBeTruthy();
    expect(screen.getByText("RTX 3090")).toBeTruthy();
  });

  it("keeps the status understandable without relying on color", async () => {
    const services = fakeServices({ health: { status: "ok" } });
    render(<HealthBar />, { wrapper: services.wrapper });

    expect(await screen.findByText("Opérationnel")).toBeTruthy();
  });
});
