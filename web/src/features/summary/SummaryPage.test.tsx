import { fireEvent, render, screen } from "@testing-library/preact";
import { afterEach, describe, expect, it } from "vitest";

import { fakeServices } from "../../test/fakes";
import { SummaryPage } from "./SummaryPage";

afterEach(() => {
  window.location.hash = "";
});

describe("SummaryPage", () => {
  it("loads a local transcription without ever sending its source audio", async () => {
    window.location.hash = "#/summarize?history=entry-1";
    const services = fakeServices({
      historyEntry: {
        id: "entry-1",
        kind: "transcription",
        resultText: "Décision validée.",
      },
    });
    render(<SummaryPage />, { wrapper: services.wrapper });

    expect(await screen.findByDisplayValue("Décision validée.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Rédiger le résumé" }));

    expect(await screen.findByText("Compte-rendu rédigé.")).toBeTruthy();
    expect(services.http.lastForm?.get("transcript")).toBe("Décision validée.");
    expect(services.http.lastForm?.has("file")).toBe(false);
  });

  it("prevents a request when the summary service is disabled", async () => {
    const services = fakeServices({ health: { summary_enabled: false } });
    render(<SummaryPage />, { wrapper: services.wrapper });

    expect(await screen.findByText(/ENABLE_SUMMARY=true/)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Rédiger le résumé" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
