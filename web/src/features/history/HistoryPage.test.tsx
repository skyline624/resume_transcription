import { fireEvent, render, screen } from "@testing-library/preact";
import { afterEach, describe, expect, it } from "vitest";

import { fakeServices } from "../../test/fakes";
import type { HistoryEntry } from "../../storage/history";
import { HistoryPage } from "./HistoryPage";

afterEach(() => {
  window.location.hash = "";
});

describe("HistoryPage", () => {
  it("reuses a transcription for a new summary", async () => {
    const transcriptionEntry: HistoryEntry = {
      id: "entry-1",
      createdAt: "2026-08-27T10:00:00Z",
      kind: "transcription",
      title: "réunion.wav",
      parameters: {},
      resultText: "Décision validée.",
      metadata: {},
    };
    render(<HistoryPage />, {
      wrapper: fakeServices({ historyEntries: [transcriptionEntry] }).wrapper,
    });

    fireEvent.click(await screen.findByRole("button", { name: "Ouvrir réunion.wav" }));
    fireEvent.click(screen.getByRole("link", { name: "Résumer ce texte" }));
    expect(window.location.hash).toBe("#/summarize?history=entry-1");
  });
});
