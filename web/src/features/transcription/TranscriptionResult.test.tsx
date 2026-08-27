import { fireEvent, render, screen, within } from "@testing-library/preact";
import { describe, expect, it, vi } from "vitest";

import type { TranscriptionResult as Result } from "../../api/contracts";
import { TranscriptionResult } from "./TranscriptionResult";

const DIARIZED_RESULT: Result = {
  text: "Bonjour. Bienvenue.",
  language: "fr",
  duration: 4,
  speakers: ["SPEAKER_00", "SPEAKER_01"],
  turns: [
    {
      speaker: "SPEAKER_00",
      start: 0,
      end: 1.5,
      text: "Bonjour.",
      words: [],
    },
    {
      speaker: "SPEAKER_01",
      start: 2,
      end: 4,
      text: "Bienvenue.",
      words: [],
    },
  ],
  timing: { asr: 1 },
  channels_used: 1,
};

describe("TranscriptionResult", () => {
  it("shows diarized turns as the primary transcription", () => {
    render(<TranscriptionResult historyId={null} result={DIARIZED_RESULT} />);

    const dialogue = screen.getByRole("list", { name: "Transcription par locuteur" });
    expect(within(dialogue).getByText("SPEAKER_00")).toBeTruthy();
    expect(within(dialogue).getByText("Bonjour.")).toBeTruthy();
    expect(within(dialogue).getByText("SPEAKER_01")).toBeTruthy();
    expect(within(dialogue).getByText("Bienvenue.")).toBeTruthy();

    const continuousText = screen.getByText("Texte continu").closest("details");
    expect(continuousText?.hasAttribute("open")).toBe(false);
  });

  it("copies the dialogue with speaker labels", () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    render(<TranscriptionResult historyId={null} result={DIARIZED_RESULT} />);

    fireEvent.click(screen.getByRole("button", { name: "Copier" }));

    expect(writeText).toHaveBeenCalledWith(
      "SPEAKER_00 : Bonjour.\nSPEAKER_01 : Bienvenue.\n",
    );
  });
});
