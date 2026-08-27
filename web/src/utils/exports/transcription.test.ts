import { describe, expect, it } from "vitest";

import type { TranscriptionResult } from "../../api/contracts";
import { toDialogue, toSrt, toText, toVtt } from "./transcription";

const result: TranscriptionResult = {
  text: "Bonjour",
  language: "fr",
  duration: 3.5,
  speakers: ["SPEAKER_00"],
  turns: [
    {
      speaker: "SPEAKER_00",
      start: 1.25,
      end: 3.5,
      text: "Bonjour",
      words: [],
    },
  ],
  timing: {},
  channels_used: 1,
};

describe("transcription exports", () => {
  it("creates valid SRT timestamps", () => {
    expect(toSrt(result)).toBe(
      "1\n00:00:01,250 --> 00:00:03,500\nSPEAKER_00 : Bonjour\n",
    );
  });

  it("offers plain text, dialogue and WebVTT from the same response", () => {
    expect(toText(result)).toBe("Bonjour\n");
    expect(toDialogue(result)).toBe("SPEAKER_00 : Bonjour\n");
    expect(toVtt(result)).toContain("00:00:01.250 --> 00:00:03.500");
  });
});
