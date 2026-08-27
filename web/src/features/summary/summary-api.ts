import type { SummaryResult } from "../../api/contracts";
import type { HttpPort } from "../../app/services";
import type { AudioSelection } from "../../media/recorder";

export type SummaryFormat = "structure" | "narratif";

export type SummaryInput =
  | { source: "text" | "history"; transcript: string; format: SummaryFormat }
  | {
      source: "audio";
      audio: AudioSelection;
      format: SummaryFormat;
      language: "" | "fr" | "en";
      channels: "mix" | "left" | "right" | "separate";
      diarize: boolean;
    };

export function summarize(http: HttpPort, input: SummaryInput): Promise<SummaryResult> {
  const form = new FormData();
  form.set("format", input.format);
  form.set("response_format", "json");
  if (input.source === "audio") {
    form.append("file", input.audio.blob, input.audio.filename);
    form.set("diarize", String(input.diarize));
    form.set("channels", input.channels);
    if (input.language) form.set("language", input.language);
  } else {
    form.set("transcript", input.transcript);
  }
  return http.postForm<SummaryResult>("/summarize", form);
}
