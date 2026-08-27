import type { TranscriptionResult } from "../../api/contracts";
import type { HttpPort } from "../../app/services";
import type { AudioSelection } from "../../media/recorder";

export interface TranscriptionInput {
  audio: AudioSelection;
  language: "" | "fr" | "en";
  diarize: boolean;
  channels: "mix" | "left" | "right" | "separate";
  wordTimestamps: boolean;
  numSpeakers: string;
  minSpeakers: string;
  maxSpeakers: string;
}

export function transcribe(http: HttpPort, input: TranscriptionInput): Promise<TranscriptionResult> {
  const form = new FormData();
  form.append("file", input.audio.blob, input.audio.filename);
  form.set("response_format", "json");
  form.set("word_timestamps", String(input.wordTimestamps));
  form.set("diarize", String(input.diarize));
  form.set("channels", input.channels);
  if (input.language) form.set("language", input.language);
  if (input.numSpeakers) form.set("num_speakers", input.numSpeakers);
  if (input.minSpeakers) form.set("min_speakers", input.minSpeakers);
  if (input.maxSpeakers) form.set("max_speakers", input.maxSpeakers);
  return http.postForm<TranscriptionResult>("/transcribe", form);
}
