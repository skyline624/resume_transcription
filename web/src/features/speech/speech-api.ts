import type { AudioFormat, AudioResult, SpeechRequest } from "../../api/contracts";
import type { HttpPort } from "../../app/services";
import type { AudioSelection } from "../../media/recorder";

export function createSpeech(http: HttpPort, request: SpeechRequest): Promise<AudioResult> {
  return http.postBlob("/v1/audio/speech", request);
}

export interface OneShotCloneInput {
  reference: AudioSelection;
  input: string;
  consent: boolean;
  transcript: string;
  language: string;
  responseFormat: AudioFormat;
  speed: number;
}

export function cloneOnce(http: HttpPort, input: OneShotCloneInput): Promise<AudioResult> {
  const form = new FormData();
  form.append("file", input.reference.blob, input.reference.filename);
  form.set("input", input.input);
  form.set("consent", String(input.consent));
  if (input.transcript.trim()) form.set("transcript", input.transcript.trim());
  form.set("language", input.language);
  form.set("response_format", input.responseFormat);
  form.set("speed", String(input.speed));
  return http.postBlob("/v1/audio/speech/clone", form);
}
