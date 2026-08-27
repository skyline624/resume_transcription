import type { ListResponse, Voice } from "../../api/contracts";
import type { HttpPort } from "../../app/services";
import type { AudioSelection } from "../../media/recorder";

export function listVoices(http: HttpPort): Promise<ListResponse<Voice>> {
  return http.getJson<ListResponse<Voice>>("/v1/voices");
}

export interface CreateVoiceInput {
  reference: AudioSelection;
  name: string;
  language: string;
  transcript: string;
  consent: boolean;
}

export function createVoice(http: HttpPort, input: CreateVoiceInput): Promise<Voice> {
  const form = new FormData();
  form.append("file", input.reference.blob, input.reference.filename);
  form.set("name", input.name.trim());
  form.set("language", input.language.trim());
  if (input.transcript.trim()) form.set("transcript", input.transcript.trim());
  form.set("consent", String(input.consent));
  return http.postForm<Voice>("/v1/voices", form);
}

export function deleteVoice(http: HttpPort, id: string): Promise<void> {
  return http.delete(`/v1/voices/${encodeURIComponent(id)}`);
}
