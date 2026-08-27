export type AudioFormat = "wav" | "mp3" | "flac" | "opus" | "aac" | "pcm";

export type TtsMode =
  | "qwen3-tts-custom-voice"
  | "qwen3-tts-clone"
  | "qwen3-tts-voice-design";

export interface Word {
  word: string;
  start: number;
  end: number;
}

export interface Turn {
  speaker: string | null;
  start: number;
  end: number;
  text: string;
  words: Word[];
}

export interface TranscriptionResult {
  text: string;
  language: string | null;
  duration: number;
  speakers: string[];
  turns: Turn[];
  timing: Record<string, number>;
  channels_used: number;
}

export interface SummaryResult {
  summary: string;
  format: "structure" | "narratif";
  model: string;
  transcript?: string;
}

export interface TtsHealth {
  enabled: boolean;
  worker: boolean;
  state: string;
  downloaded_models: string[];
  loaded_model: string | null;
  precision: string | null;
  device: string | null;
  attention: string | null;
  features: string[];
  last_error: string | null;
  pid: number | null;
  vram_allocated_mib: number | null;
}

export interface Health {
  status: string;
  device: string;
  asr_model: string;
  diarization_model: string;
  diarization_enabled: boolean;
  vad_model: string;
  vad_device: string | null;
  vad_enabled: boolean;
  summary_model: string;
  summary_enabled: boolean;
  gpu: Record<string, unknown> | null;
  tts: TtsHealth;
}

export interface Voice {
  id: string;
  name: string;
  kind: "builtin" | "clone";
  language?: string;
  created_at?: string;
  duration_s?: number;
}

export interface ListResponse<T> {
  object: "list";
  data: T[];
}

export interface SpeechRequest {
  model: TtsMode;
  input: string;
  voice?: string;
  instructions?: string;
  response_format: AudioFormat;
  speed: number;
  language: string;
}

export interface AudioResult {
  blob: Blob;
  contentType: string;
  filename?: string;
}
