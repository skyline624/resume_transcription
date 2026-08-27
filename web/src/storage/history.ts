export type HistoryKind = "transcription" | "summary" | "speech";

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface HistoryEntry {
  id: string;
  createdAt: string;
  kind: HistoryKind;
  title: string;
  parameters: Record<string, string | number | boolean | null>;
  resultText: string;
  metadata: Record<string, JsonValue>;
  audio?: Blob;
}

export type HistoryDraft = Omit<HistoryEntry, "id" | "createdAt" | "audio">;

export interface HistoryLimits {
  maxEntries: number;
  maxAudioBytes: number;
}

export interface HistoryRepository {
  add(draft: HistoryDraft): Promise<HistoryEntry>;
  list(): Promise<HistoryEntry[]>;
  get(id: string): Promise<HistoryEntry | undefined>;
  remove(id: string): Promise<void>;
  keepAudio(id: string, audio: Blob): Promise<{ evictedIds: string[] }>;
  confirmAudioEviction(id: string, audio: Blob, evictedIds: string[]): Promise<void>;
  clear(): Promise<void>;
  getLimits(): Promise<HistoryLimits>;
  setLimits(limits: HistoryLimits): Promise<void>;
}
