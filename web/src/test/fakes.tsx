import type { ComponentChildren } from "preact";
import { vi } from "vitest";

import type { AudioResult, Health, ListResponse, Voice } from "../api/contracts";
import { ServicesProvider, type AppServices, type HttpPort } from "../app/services";
import type { RecorderPort } from "../media/recorder";
import type { HistoryEntry, HistoryLimits, HistoryRepository } from "../storage/history";

export type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

export interface FakeOptions {
  health?: DeepPartial<Health>;
  historyEntry?: DeepPartial<HistoryEntry>;
  historyEntries?: HistoryEntry[];
  voices?: Voice[];
  speechBlob?: Blob;
}

export function fakeServices(options: FakeOptions = {}) {
  const health = mergeHealth(options.health);
  const entries = options.historyEntries ?? [];
  const speechBlob = options.speechBlob ?? new Blob(["audio"], { type: "audio/mpeg" });
  const httpImpl = {
    lastForm: undefined as FormData | undefined,
    getJson: vi.fn(async (path: string): Promise<unknown> => {
      if (path === "/health") return health;
      if (path === "/v1/voices") {
        return { object: "list", data: options.voices ?? [] } as ListResponse<Voice>;
      }
      throw new Error(`GET non simulé : ${path}`);
    }),
    postJson: vi.fn(async (): Promise<unknown> => ({})),
    postForm: vi.fn(async (_path: string, form: FormData): Promise<unknown> => {
      httpImpl.lastForm = form;
      return {};
    }),
    postBlob: vi.fn(async (): Promise<AudioResult> => ({
      blob: speechBlob,
      contentType: speechBlob.type,
      filename: "speech.mp3",
    })),
    delete: vi.fn(async () => undefined),
  };
  const http = httpImpl as unknown as HttpPort & typeof httpImpl;

  const history: HistoryRepository = {
    add: vi.fn(async (draft) => ({
      ...draft,
      id: "history-1",
      createdAt: "2026-08-27T12:00:00.000Z",
    })),
    list: vi.fn(async () => entries),
    get: vi.fn(async (id) => {
      const complete = options.historyEntry ? mergeHistoryEntry(options.historyEntry) : undefined;
      return complete?.id === id ? complete : entries.find((entry) => entry.id === id);
    }),
    remove: vi.fn(async () => undefined),
    keepAudio: vi.fn(async () => ({ evictedIds: [] })),
    confirmAudioEviction: vi.fn(async () => undefined),
    clear: vi.fn(async () => undefined),
    getLimits: vi.fn(async () => ({ maxEntries: 100, maxAudioBytes: 262_144_000 })),
    setLimits: vi.fn(async (_limits: HistoryLimits) => undefined),
  };
  const recorderFactory = vi.fn<() => RecorderPort>(() => {
    throw new Error("Aucun faux microphone configuré.");
  });
  const services: AppServices = { http, history, recorderFactory };
  const wrapper = ({ children }: { children: ComponentChildren }) => (
    <ServicesProvider services={services}>{children}</ServicesProvider>
  );
  return { wrapper, http, history, recorderFactory };
}

function mergeHealth(partial: DeepPartial<Health> = {}): Health {
  const base: Health = {
    status: "ok",
    device: "cuda",
    asr_model: "nvidia/parakeet-tdt-0.6b-v3",
    diarization_model: "pyannote/speaker-diarization-3.1",
    diarization_enabled: true,
    vad_model: "nvidia/frame_vad_multilingual_marblenet_v2.0",
    vad_device: "cuda",
    vad_enabled: true,
    summary_model: "qwen2.5:7b",
    summary_enabled: true,
    gpu: { name: "GPU local" },
    tts: {
      enabled: true,
      worker: true,
      state: "ready",
      downloaded_models: ["Qwen3-TTS-12Hz-1.7B-CustomVoice"],
      loaded_model: "Qwen3-TTS-12Hz-1.7B-CustomVoice",
      precision: "fp16",
      device: "cuda",
      attention: "flash_attention_2",
      features: ["custom_voice", "clone", "voice_design"],
      last_error: null,
      pid: 42,
      vram_allocated_mib: 4096,
    },
  };
  return {
    ...base,
    ...partial,
    gpu: partial.gpu === undefined ? base.gpu : (partial.gpu as Record<string, unknown> | null),
    tts: { ...base.tts, ...partial.tts } as Health["tts"],
  };
}

function mergeHistoryEntry(partial: DeepPartial<HistoryEntry>): HistoryEntry {
  return {
    id: "history-1",
    createdAt: "2026-08-27T12:00:00.000Z",
    kind: "transcription",
    title: "Réunion",
    parameters: {},
    resultText: "Bonjour",
    metadata: {},
    ...partial,
  } as HistoryEntry;
}
