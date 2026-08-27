export interface RecordedAudio {
  blob: Blob;
  filename: string;
  durationMs: number;
}

export interface AudioSelection {
  blob: Blob;
  filename: string;
  durationMs?: number;
  origin: "file" | "recording";
}

export type RecorderState = "idle" | "recording" | "stopping";

export interface MediaRecorderPort {
  ondataavailable: ((event: { data: Blob }) => void) | null;
  onstop: (() => void) | null;
  readonly mimeType?: string;
  start(): void;
  stop(): void;
}

export interface RecorderPort {
  start(): Promise<void>;
  stop(): Promise<RecordedAudio>;
  cancel(): void;
  state(): RecorderState;
  mediaStream(): MediaStream | null;
}

type StreamFactory = () => Promise<MediaStream>;
type RecorderFactory = (stream: MediaStream, mimeType?: string) => MediaRecorderPort;

const MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/ogg;codecs=opus",
  "audio/webm",
] as const;

export class BrowserRecorder implements RecorderPort {
  private currentState: RecorderState = "idle";
  private stream: MediaStream | null = null;
  private recorder: MediaRecorderPort | null = null;
  private chunks: Blob[] = [];
  private startedAt = 0;
  private stopResolve: ((audio: RecordedAudio) => void) | null = null;
  private stopReject: ((reason: unknown) => void) | null = null;

  constructor(
    private readonly getStream: StreamFactory = () =>
      navigator.mediaDevices.getUserMedia({ audio: true }),
    private readonly now: () => number = Date.now,
    private readonly createRecorder: RecorderFactory = createNativeRecorder,
    private readonly supportsMimeType: (mimeType: string) => boolean = supportsNativeMimeType,
  ) {}

  async start(): Promise<void> {
    if (this.currentState !== "idle") {
      throw new Error("Un enregistrement est déjà en cours.");
    }
    this.chunks = [];
    try {
      this.stream = await this.getStream();
      const mimeType = MIME_CANDIDATES.find(this.supportsMimeType);
      this.recorder = this.createRecorder(this.stream, mimeType);
      this.recorder.ondataavailable = ({ data }) => {
        if (data.size > 0) this.chunks.push(data);
      };
      this.recorder.onstop = () => this.finishCapture();
      this.startedAt = this.now();
      this.currentState = "recording";
      this.recorder.start();
    } catch (error) {
      this.reset();
      throw error;
    }
  }

  stop(): Promise<RecordedAudio> {
    if (this.currentState !== "recording" || !this.recorder) {
      return Promise.reject(new Error("Aucun enregistrement en cours."));
    }
    this.currentState = "stopping";
    return new Promise<RecordedAudio>((resolve, reject) => {
      this.stopResolve = resolve;
      this.stopReject = reject;
      try {
        this.recorder?.stop();
      } catch (error) {
        this.stopResolve = null;
        this.stopReject = null;
        this.reset();
        reject(error);
      }
    });
  }

  cancel(): void {
    const pendingReject = this.stopReject;
    this.stopResolve = null;
    this.stopReject = null;
    if (this.recorder && this.currentState !== "idle") {
      try {
        this.recorder.stop();
      } catch {
        // The browser may already have moved the recorder to inactive.
      }
    }
    this.reset();
    pendingReject?.(new Error("Enregistrement annulé."));
  }

  state(): RecorderState {
    return this.currentState;
  }

  mediaStream(): MediaStream | null {
    return this.stream;
  }

  private finishCapture(): void {
    if (!this.stopResolve) {
      this.reset();
      return;
    }
    const resolve = this.stopResolve;
    const mimeType = this.recorder?.mimeType || this.chunks[0]?.type || "audio/webm";
    const blob = new Blob(this.chunks, { type: mimeType });
    const durationMs = Math.max(0, this.now() - this.startedAt);
    const filename = `enregistrement-${new Date(this.startedAt).toISOString().replace(/[:.]/g, "-")}.${extensionFor(mimeType)}`;
    this.stopResolve = null;
    this.stopReject = null;
    this.reset();
    resolve({ blob, filename, durationMs });
  }

  private reset(): void {
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    if (this.recorder) {
      this.recorder.ondataavailable = null;
      this.recorder.onstop = null;
    }
    this.stream = null;
    this.recorder = null;
    this.chunks = [];
    this.currentState = "idle";
  }
}

function createNativeRecorder(stream: MediaStream, mimeType?: string): MediaRecorderPort {
  return new MediaRecorder(stream, mimeType ? { mimeType } : undefined) as unknown as MediaRecorderPort;
}

function supportsNativeMimeType(mimeType: string): boolean {
  return typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(mimeType);
}

function extensionFor(mimeType: string): "webm" | "ogg" {
  return mimeType.toLowerCase().includes("ogg") ? "ogg" : "webm";
}
