/** Schedules mono signed little-endian PCM as soon as complete samples arrive. */
export class PcmPlayer {
  private tail: number | undefined;
  private nextTime = 0;
  private sources = new Set<AudioBufferSourceNode>();
  private finished = false;
  private stopped = false;
  private drained: (() => void) | undefined;

  constructor(private readonly context = new AudioContext()) {}

  async start(): Promise<void> { await this.context.resume(); }

  push = (chunk: Uint8Array): void => {
    if (this.stopped) return;
    const bytes = new Uint8Array(chunk.length + (this.tail === undefined ? 0 : 1));
    if (this.tail !== undefined) bytes[0] = this.tail;
    bytes.set(chunk, this.tail === undefined ? 0 : 1);
    this.tail = bytes.length % 2 ? bytes[bytes.length - 1] : undefined;
    const count = Math.floor(bytes.length / 2);
    if (!count) return;
    const samples = new Float32Array(count);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < count; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
    const buffer = this.context.createBuffer(1, count, 24000);
    buffer.copyToChannel(samples, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    this.sources.add(source);
    source.onended = () => {
      source.disconnect();
      this.sources.delete(source);
      if (this.finished && this.sources.size === 0) this.drained?.();
    };
    const time = Math.max(this.context.currentTime + 0.08, this.nextTime);
    source.start(time);
    this.nextTime = time + count / 24000;
  };

  async finish(): Promise<void> {
    if (this.tail !== undefined) throw new Error("Le flux audio est incomplet.");
    this.finished = true;
    if (!this.stopped && this.sources.size) await new Promise<void>((resolve) => { this.drained = resolve; });
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    for (const source of this.sources) { source.stop(); source.disconnect(); }
    this.sources.clear();
    this.drained?.();
    void this.context.close().catch(() => undefined);
  }
}
