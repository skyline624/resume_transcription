import { describe, expect, it, vi } from "vitest";
import { PcmPlayer } from "./pcm-player";

function audioContext() {
  const sources: { start: ReturnType<typeof vi.fn>; stop: ReturnType<typeof vi.fn>; onended?: () => void }[] = [];
  const samples: Float32Array[] = [];
  const context = {
    currentTime: 0, destination: {}, resume: vi.fn(async () => undefined), close: vi.fn(async () => undefined),
    createBuffer: vi.fn(() => ({ copyToChannel: (value: Float32Array) => samples.push(value) })),
    createBufferSource: () => {
      const source = { start: vi.fn(), stop: vi.fn(), connect: vi.fn(), disconnect: vi.fn() };
      sources.push(source); return source;
    },
  };
  return { context, sources, samples };
}

describe("PcmPlayer", () => {
  it("joins split samples and schedules consecutive buffers without overlap", async () => {
    const { context, sources, samples } = audioContext();
    const player = new PcmPlayer(context as unknown as AudioContext);
    await player.start();
    player.push(new Uint8Array([0, 128, 255]));
    player.push(new Uint8Array([127]));
    expect(samples.map((item) => [...item])).toEqual([[-1], [32767 / 32768]]);
    expect(sources[0]?.start).toHaveBeenCalledWith(0.08);
    expect(sources[1]?.start).toHaveBeenCalledWith(0.08 + 1 / 24000);
    const finished = vi.fn();
    const result = player.finish().then(finished);
    await Promise.resolve();
    expect(finished).not.toHaveBeenCalled();
    sources.forEach((source) => source.onended?.());
    await result;
    player.stop();
    expect(context.close).toHaveBeenCalledOnce();
  });

  it("stops queued playback and releases a pending finish", async () => {
    const { context, sources } = audioContext();
    const player = new PcmPlayer(context as unknown as AudioContext);
    player.push(new Uint8Array([0, 0]));
    const result = player.finish();
    player.stop(); player.stop();
    await result;
    expect(sources[0]?.stop).toHaveBeenCalledOnce();
    expect(context.close).toHaveBeenCalledOnce();
  });
});
