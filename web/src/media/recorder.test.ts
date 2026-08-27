// @vitest-environment node
import { describe, expect, it, vi } from "vitest";

import { BrowserRecorder, type MediaRecorderPort } from "./recorder";

class FakeMediaRecorder implements MediaRecorderPort {
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  start = vi.fn();
  stop = vi.fn();

  finish(blob: Blob): void {
    this.ondataavailable?.({ data: blob });
    this.onstop?.();
  }
}

describe("BrowserRecorder", () => {
  it("stops every media track after capture", async () => {
    const stopTrack = vi.fn();
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream;
    const mediaRecorder = new FakeMediaRecorder();
    let now = 1_000;
    const recorder = new BrowserRecorder(
      async () => stream,
      () => now,
      () => mediaRecorder,
    );
    await recorder.start();
    const pending = recorder.stop();
    now = 4_500;
    mediaRecorder.finish(new Blob(["voice"], { type: "audio/webm" }));

    const result = await pending;

    expect(result.durationMs).toBe(3_500);
    expect(result.filename).toMatch(/^enregistrement-.*\.webm$/);
    expect(stopTrack).toHaveBeenCalledOnce();
  });

  it("cancels capture and releases every track without producing audio", async () => {
    const stopTrack = vi.fn();
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream;
    const mediaRecorder = new FakeMediaRecorder();
    const recorder = new BrowserRecorder(
      async () => stream,
      () => 1_000,
      () => mediaRecorder,
    );
    await recorder.start();

    recorder.cancel();

    expect(mediaRecorder.stop).toHaveBeenCalledOnce();
    expect(stopTrack).toHaveBeenCalledOnce();
    expect(recorder.state()).toBe("idle");
  });
});
