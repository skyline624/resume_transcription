import { fireEvent, render, screen, waitFor } from "@testing-library/preact";
import { useState } from "preact/hooks";
import { describe, expect, it, vi } from "vitest";

import { AudioSourcePicker } from "./AudioSourcePicker";
import type { AudioSelection, RecordedAudio, RecorderPort, RecorderState } from "./recorder";

class FakeRecorder implements RecorderPort {
  current: RecorderState = "idle";
  start = vi.fn(async () => {
    this.current = "recording";
  });
  stop = vi.fn<() => Promise<RecordedAudio>>();
  cancel = vi.fn(() => {
    this.current = "idle";
  });
  state = vi.fn(() => this.current);
  mediaStream = vi.fn(() => null);
}

describe("AudioSourcePicker", () => {
  it("cancels an active microphone capture", async () => {
    const recorder = new FakeRecorder();
    render(
      <AudioSourcePicker
        value={null}
        onChange={() => undefined}
        recorderFactory={() => recorder}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Enregistrer au micro" }));
    await screen.findByRole("button", { name: "Annuler l’enregistrement" });
    fireEvent.click(screen.getByRole("button", { name: "Annuler l’enregistrement" }));

    expect(recorder.cancel).toHaveBeenCalledOnce();
  });

  it("revokes the previous object URL when the selected media changes", async () => {
    const createObjectURL = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValueOnce("blob:first")
      .mockReturnValueOnce("blob:second");
    const revokeObjectURL = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
    const first: AudioSelection = {
      blob: new File(["first"], "first.wav", { type: "audio/wav" }),
      filename: "first.wav",
      origin: "file",
    };
    const second: AudioSelection = {
      blob: new File(["second"], "second.wav", { type: "audio/wav" }),
      filename: "second.wav",
      origin: "file",
    };
    const view = render(<AudioSourcePicker value={first} onChange={() => undefined} />);

    view.rerender(<AudioSourcePicker value={second} onChange={() => undefined} />);

    await waitFor(() => expect(revokeObjectURL).toHaveBeenCalledWith("blob:first"));
    expect(createObjectURL).toHaveBeenCalledTimes(2);
  });

  it("marks a microphone reference shorter than three seconds as invalid", async () => {
    const recorder = new FakeRecorder();
    recorder.stop.mockResolvedValue({
      blob: new Blob(["voice"], { type: "audio/webm" }),
      filename: "voice.webm",
      durationMs: 2_900,
    });

    function Harness() {
      const [value, setValue] = useState<AudioSelection | null>(null);
      const [valid, setValid] = useState(false);
      return (
        <>
          <AudioSourcePicker
            value={value}
            onChange={setValue}
            onValidityChange={setValid}
            recorderFactory={() => recorder}
            referenceMode
          />
          <button disabled={!valid}>Envoyer la référence</button>
        </>
      );
    }

    render(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "Enregistrer au micro" }));
    await screen.findByRole("button", { name: "Arrêter l’enregistrement" });
    fireEvent.click(screen.getByRole("button", { name: "Arrêter l’enregistrement" }));

    expect(await screen.findByText(/au moins 3 secondes/i)).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: "Envoyer la référence" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });
});
