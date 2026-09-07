import { fireEvent, render, screen, waitFor } from "@testing-library/preact";
import { describe, expect, it, vi } from "vitest";

import { fakeServices } from "../../test/fakes";
import { SpeechPage } from "./SpeechPage";

describe("SpeechPage", () => {
  it("aborts an in-flight live request without saving partial audio", async () => {
    const close = vi.fn(async () => undefined);
    vi.stubGlobal("AudioContext", class {
      resume = async () => undefined;
      close = close;
    });
    try {
      const services = fakeServices({ voices: [{ id: "Ryan", name: "Ryan", kind: "builtin" }] });
      let signal: AbortSignal | undefined;
      services.http.postStream.mockImplementation(async (_path, _body, _chunk, init) => {
        signal = init?.signal ?? undefined;
        return new Promise((_resolve, reject) => signal?.addEventListener("abort", () => reject(signal?.reason)));
      });
      render(<SpeechPage />, { wrapper: services.wrapper });
      await screen.findByRole("option", { name: "Ryan" });
      fireEvent.input(screen.getByLabelText("Texte à prononcer"), { target: { value: "Bonjour" } });
      fireEvent.click(screen.getByRole("button", { name: "Écouter en direct" }));
      await waitFor(() => expect(services.http.postStream).toHaveBeenCalledOnce());
      expect(services.http.postStream.mock.calls[0]?.[1]).toMatchObject({ stream: true, response_format: "pcm" });
      fireEvent.click(screen.getByRole("button", { name: "Arrêter" }));
      await waitFor(() => expect(screen.queryByRole("button", { name: "Arrêter" })).toBeNull());
      expect(signal?.aborted).toBe(true);
      expect(close).toHaveBeenCalledOnce();
      expect(services.history.add).not.toHaveBeenCalled();
      expect(screen.queryByRole("alert")).toBeNull();
    } finally { vi.unstubAllGlobals(); }
  });

  it("requires an instruction only for VoiceDesign", () => {
    render(<SpeechPage />, { wrapper: fakeServices().wrapper });
    fireEvent.change(screen.getByLabelText("Mode vocal"), {
      target: { value: "qwen3-tts-voice-design" },
    });

    expect(screen.getByLabelText("Description de la voix")).toBeTruthy();
    expect(screen.queryByLabelText("Voix")).toBeNull();
  });

  it("does not retain generated audio until the user explicitly asks", async () => {
    const services = fakeServices({
      voices: [{ id: "Ryan", name: "Ryan", kind: "builtin" }],
      speechBlob: new Blob(["mp3"], { type: "audio/mpeg" }),
    });
    render(<SpeechPage />, { wrapper: services.wrapper });
    await screen.findByRole("option", { name: "Ryan" });
    fireEvent.input(screen.getByLabelText("Texte à prononcer"), {
      target: { value: "Bonjour" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Créer l’audio" }));

    await screen.findByRole("button", { name: "Conserver" });
    expect(services.history.keepAudio).not.toHaveBeenCalled();
  });
});
