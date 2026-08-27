import { fireEvent, render, screen } from "@testing-library/preact";
import { describe, expect, it } from "vitest";

import { fakeServices } from "../../test/fakes";
import { SpeechPage } from "./SpeechPage";

describe("SpeechPage", () => {
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
