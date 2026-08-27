import { fireEvent, render, screen } from "@testing-library/preact";
import { describe, expect, it } from "vitest";

import { fakeServices } from "../../test/fakes";
import { TranscriptionPage } from "./TranscriptionPage";

describe("TranscriptionPage", () => {
  it("sends a diarized audio file and stores the text result locally", async () => {
    const services = fakeServices();
    render(<TranscriptionPage />, { wrapper: services.wrapper });
    fireEvent.change(screen.getByLabelText("Fichier audio"), {
      target: { files: [new File(["audio"], "réunion.wav", { type: "audio/wav" })] },
    });
    fireEvent.click(screen.getByLabelText("Séparer les locuteurs"));
    fireEvent.click(screen.getByRole("button", { name: "Transcrire" }));

    expect(await screen.findByText("Bonjour à tous")).toBeTruthy();
    expect(services.http.lastForm?.get("diarize")).toBe("true");
    expect(services.history.add).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "transcription" }),
    );
  });
});
