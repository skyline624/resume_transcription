import { render, screen } from "@testing-library/preact";
import { describe, expect, it } from "vitest";

import { fakeServices } from "../../test/fakes";
import { VoiceEnrollment } from "./VoiceEnrollment";

describe("VoiceEnrollment", () => {
  it("forbids enrollment without explicit consent", () => {
    render(<VoiceEnrollment onCreated={() => undefined} />, {
      wrapper: fakeServices().wrapper,
    });

    const button = screen.getByRole("button", { name: "Enregistrer la voix" });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByLabelText(/confirme avoir le droit/i) as HTMLInputElement).checked).toBe(false);
  });
});
