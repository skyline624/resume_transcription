import { render, screen } from "@testing-library/preact";
import { describe, expect, it } from "vitest";

import { fakeServices } from "../../test/fakes";
import { OneShotClone } from "./OneShotClone";

describe("OneShotClone", () => {
  it("never preselects voice consent", () => {
    render(<OneShotClone />, { wrapper: fakeServices().wrapper });

    expect((screen.getByLabelText(/confirme avoir le droit/i) as HTMLInputElement).checked).toBe(false);
    expect((screen.getByRole("button", { name: "Cloner et synthétiser" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
