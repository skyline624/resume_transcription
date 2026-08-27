import { render, screen } from "@testing-library/preact";
import { describe, expect, it } from "vitest";

import { fakeServices } from "../../test/fakes";
import { VoicesPage } from "./VoicesPage";

describe("VoicesPage", () => {
  it("never offers deletion for a built-in voice", async () => {
    render(<VoicesPage />, {
      wrapper: fakeServices({
        voices: [
          { id: "Ryan", name: "Ryan", kind: "builtin" },
          {
            id: "clone-1",
            name: "Ma voix",
            kind: "clone",
            language: "fr",
            duration: 8,
          },
        ],
      }).wrapper,
    });

    expect(await screen.findByText("Ryan")).toBeTruthy();
    expect(screen.getAllByRole("button", { name: /Supprimer/ })).toHaveLength(1);
  });
});
