import { fireEvent, render, screen } from "@testing-library/preact";
import { afterEach, describe, expect, it } from "vitest";

import { fakeServices } from "../test/fakes";
import { App } from "./App";

describe("App", () => {
  afterEach(() => {
    window.location.hash = "";
  });

  it("ouvre Transcrire et expose les cinq destinations", () => {
    render(<App />, { wrapper: fakeServices().wrapper });

    expect(screen.getByRole("heading", { name: "Transcrire" })).toBeTruthy();
    for (const name of [
      "Transcrire",
      "Résumer",
      "Synthétiser",
      "Voix",
      "Historique",
    ]) {
      expect(screen.getByRole("link", { name })).toBeTruthy();
    }
  });

  it("suit une ancre sans requête serveur", () => {
    render(<App />, { wrapper: fakeServices().wrapper });

    fireEvent.click(screen.getByRole("link", { name: "Voix" }));
    window.dispatchEvent(new HashChangeEvent("hashchange"));

    expect(window.location.hash).toBe("#/voices");
    expect(screen.getByRole("heading", { name: "Voix" })).toBeTruthy();
  });
});
