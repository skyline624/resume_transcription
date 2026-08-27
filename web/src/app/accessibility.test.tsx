import { fireEvent, render, screen } from "@testing-library/preact";
import { afterEach, describe, expect, it } from "vitest";

import { fakeServices } from "../test/fakes";
import { App } from "./App";

afterEach(() => {
  window.location.hash = "";
});

describe("accessible application shell", () => {
  it("keeps one page title and moves focus to it after navigation", () => {
    render(<App />, { wrapper: fakeServices().wrapper });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("navigation", { name: "Navigation principale" })).toBeTruthy();

    fireEvent.click(screen.getByRole("link", { name: "Voix" }));
    window.dispatchEvent(new HashChangeEvent("hashchange"));

    const heading = screen.getByRole("heading", { level: 1, name: "Voix" });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(document.activeElement).toBe(heading);
  });

  it("gives every visible form control an accessible name", () => {
    render(<App />, { wrapper: fakeServices().wrapper });

    for (const route of ["Transcrire", "Résumer", "Synthétiser", "Voix", "Historique"]) {
      fireEvent.click(screen.getByRole("link", { name: route }));
      window.dispatchEvent(new HashChangeEvent("hashchange"));
      expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
      for (const control of document.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(
        "input:not([type='hidden']), select, textarea",
      )) {
        const labelled = control.labels && control.labels.length > 0;
        expect(labelled || Boolean(control.getAttribute("aria-label"))).toBe(true);
      }
    }
  });
});
