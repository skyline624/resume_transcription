import { render, screen } from "@testing-library/preact";
import { expect, it } from "vitest";

import { ConfirmDialog } from "./ConfirmDialog";

it("place le focus sur Annuler", () => {
  render(
    <ConfirmDialog
      open
      title="Supprimer ?"
      confirmLabel="Supprimer"
      danger
      onConfirm={() => undefined}
      onCancel={() => undefined}
    >
      Cette action est définitive.
    </ConfirmDialog>,
  );

  expect(document.activeElement).toBe(
    screen.getByRole("button", { name: "Annuler" }),
  );
});
