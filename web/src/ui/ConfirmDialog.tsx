import type { ComponentChildren } from "preact";
import { useEffect, useRef } from "preact/hooks";

import { Button } from "./Button";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  confirmLabel: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  children: ComponentChildren;
}

export function ConfirmDialog({
  open,
  title,
  confirmLabel,
  danger = false,
  onConfirm,
  onCancel,
  children,
}: ConfirmDialogProps) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    previousFocus.current = document.activeElement as HTMLElement | null;
    cancelRef.current?.focus();
    return () => previousFocus.current?.focus();
  }, [open]);

  if (!open) return null;
  return (
    <div class="dialog-backdrop" role="presentation">
      <section
        class="confirm-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
      >
        <h2 id="confirm-dialog-title">{title}</h2>
        <div class="confirm-dialog__body">{children}</div>
        <div class="confirm-dialog__actions">
          <Button buttonRef={cancelRef} onClick={onCancel}>
            Annuler
          </Button>
          <Button
            variant={danger ? "danger" : "primary"}
            onClick={onConfirm}
          >
            {confirmLabel}
          </Button>
        </div>
      </section>
    </div>
  );
}
