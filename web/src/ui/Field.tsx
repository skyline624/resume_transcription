import type { ComponentChildren } from "preact";

interface FieldProps {
  id: string;
  label: string;
  hint?: string;
  error?: string;
  children: ComponentChildren;
}

export function Field({ id, label, hint, error, children }: FieldProps) {
  const describedBy = [hint ? `${id}-hint` : "", error ? `${id}-error` : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <div class="field" data-describedby={describedBy || undefined}>
      <label class="field__label" for={id}>
        {label}
      </label>
      {children}
      {hint && (
        <p class="field__hint" id={`${id}-hint`}>
          {hint}
        </p>
      )}
      {error && (
        <p class="field__error" id={`${id}-error`} role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
