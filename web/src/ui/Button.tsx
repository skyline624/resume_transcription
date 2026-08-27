import type { ComponentChildren, JSX, Ref } from "preact";

type ButtonVariant = "primary" | "secondary" | "danger";

interface ButtonProps extends JSX.ButtonHTMLAttributes<HTMLButtonElement> {
  children: ComponentChildren;
  variant?: ButtonVariant;
  buttonRef?: Ref<HTMLButtonElement>;
}

export function Button({
  children,
  variant = "secondary",
  buttonRef,
  class: className,
  ...props
}: ButtonProps) {
  return (
    <button
      class={`button button--${variant}${className ? ` ${className}` : ""}`}
      ref={buttonRef}
      {...props}
    >
      {children}
    </button>
  );
}
