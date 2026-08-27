import { useEffect, useState } from "preact/hooks";

interface OperationStatusProps {
  active: boolean;
  label: string;
  startedAt: number;
}

function elapsedLabel(startedAt: number, now: number): string {
  const total = Math.max(0, Math.floor((now - startedAt) / 1_000));
  const minutes = Math.floor(total / 60).toString().padStart(2, "0");
  const seconds = (total % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

export function OperationStatus({
  active,
  label,
  startedAt,
}: OperationStatusProps) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [active, startedAt]);

  if (!active) return null;
  return (
    <p class="operation-status" aria-live="polite">
      <span class="operation-status__pulse" aria-hidden="true" />
      {label} — <time>{elapsedLabel(startedAt, now)}</time>
    </p>
  );
}
