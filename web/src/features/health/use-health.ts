import { useCallback, useEffect, useState } from "preact/hooks";

import type { Health } from "../../api/contracts";
import { useServices } from "../../app/services";
import { getHealth } from "./health-api";

export interface HealthState {
  health: Health | null;
  error: Error | null;
  updatedAt: Date | null;
  refresh(): Promise<void>;
}

export function useHealth(): HealthState {
  const { http } = useServices();
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await getHealth(http);
      setHealth(next);
      setError(null);
      setUpdatedAt(new Date());
    } catch (reason) {
      setError(reason instanceof Error ? reason : new Error("Serveur injoignable."));
    }
  }, [http]);

  useEffect(() => {
    let timer = 0;
    let disposed = false;
    const schedule = () => {
      window.clearTimeout(timer);
      const delay = document.visibilityState === "visible" ? 10_000 : 30_000;
      timer = window.setTimeout(async () => {
        await refresh();
        if (!disposed) schedule();
      }, delay);
    };
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") void refresh();
      schedule();
    };

    void refresh();
    schedule();
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      disposed = true;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [refresh]);

  return { health, error, updatedAt, refresh };
}
