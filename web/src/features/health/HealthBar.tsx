import { useState } from "preact/hooks";

import { HealthPanel } from "./HealthPanel";
import { useHealth } from "./use-health";

export function HealthBar() {
  const { health, error } = useHealth();
  const [open, setOpen] = useState(false);
  const offline = Boolean(error && !health);
  const degraded = Boolean(health && (health.status !== "ok" || (health.tts.enabled && !health.tts.worker)));
  const label = offline ? "Hors ligne" : degraded ? "Dégradé" : health ? "Opérationnel" : "Connexion…";

  return (
    <div class="health-bar">
      <button
        aria-expanded={open}
        aria-label="Détails du serveur"
        class={`server-pill server-pill--${offline ? "offline" : degraded ? "degraded" : "ready"}`}
        onClick={() => setOpen((value) => !value)}
        type="button"
      >
        <span class="server-pill__dot" aria-hidden="true" />
        <span>
          <strong>{label}</strong>
          {health?.tts.enabled && !health.tts.worker ? <small>Synthèse indisponible</small> : null}
        </span>
      </button>
      {open ? <HealthPanel error={error} health={health} onClose={() => setOpen(false)} /> : null}
    </div>
  );
}
