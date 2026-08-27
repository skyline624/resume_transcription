import type { Health } from "../../api/contracts";
import { Button } from "../../ui/Button";

export function HealthPanel({ health, error, onClose }: {
  health: Health | null;
  error: Error | null;
  onClose(): void;
}) {
  return (
    <aside aria-label="État détaillé du serveur" class="health-panel">
      <div class="health-panel__heading">
        <div>
          <p class="eyebrow">Serveur local</p>
          <h2>État des modèles</h2>
        </div>
        <Button onClick={onClose}>Fermer</Button>
      </div>
      {error ? <p class="field__error">{error.message}</p> : null}
      {health ? (
        <dl class="health-grid">
          <HealthMetric label="GPU" value={readGpuName(health.gpu) ?? health.device} />
          <HealthMetric
            label="VRAM Qwen"
            value={health.tts.vram_allocated_mib == null ? "—" : `${health.tts.vram_allocated_mib} MiB`}
          />
          <HealthMetric label="Parakeet" value={health.asr_model} />
          <HealthMetric
            label="Diarisation"
            value={health.diarization_enabled ? health.diarization_model : "Désactivée"}
          />
          <HealthMetric
            label="Résumé"
            value={health.summary_enabled ? health.summary_model : "Désactivé"}
          />
          <HealthMetric
            label="Qwen"
            value={health.tts.worker ? health.tts.loaded_model ?? health.tts.state : "Indisponible"}
          />
          <HealthMetric label="Précision" value={health.tts.precision ?? "—"} />
          <HealthMetric
            label="Modèles téléchargés"
            value={health.tts.downloaded_models.join(", ") || "Aucun"}
          />
        </dl>
      ) : (
        <p>En attente de la première réponse du serveur.</p>
      )}
      {health?.tts.last_error ? (
        <details>
          <summary>Erreur technique Qwen</summary>
          <code>{health.tts.last_error}</code>
        </details>
      ) : null}
    </aside>
  );
}

function HealthMetric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function readGpuName(gpu: Record<string, unknown> | null): string | null {
  const name = gpu?.name;
  return typeof name === "string" ? name : null;
}
