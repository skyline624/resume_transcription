import { useCallback, useEffect, useState } from "preact/hooks";

import type { Voice } from "../../api/contracts";
import { useServices } from "../../app/services";
import { Button } from "../../ui/Button";
import { ConfirmDialog } from "../../ui/ConfirmDialog";
import { VoiceEnrollment } from "./VoiceEnrollment";
import { deleteVoice, listVoices } from "./voices-api";

export function VoicesPage() {
  const { http } = useServices();
  const [voices, setVoices] = useState<Voice[]>([]);
  const [deleting, setDeleting] = useState<Voice | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setVoices((await listVoices(http)).data);
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "La liste des voix est indisponible.");
    }
  }, [http]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const confirmDelete = async () => {
    if (!deleting) return;
    try {
      await deleteVoice(http, deleting.id);
      setDeleting(null);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "La voix n’a pas pu être supprimée.");
    }
  };

  const builtins = voices.filter((voice) => voice.kind === "builtin");
  const clones = voices.filter((voice) => voice.kind === "clone");
  return (
    <div class="workflow-stack">
      {error ? <p class="field__error" role="alert">{error}</p> : null}
      <VoiceGroup title="Voix prédéfinies" voices={builtins} />
      <VoiceGroup onDelete={setDeleting} title="Mes voix consenties" voices={clones} />
      <VoiceEnrollment onCreated={() => void refresh()} />
      <ConfirmDialog
        confirmLabel="Supprimer"
        danger
        onCancel={() => setDeleting(null)}
        onConfirm={() => void confirmDelete()}
        open={Boolean(deleting)}
        title={`Supprimer ${deleting?.name ?? "cette voix"} ?`}
      >
        Le profil et sa référence seront supprimés du volume Docker. Cette action est irréversible.
      </ConfirmDialog>
    </div>
  );
}

function VoiceGroup({ title, voices, onDelete }: {
  title: string;
  voices: Voice[];
  onDelete?: (voice: Voice) => void;
}) {
  return (
    <section class="voice-group">
      <h2>{title}</h2>
      {voices.length === 0 ? <p class="empty-guidance">Aucune voix dans ce groupe.</p> : (
        <div class="voice-grid">
          {voices.map((voice) => (
            <article class="voice-card" key={voice.id}>
              <div class="voice-card__mark" aria-hidden="true">{voice.name.slice(0, 1).toUpperCase()}</div>
              <div>
                <h3>{voice.name}</h3>
                <p>{voice.kind === "builtin" ? "Prédéfinie" : `${voice.language ?? "—"} · ${formatDuration(voice.duration)}`}</p>
                {voice.transcript_source ? <small>Transcription : {voice.transcript_source === "parakeet" ? "Parakeet" : "fournie"}</small> : null}
              </div>
              {voice.kind === "clone" && onDelete ? (
                <Button onClick={() => onDelete(voice)} variant="danger">Supprimer {voice.name}</Button>
              ) : null}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function formatDuration(duration?: number): string {
  return duration == null ? "durée inconnue" : `${duration.toFixed(1)} s`;
}
