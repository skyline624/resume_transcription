import { useEffect, useState } from "preact/hooks";

import { routeHref } from "../../app/routes";
import type { HistoryEntry } from "../../storage/history";
import { Button } from "../../ui/Button";
import { ConfirmDialog } from "../../ui/ConfirmDialog";

export function HistoryDetail({ entry, onDelete }: {
  entry: HistoryEntry;
  onDelete(id: string): Promise<void>;
}) {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (!entry.audio) {
      setAudioUrl(null);
      return;
    }
    const url = URL.createObjectURL(entry.audio);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [entry]);

  return (
    <section class="history-detail" aria-label={`Détail ${entry.title}`}>
      <div class="result-panel__heading">
        <div>
          <p class="eyebrow">{kindLabel(entry.kind)}</p>
          <h2>{entry.title}</h2>
        </div>
        <time>{new Date(entry.createdAt).toLocaleString("fr-FR")}</time>
      </div>
      <div class="history-result">{entry.resultText}</div>
      <details>
        <summary>Paramètres et métadonnées</summary>
        <pre>{JSON.stringify({ parameters: entry.parameters, metadata: entry.metadata }, null, 2)}</pre>
      </details>
      {audioUrl ? <audio controls src={audioUrl} /> : null}
      <div class="result-actions">
        <a
          class="button button--secondary action-link"
          href={`${routeHref("summarize")}?history=${encodeURIComponent(entry.id)}`}
          onClick={() => { window.location.hash = `${routeHref("summarize")}?history=${encodeURIComponent(entry.id)}`; }}
        >Résumer ce texte</a>
        <a
          class="button button--secondary action-link"
          href={`${routeHref("speech")}?history=${encodeURIComponent(entry.id)}`}
          onClick={() => { window.location.hash = `${routeHref("speech")}?history=${encodeURIComponent(entry.id)}`; }}
        >Synthétiser ce texte</a>
        {audioUrl ? <a class="button button--secondary action-link" download="audio-conserve" href={audioUrl}>Télécharger l’audio</a> : null}
        <Button onClick={() => setConfirming(true)} variant="danger">Supprimer cette entrée</Button>
      </div>
      <ConfirmDialog
        confirmLabel="Supprimer"
        danger
        onCancel={() => setConfirming(false)}
        onConfirm={() => void onDelete(entry.id)}
        open={confirming}
        title={`Supprimer ${entry.title} ?`}
      >
        Le texte, les métadonnées et l’éventuel audio conservé seront supprimés de ce navigateur.
      </ConfirmDialog>
    </section>
  );
}

function kindLabel(kind: HistoryEntry["kind"]): string {
  return kind === "transcription" ? "Transcription" : kind === "summary" ? "Résumé" : "Synthèse vocale";
}
