import { useEffect, useState } from "preact/hooks";

import type { AudioResult } from "../../api/contracts";
import type { HistoryRepository } from "../../storage/history";
import { Button } from "../../ui/Button";
import { ConfirmDialog } from "../../ui/ConfirmDialog";

export function SpeechResult({
  audio,
  historyId,
  history,
}: {
  audio: AudioResult;
  historyId: string;
  history: HistoryRepository;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [evictedIds, setEvictedIds] = useState<string[] | null>(null);
  const [kept, setKept] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const next = URL.createObjectURL(audio.blob);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [audio]);

  const keep = async () => {
    try {
      const proposal = await history.keepAudio(historyId, audio.blob);
      if (proposal.evictedIds.length > 0) setEvictedIds(proposal.evictedIds);
      else setKept(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "L’audio n’a pas pu être conservé.");
    }
  };

  const confirm = async () => {
    if (!evictedIds) return;
    await history.confirmAudioEviction(historyId, audio.blob, evictedIds);
    setEvictedIds(null);
    setKept(true);
  };

  return (
    <section class="speech-result" aria-label="Audio généré">
      <p class="eyebrow">Audio prêt</p>
      {url ? <audio controls src={url} /> : null}
      <div class="result-actions">
        {url ? <a class="button button--secondary action-link" download={audio.filename ?? "speech"} href={url}>Télécharger</a> : null}
        <Button disabled={kept} onClick={() => void keep()}>{kept ? "Conservé" : "Conserver"}</Button>
      </div>
      {error ? <p class="field__error" role="alert">{error}</p> : null}
      <ConfirmDialog
        confirmLabel="Libérer et conserver"
        onCancel={() => setEvictedIds(null)}
        onConfirm={() => void confirm()}
        open={Boolean(evictedIds)}
        title="Libérer les anciens audios ?"
      >
        Cette action retire l’audio de {evictedIds?.length ?? 0} ancienne(s) entrée(s), sans supprimer leur texte.
      </ConfirmDialog>
    </section>
  );
}
