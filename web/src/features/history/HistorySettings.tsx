import { useEffect, useMemo, useState } from "preact/hooks";

import type { HistoryEntry, HistoryRepository } from "../../storage/history";
import { Button } from "../../ui/Button";
import { ConfirmDialog } from "../../ui/ConfirmDialog";

const MIB = 1024 * 1024;

export function HistorySettings({ repository, entries, onCleared }: {
  repository: HistoryRepository;
  entries: HistoryEntry[];
  onCleared(): void;
}) {
  const [maxEntries, setMaxEntries] = useState(100);
  const [maxAudioMiB, setMaxAudioMiB] = useState(250);
  const [maxAllowedMiB, setMaxAllowedMiB] = useState(2048);
  const [confirming, setConfirming] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const usedBytes = useMemo(
    () => entries.reduce((total, entry) => total + (entry.audio?.size ?? 0), 0),
    [entries],
  );

  useEffect(() => {
    void repository.getLimits().then((limits) => {
      setMaxEntries(limits.maxEntries);
      setMaxAudioMiB(Math.round(limits.maxAudioBytes / MIB));
    });
    void navigator.storage?.estimate().then((estimate) => {
      if (!estimate.quota) return;
      setMaxAllowedMiB(Math.max(10, Math.floor(Math.min(2 * 1024, (estimate.quota * 0.8) / MIB))));
    });
  }, [repository]);

  const valid = maxEntries >= 10 && maxEntries <= 1000 && maxAudioMiB >= 10 && maxAudioMiB <= maxAllowedMiB;
  const save = async () => {
    if (!valid) return;
    await repository.setLimits({ maxEntries, maxAudioBytes: maxAudioMiB * MIB });
    setMessage("Limites enregistrées dans ce navigateur.");
  };
  const clear = async () => {
    await repository.clear();
    setConfirming(false);
    onCleared();
  };

  return (
    <details class="history-settings">
      <summary>Rétention locale</summary>
      <div class="workflow-form">
        <p class="storage-usage">{entries.length} opération(s) · {formatBytes(usedBytes)} d’audio conservé</p>
        <div class="options-grid options-grid--always">
          <label>
            <span>Nombre maximal d’opérations</span>
            <input aria-label="Nombre maximal d’opérations" max="1000" min="10" onInput={(event) => setMaxEntries(Number(event.currentTarget.value))} type="number" value={maxEntries} />
          </label>
          <label>
            <span>Limite audio (MiB)</span>
            <input aria-label="Limite audio" max={maxAllowedMiB} min="10" onInput={(event) => setMaxAudioMiB(Number(event.currentTarget.value))} type="number" value={maxAudioMiB} />
          </label>
        </div>
        {!valid ? <p class="field__error">Choisissez 10 à 1 000 opérations et 10 à {maxAllowedMiB} MiB d’audio.</p> : null}
        {message ? <p role="status">{message}</p> : null}
        <div class="result-actions">
          <Button disabled={!valid} onClick={() => void save()} variant="primary">Enregistrer les limites</Button>
          <Button onClick={() => setConfirming(true)} variant="danger">Effacer l’historique local</Button>
        </div>
      </div>
      <ConfirmDialog
        confirmLabel="Tout effacer"
        danger
        onCancel={() => setConfirming(false)}
        onConfirm={() => void clear()}
        open={confirming}
        title="Effacer tout l’historique local ?"
      >
        Toutes les opérations et tous les audios conservés seront supprimés de ce navigateur.
      </ConfirmDialog>
    </details>
  );
}

function formatBytes(bytes: number): string {
  return bytes < MIB ? `${Math.round(bytes / 1024)} KiB` : `${(bytes / MIB).toFixed(1)} MiB`;
}
