import { useCallback, useEffect, useMemo, useState } from "preact/hooks";

import { useServices } from "../../app/services";
import type { HistoryEntry, HistoryKind } from "../../storage/history";
import { HistoryDetail } from "./HistoryDetail";
import { HistorySettings } from "./HistorySettings";

type Filter = "all" | HistoryKind;

export function HistoryPage() {
  const { history } = useServices();
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setEntries(await history.list());
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "L’historique local est indisponible.");
    }
  }, [history]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const filtered = useMemo(
    () => entries.filter((entry) => filter === "all" || entry.kind === filter),
    [entries, filter],
  );
  const selected = entries.find((entry) => entry.id === selectedId) ?? null;
  const remove = async (id: string) => {
    await history.remove(id);
    setSelectedId(null);
    await refresh();
  };

  return (
    <div class="workflow-stack">
      <label class="text-field text-field--compact">
        <span>Filtrer</span>
        <select aria-label="Filtrer l’historique" onChange={(event) => setFilter(event.currentTarget.value as Filter)} value={filter}>
          <option value="all">Toutes les opérations</option>
          <option value="transcription">Transcriptions</option>
          <option value="summary">Résumés</option>
          <option value="speech">Synthèses vocales</option>
        </select>
      </label>
      {error ? <p class="field__error" role="alert">{error}</p> : null}
      {filtered.length === 0 ? <p class="empty-guidance">Aucun résultat local pour ce filtre.</p> : (
        <div class="history-list">
          {filtered.map((entry) => (
            <article class={`history-row${entry.id === selectedId ? " history-row--selected" : ""}`} key={entry.id}>
              <div>
                <span class="history-row__kind">{entry.kind}</span>
                <strong>{entry.title}</strong>
                <time>{new Date(entry.createdAt).toLocaleString("fr-FR")}</time>
              </div>
              <button aria-label={`Ouvrir ${entry.title}`} class="history-row__open" onClick={() => setSelectedId(entry.id)} type="button">Ouvrir</button>
            </article>
          ))}
        </div>
      )}
      {selected ? <HistoryDetail entry={selected} onDelete={remove} /> : null}
      <HistorySettings entries={entries} onCleared={() => { setEntries([]); setSelectedId(null); }} repository={history} />
    </div>
  );
}
