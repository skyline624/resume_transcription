import type { TranscriptionInput } from "./transcription-api";

export type TranscriptionOptionsValue = Omit<TranscriptionInput, "audio">;

export function TranscriptionOptions({
  value,
  onChange,
}: {
  value: TranscriptionOptionsValue;
  onChange(value: TranscriptionOptionsValue): void;
}) {
  const patch = (next: Partial<TranscriptionOptionsValue>) => onChange({ ...value, ...next });
  return (
    <details class="options-panel">
      <summary>Options de transcription</summary>
      <div class="options-grid">
        <label>
          <span>Langue</span>
          <select
            aria-label="Langue"
            onChange={(event) => patch({ language: event.currentTarget.value as "" | "fr" | "en" })}
            value={value.language}
          >
            <option value="">Détection automatique</option>
            <option value="fr">Français</option>
            <option value="en">Anglais</option>
          </select>
        </label>
        <label>
          <span>Canaux</span>
          <select
            aria-label="Canaux"
            onChange={(event) => patch({ channels: event.currentTarget.value as TranscriptionOptionsValue["channels"] })}
            value={value.channels}
          >
            <option value="mix">Mélanger</option>
            <option value="left">Gauche</option>
            <option value="right">Droite</option>
            <option value="separate">Séparer</option>
          </select>
        </label>
        <label class="check-field">
          <input
            checked={value.diarize}
            onChange={(event) => patch({ diarize: event.currentTarget.checked })}
            type="checkbox"
          />
          <span>Séparer les locuteurs</span>
        </label>
        <label class="check-field">
          <input
            checked={value.wordTimestamps}
            onChange={(event) => patch({ wordTimestamps: event.currentTarget.checked })}
            type="checkbox"
          />
          <span>Horodatage des mots</span>
        </label>
        <label>
          <span>Nombre exact de locuteurs</span>
          <input
            aria-label="Nombre exact de locuteurs"
            min="1"
            onInput={(event) => patch({
              numSpeakers: event.currentTarget.value,
              minSpeakers: "",
              maxSpeakers: "",
            })}
            type="number"
            value={value.numSpeakers}
          />
        </label>
        <label>
          <span>Minimum de locuteurs</span>
          <input
            aria-label="Minimum de locuteurs"
            min="1"
            onInput={(event) => patch({ minSpeakers: event.currentTarget.value, numSpeakers: "" })}
            type="number"
            value={value.minSpeakers}
          />
        </label>
        <label>
          <span>Maximum de locuteurs</span>
          <input
            aria-label="Maximum de locuteurs"
            min="1"
            onInput={(event) => patch({ maxSpeakers: event.currentTarget.value, numSpeakers: "" })}
            type="number"
            value={value.maxSpeakers}
          />
        </label>
      </div>
    </details>
  );
}
