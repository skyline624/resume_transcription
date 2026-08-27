import { useEffect, useState } from "preact/hooks";

import type { SummaryResult } from "../../api/contracts";
import { useServices } from "../../app/services";
import { useHealth } from "../health/use-health";
import { AudioSourcePicker } from "../../media/AudioSourcePicker";
import type { AudioSelection } from "../../media/recorder";
import { Button } from "../../ui/Button";
import { OperationStatus } from "../../ui/OperationStatus";
import { downloadText } from "../../utils/exports/transcription";
import { summarize, type SummaryFormat } from "./summary-api";

type Source = "audio" | "text" | "history";

export function SummaryPage() {
  const { http, history, recorderFactory } = useServices();
  const { health } = useHealth();
  const historyId = historyIdFromHash();
  const [source, setSource] = useState<Source>(historyId ? "history" : "text");
  const [transcript, setTranscript] = useState("");
  const [audio, setAudio] = useState<AudioSelection | null>(null);
  const [audioValid, setAudioValid] = useState(false);
  const [format, setFormat] = useState<SummaryFormat>("structure");
  const [language, setLanguage] = useState<"" | "fr" | "en">("");
  const [channels, setChannels] = useState<"mix" | "left" | "right" | "separate">("mix");
  const [diarize, setDiarize] = useState(false);
  const [result, setResult] = useState<SummaryResult | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [startedAt, setStartedAt] = useState(0);

  useEffect(() => {
    if (!historyId) return;
    let disposed = false;
    void history.get(historyId).then((entry) => {
      if (disposed) return;
      if (!entry || entry.kind !== "transcription") {
        setLoadError("Cette transcription n’existe plus dans l’historique local.");
        setTranscript("");
        return;
      }
      setTranscript(entry.resultText);
    });
    return () => {
      disposed = true;
    };
  }, [history, historyId]);

  const changeSource = (next: Source) => {
    setSource(next);
    setTranscript("");
    setAudio(null);
    setLoadError(null);
  };

  const canSubmit =
    health?.summary_enabled !== false &&
    !pending &&
    (source === "audio" ? audioValid : transcript.trim().length > 0);

  const submit = async (event: Event) => {
    event.preventDefault();
    if (!canSubmit) return;
    setPending(true);
    setStartedAt(Date.now());
    setError(null);
    try {
      const response = await summarize(
        http,
        source === "audio" && audio
          ? { source, audio, format, language, channels, diarize }
          : { source: source === "history" ? "history" : "text", transcript, format },
      );
      setResult(response);
      await history.add({
        kind: "summary",
        title: source === "audio" ? audio?.filename ?? "Résumé audio" : "Compte-rendu",
        parameters: { source, format, model: response.model },
        resultText: response.summary,
        metadata: { model: response.model, format: response.format },
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "La rédaction du résumé a échoué.");
    } finally {
      setPending(false);
    }
  };

  return (
    <div class="workflow-stack">
      <form class="workflow-form" onSubmit={(event) => void submit(event)}>
        <fieldset class="segmented-control">
          <legend>Source du résumé</legend>
          {(["audio", "text", "history"] as const).map((item) => (
            <label key={item}>
              <input
                checked={source === item}
                name="summary-source"
                onChange={() => changeSource(item)}
                type="radio"
              />
              {item === "audio" ? "Audio" : item === "text" ? "Texte" : "Historique"}
            </label>
          ))}
        </fieldset>

        {source === "audio" ? (
          <>
            <AudioSourcePicker
              onChange={setAudio}
              onValidityChange={setAudioValid}
              recorderFactory={recorderFactory}
              value={audio}
            />
            <div class="options-grid options-grid--always">
              <label>
                <span>Langue</span>
                <select aria-label="Langue" onChange={(event) => setLanguage(event.currentTarget.value as typeof language)} value={language}>
                  <option value="">Détection automatique</option>
                  <option value="fr">Français</option>
                  <option value="en">Anglais</option>
                </select>
              </label>
              <label>
                <span>Canaux</span>
                <select aria-label="Canaux" onChange={(event) => setChannels(event.currentTarget.value as typeof channels)} value={channels}>
                  <option value="mix">Mélanger</option>
                  <option value="left">Gauche</option>
                  <option value="right">Droite</option>
                  <option value="separate">Séparer</option>
                </select>
              </label>
              <label class="check-field">
                <input checked={diarize} onChange={(event) => setDiarize(event.currentTarget.checked)} type="checkbox" />
                <span>Séparer les locuteurs</span>
              </label>
            </div>
          </>
        ) : (
          <label class="text-field">
            <span>Transcription à résumer</span>
            <textarea
              aria-label="Transcription à résumer"
              onInput={(event) => setTranscript(event.currentTarget.value)}
              placeholder="Collez ici la transcription…"
              rows={10}
              value={transcript}
            />
          </label>
        )}

        <label class="text-field text-field--compact">
          <span>Format</span>
          <select aria-label="Format du résumé" onChange={(event) => setFormat(event.currentTarget.value as SummaryFormat)} value={format}>
            <option value="structure">Structuré</option>
            <option value="narratif">Narratif</option>
          </select>
        </label>
        {loadError ? <p class="field__error" role="alert">{loadError}</p> : null}
        {health?.summary_enabled === false ? (
          <p class="service-guidance">Le résumé est désactivé. Configurez <code>ENABLE_SUMMARY=true</code> puis redémarrez le conteneur.</p>
        ) : null}
        <OperationStatus active={pending} label="Rédaction en cours" startedAt={startedAt} />
        {error ? <p class="field__error" role="alert">{error}</p> : null}
        <Button disabled={!canSubmit} type="submit" variant="primary">Rédiger le résumé</Button>
      </form>

      {result ? (
        <section class="result-panel" aria-label="Résumé produit">
          <div class="result-panel__heading">
            <div><p class="eyebrow">{result.model}</p><h2>Résumé prêt</h2></div>
          </div>
          <div class="summary-text">{result.summary}</div>
          <div class="result-actions">
            <Button onClick={() => void navigator.clipboard?.writeText(result.summary)}>Copier</Button>
            <Button onClick={() => downloadText("compte-rendu.md", result.summary, "text/markdown;charset=utf-8")}>Télécharger .md</Button>
          </div>
        </section>
      ) : null}
    </div>
  );
}

function historyIdFromHash(): string | null {
  const query = window.location.hash.split("?", 2)[1];
  return query ? new URLSearchParams(query).get("history") : null;
}
