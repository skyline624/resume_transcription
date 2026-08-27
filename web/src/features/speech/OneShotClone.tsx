import { useState } from "preact/hooks";

import type { AudioFormat, AudioResult } from "../../api/contracts";
import { useServices } from "../../app/services";
import { AudioSourcePicker } from "../../media/AudioSourcePicker";
import type { AudioSelection } from "../../media/recorder";
import { Button } from "../../ui/Button";
import { OperationStatus } from "../../ui/OperationStatus";
import { SpeechResult } from "./SpeechResult";
import { cloneOnce } from "./speech-api";

export function OneShotClone() {
  const { http, history, recorderFactory } = useServices();
  const [reference, setReference] = useState<AudioSelection | null>(null);
  const [referenceValid, setReferenceValid] = useState(false);
  const [input, setInput] = useState("");
  const [transcript, setTranscript] = useState("");
  const [consent, setConsent] = useState(false);
  const [language, setLanguage] = useState("fr");
  const [responseFormat, setResponseFormat] = useState<AudioFormat>("mp3");
  const [speed, setSpeed] = useState(1);
  const [result, setResult] = useState<AudioResult | null>(null);
  const [historyId, setHistoryId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [startedAt, setStartedAt] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const canSubmit = referenceValid && consent && input.trim().length > 0 && !pending;

  const submit = async (event: Event) => {
    event.preventDefault();
    if (!reference || !canSubmit) return;
    setPending(true);
    setStartedAt(Date.now());
    setError(null);
    try {
      const audio = await cloneOnce(http, {
        reference,
        input,
        consent,
        transcript,
        language,
        responseFormat,
        speed,
      });
      const entry = await history.add({
        kind: "speech",
        title: input.slice(0, 60),
        parameters: { mode: "clone-once", language, responseFormat, speed },
        resultText: input,
        metadata: {},
      });
      setResult(audio);
      setHistoryId(entry.id);
      setConsent(false);
      setReference(null);
      setReferenceValid(false);
      setTranscript("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Le clonage ponctuel a échoué.");
    } finally {
      setPending(false);
    }
  };

  return (
    <details class="one-shot-clone">
      <summary>Clone ponctuel</summary>
      <form class="workflow-form" onSubmit={(event) => void submit(event)}>
        <p class="field__hint">La référence sert uniquement à cette génération et n’est pas ajoutée à l’historique.</p>
        <AudioSourcePicker
          onChange={setReference}
          onValidityChange={setReferenceValid}
          recorderFactory={recorderFactory}
          referenceMode
          value={reference}
        />
        <label class="text-field">
          <span>Texte à prononcer avec le clone ponctuel</span>
          <textarea maxLength={4096} onInput={(event) => setInput(event.currentTarget.value)} rows={4} value={input} />
        </label>
        <label class="text-field">
          <span>Transcription de la référence (facultative)</span>
          <textarea onInput={(event) => setTranscript(event.currentTarget.value)} rows={3} value={transcript} />
        </label>
        <div class="options-grid options-grid--always">
          <label><span>Langue</span><input onInput={(event) => setLanguage(event.currentTarget.value)} value={language} /></label>
          <label><span>Vitesse</span><input max="4" min="0.25" onInput={(event) => setSpeed(Number(event.currentTarget.value))} step="0.05" type="number" value={speed} /></label>
          <label><span>Format</span><select onChange={(event) => setResponseFormat(event.currentTarget.value as AudioFormat)} value={responseFormat}>{audioFormatOptions()}</select></label>
        </div>
        <label class="consent-field">
          <input checked={consent} onChange={(event) => setConsent(event.currentTarget.checked)} type="checkbox" />
          <span>Je confirme avoir le droit d’utiliser cette voix pour cette synthèse.</span>
        </label>
        <OperationStatus active={pending} label="Clonage et synthèse en cours" startedAt={startedAt} />
        {error ? <p class="field__error" role="alert">{error}</p> : null}
        <Button disabled={!canSubmit} type="submit" variant="primary">Cloner et synthétiser</Button>
      </form>
      {result && historyId ? <SpeechResult audio={result} history={history} historyId={historyId} /> : null}
    </details>
  );
}

export function audioFormatOptions() {
  return (["mp3", "wav", "flac", "opus", "aac", "pcm"] as const).map((format) => (
    <option key={format} value={format}>{format.toUpperCase()}</option>
  ));
}
