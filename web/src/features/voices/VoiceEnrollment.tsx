import { useState } from "preact/hooks";

import { useServices } from "../../app/services";
import { AudioSourcePicker } from "../../media/AudioSourcePicker";
import type { AudioSelection } from "../../media/recorder";
import { Button } from "../../ui/Button";
import { createVoice } from "./voices-api";

export function VoiceEnrollment({ onCreated }: { onCreated(): void }) {
  const { http, recorderFactory } = useServices();
  const [reference, setReference] = useState<AudioSelection | null>(null);
  const [referenceValid, setReferenceValid] = useState(false);
  const [name, setName] = useState("");
  const [language, setLanguage] = useState("fr");
  const [transcript, setTranscript] = useState("");
  const [consent, setConsent] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canSubmit = referenceValid && name.trim().length > 0 && consent && !pending;

  const submit = async (event: Event) => {
    event.preventDefault();
    if (!reference || !canSubmit) return;
    setPending(true);
    setError(null);
    try {
      await createVoice(http, { reference, name, language, transcript, consent });
      setReference(null);
      setReferenceValid(false);
      setName("");
      setLanguage("fr");
      setTranscript("");
      setConsent(false);
      onCreated();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "La voix n’a pas pu être enregistrée.");
    } finally {
      setPending(false);
    }
  };

  return (
    <details class="voice-enrollment">
      <summary>Ajouter une voix personnelle</summary>
      <form class="workflow-form" onSubmit={(event) => void submit(event)}>
        <AudioSourcePicker
          onChange={setReference}
          onValidityChange={setReferenceValid}
          recorderFactory={recorderFactory}
          referenceMode
          value={reference}
        />
        <div class="options-grid options-grid--always">
          <label>
            <span>Nom de la voix</span>
            <input aria-label="Nom de la voix" maxLength={80} onInput={(event) => setName(event.currentTarget.value)} value={name} />
          </label>
          <label>
            <span>Langue</span>
            <input aria-label="Langue de la voix" onInput={(event) => setLanguage(event.currentTarget.value)} value={language} />
          </label>
        </div>
        <label class="text-field">
          <span>Transcription de la référence (facultative)</span>
          <textarea aria-label="Transcription de la référence" onInput={(event) => setTranscript(event.currentTarget.value)} rows={3} value={transcript} />
          <small class="field__hint">Si ce champ reste vide, Parakeet transcrira la référence.</small>
        </label>
        <p class="consent-copy">Vous confirmez avoir le droit d’utiliser cette voix. La référence sera stockée localement dans le volume Docker.</p>
        <label class="consent-field">
          <input checked={consent} onChange={(event) => setConsent(event.currentTarget.checked)} type="checkbox" />
          <span>Je confirme avoir le droit d’utiliser cette voix.</span>
        </label>
        {error ? <p class="field__error" role="alert">{error}</p> : null}
        <Button disabled={!canSubmit} type="submit" variant="primary">{pending ? "Enregistrement…" : "Enregistrer la voix"}</Button>
      </form>
    </details>
  );
}
