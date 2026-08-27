import { useEffect, useMemo, useState } from "preact/hooks";

import type { AudioFormat, AudioResult, ListResponse, TtsMode, Voice } from "../../api/contracts";
import { useServices } from "../../app/services";
import { Button } from "../../ui/Button";
import { OperationStatus } from "../../ui/OperationStatus";
import { OneShotClone, audioFormatOptions } from "./OneShotClone";
import { createSpeech } from "./speech-api";
import { SpeechResult } from "./SpeechResult";

export function SpeechPage() {
  const { http, history } = useServices();
  const [mode, setMode] = useState<TtsMode>("qwen3-tts-custom-voice");
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voice, setVoice] = useState("");
  const [input, setInput] = useState("");
  const [instructions, setInstructions] = useState("");
  const [language, setLanguage] = useState("fr");
  const [responseFormat, setResponseFormat] = useState<AudioFormat>("mp3");
  const [speed, setSpeed] = useState(1);
  const [result, setResult] = useState<AudioResult | null>(null);
  const [historyId, setHistoryId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [startedAt, setStartedAt] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const filteredVoices = useMemo(
    () => voices.filter((item) => item.kind === (mode === "qwen3-tts-clone" ? "clone" : "builtin")),
    [mode, voices],
  );

  useEffect(() => {
    let disposed = false;
    void http.getJson<ListResponse<Voice>>("/v1/voices").then((response) => {
      if (!disposed) {
        setVoices(response.data);
        setVoice(response.data.find((item) => item.kind === "builtin")?.id ?? "");
      }
    }).catch((reason) => {
      if (!disposed) setError(reason instanceof Error ? reason.message : "Les voix sont indisponibles.");
    });
    return () => {
      disposed = true;
    };
  }, [http]);

  useEffect(() => {
    setVoice(filteredVoices[0]?.id ?? "");
  }, [filteredVoices]);

  const voiceDesign = mode === "qwen3-tts-voice-design";
  const canSubmit =
    input.trim().length > 0 && input.length <= 4096 && !pending &&
    (voiceDesign ? instructions.trim().length > 0 : voice.length > 0);

  const submit = async (event: Event) => {
    event.preventDefault();
    if (!canSubmit) return;
    setPending(true);
    setStartedAt(Date.now());
    setError(null);
    try {
      const audio = await createSpeech(http, {
        model: mode,
        input,
        ...(voiceDesign ? { instructions } : { voice }),
        response_format: responseFormat,
        speed,
        language,
      });
      const entry = await history.add({
        kind: "speech",
        title: input.slice(0, 60),
        parameters: {
          mode,
          voice: voiceDesign ? null : voice,
          language,
          responseFormat,
          speed,
        },
        resultText: input,
        metadata: {},
      });
      setResult(audio);
      setHistoryId(entry.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "La synthèse vocale a échoué.");
    } finally {
      setPending(false);
    }
  };

  return (
    <div class="workflow-stack">
      <form class="workflow-form" onSubmit={(event) => void submit(event)}>
        <label class="text-field text-field--compact">
          <span>Mode vocal</span>
          <select aria-label="Mode vocal" onChange={(event) => setMode(event.currentTarget.value as TtsMode)} value={mode}>
            <option value="qwen3-tts-custom-voice">Voix prédéfinie</option>
            <option value="qwen3-tts-clone">Clone enregistré</option>
            <option value="qwen3-tts-voice-design">VoiceDesign</option>
          </select>
        </label>
        {!voiceDesign ? (
          <label class="text-field text-field--compact">
            <span>Voix</span>
            <select aria-label="Voix" onChange={(event) => setVoice(event.currentTarget.value)} value={voice}>
              {filteredVoices.length === 0 ? <option value="">Aucune voix disponible</option> : null}
              {filteredVoices.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select>
          </label>
        ) : (
          <label class="text-field">
            <span>Description de la voix</span>
            <textarea aria-label="Description de la voix" onInput={(event) => setInstructions(event.currentTarget.value)} rows={4} value={instructions} />
          </label>
        )}
        <label class="text-field">
          <span>Texte à prononcer</span>
          <textarea aria-label="Texte à prononcer" maxLength={4096} onInput={(event) => setInput(event.currentTarget.value)} rows={7} value={input} />
          <small class="character-count">{input.length} / 4096</small>
        </label>
        <div class="options-grid options-grid--always">
          <label><span>Langue</span><input aria-label="Langue de synthèse" onInput={(event) => setLanguage(event.currentTarget.value)} value={language} /></label>
          <label><span>Vitesse</span><input aria-label="Vitesse" max="4" min="0.25" onInput={(event) => setSpeed(Number(event.currentTarget.value))} step="0.05" type="number" value={speed} /></label>
          <label><span>Format</span><select aria-label="Format audio" onChange={(event) => setResponseFormat(event.currentTarget.value as AudioFormat)} value={responseFormat}>{audioFormatOptions()}</select></label>
        </div>
        <OperationStatus active={pending} label="Synthèse en cours" startedAt={startedAt} />
        {error ? <p class="field__error" role="alert">{error}</p> : null}
        <Button disabled={!canSubmit} type="submit" variant="primary">Créer l’audio</Button>
      </form>
      {result && historyId ? <SpeechResult audio={result} history={history} historyId={historyId} /> : null}
      <OneShotClone />
    </div>
  );
}
