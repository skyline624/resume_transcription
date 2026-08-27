import { useState } from "preact/hooks";

import type { TranscriptionResult as Result } from "../../api/contracts";
import { useServices } from "../../app/services";
import { AudioSourcePicker } from "../../media/AudioSourcePicker";
import type { AudioSelection } from "../../media/recorder";
import type { JsonValue } from "../../storage/history";
import { Button } from "../../ui/Button";
import { OperationStatus } from "../../ui/OperationStatus";
import { TranscriptionOptions, type TranscriptionOptionsValue } from "./TranscriptionOptions";
import { TranscriptionResult } from "./TranscriptionResult";
import { transcribe } from "./transcription-api";

const DEFAULT_OPTIONS: TranscriptionOptionsValue = {
  language: "",
  diarize: false,
  channels: "mix",
  wordTimestamps: true,
  numSpeakers: "",
  minSpeakers: "",
  maxSpeakers: "",
};

export function TranscriptionPage() {
  const { http, history, recorderFactory } = useServices();
  const [audio, setAudio] = useState<AudioSelection | null>(null);
  const [audioValid, setAudioValid] = useState(false);
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  const [result, setResult] = useState<Result | null>(null);
  const [historyId, setHistoryId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [startedAt, setStartedAt] = useState(0);

  const submit = async (event: Event) => {
    event.preventDefault();
    if (!audio || !audioValid || pending) return;
    setPending(true);
    setStartedAt(Date.now());
    setError(null);
    try {
      const response = await transcribe(http, { audio, ...options });
      setResult(response);
      const entry = await history.add({
        kind: "transcription",
        title: audio.filename,
        parameters: {
          language: options.language || "auto",
          diarize: options.diarize,
          channels: options.channels,
          wordTimestamps: options.wordTimestamps,
        },
        resultText: response.text,
        metadata: {
          duration: response.duration,
          speakers: response.speakers,
          response: JSON.parse(JSON.stringify(response)) as JsonValue,
        },
      });
      setHistoryId(entry.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "La transcription a échoué.");
    } finally {
      setPending(false);
    }
  };

  return (
    <div class="workflow-stack">
      <form class="workflow-form" onSubmit={(event) => void submit(event)}>
        <AudioSourcePicker
          onChange={setAudio}
          onValidityChange={setAudioValid}
          recorderFactory={recorderFactory}
          value={audio}
        />
        <TranscriptionOptions onChange={setOptions} value={options} />
        <OperationStatus active={pending} label="Transcription en cours" startedAt={startedAt} />
        {error ? <p class="field__error" role="alert">{error}</p> : null}
        <Button disabled={!audioValid || pending} type="submit" variant="primary">
          Transcrire
        </Button>
      </form>
      {result ? <TranscriptionResult historyId={historyId} result={result} /> : null}
    </div>
  );
}
