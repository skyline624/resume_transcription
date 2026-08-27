import { useEffect, useState } from "preact/hooks";

import { Button } from "../ui/Button";
import { type AudioSelection, BrowserRecorder, type RecorderPort } from "./recorder";
import { useRecorder } from "./use-recorder";
import { Waveform } from "./Waveform";

interface AudioSourcePickerProps {
  value: AudioSelection | null;
  onChange(value: AudioSelection | null): void;
  onValidityChange?(valid: boolean): void;
  referenceMode?: boolean;
  recorderFactory?: () => RecorderPort;
}

export function AudioSourcePicker({
  value,
  onChange,
  onValidityChange,
  referenceMode = false,
  recorderFactory = () => new BrowserRecorder(),
}: AudioSourcePickerProps) {
  const recorder = useRecorder(recorderFactory);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const validation = validateSelection(value, referenceMode);

  useEffect(() => {
    if (!value) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(value.blob);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [value]);

  useEffect(() => {
    onValidityChange?.(validation.valid);
  }, [onValidityChange, validation.valid]);

  const selectFile = (event: Event) => {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    onChange({ blob: file, filename: file.name, durationMs: 0, origin: "file" });
  };

  const stopRecording = async () => {
    const captured = await recorder.stop();
    if (captured) onChange({ ...captured, origin: "recording" });
  };

  const cancelRecording = () => {
    recorder.cancel();
    onChange(null);
  };

  const readDuration = (event: Event) => {
    if (!value) return;
    const duration = (event.currentTarget as HTMLAudioElement).duration;
    if (Number.isFinite(duration)) onChange({ ...value, durationMs: Math.round(duration * 1_000) });
  };

  return (
    <section class="audio-source" aria-label="Source audio">
      <label class="audio-source__file">
        <span>Choisir un fichier audio</span>
        <input accept="audio/*" aria-label="Fichier audio" onChange={selectFile} type="file" />
      </label>

      <div class="audio-source__divider" aria-hidden="true">
        ou
      </div>

      {recorder.state === "idle" || recorder.state === "stopped" || recorder.state === "error" ? (
        <Button onClick={() => void recorder.start()}>Enregistrer au micro</Button>
      ) : null}
      {recorder.state === "requesting" ? <p role="status">Autorisation du microphone…</p> : null}
      {recorder.state === "recording" ? (
        <div class="audio-source__recording">
          <Waveform stream={recorder.stream} />
          <p class="audio-source__duration" role="timer">
            {formatDuration(recorder.elapsedMs)}
          </p>
          <div class="audio-source__actions">
            <Button onClick={() => void stopRecording()} variant="primary">
              Arrêter l’enregistrement
            </Button>
            <Button onClick={cancelRecording}>Annuler l’enregistrement</Button>
          </div>
        </div>
      ) : null}
      {recorder.error ? <p class="field__error">{recorder.error}</p> : null}

      {value && previewUrl ? (
        <div class="audio-source__preview">
          <div>
            <strong>{value.filename}</strong>
            {(value.durationMs ?? 0) > 0 ? <span>{formatDuration(value.durationMs ?? 0)}</span> : null}
          </div>
          <audio controls onLoadedMetadata={readDuration} src={previewUrl} />
          <Button onClick={() => onChange(null)}>Retirer l’audio</Button>
        </div>
      ) : null}
      {!validation.valid && validation.message ? (
        <p class="field__error" role="alert">
          {validation.message}
        </p>
      ) : null}
    </section>
  );
}

export function validateSelection(
  value: AudioSelection | null,
  referenceMode: boolean,
): { valid: boolean; message?: string } {
  if (!value) return { valid: false };
  if (!referenceMode) return { valid: true };
  const durationMs = value.durationMs ?? 0;
  if (durationMs <= 0) {
    return { valid: false, message: "La durée de la référence est en cours de lecture." };
  }
  if (durationMs < 3_000) {
    return { valid: false, message: "La référence doit durer au moins 3 secondes." };
  }
  if (durationMs > 30_000) {
    return { valid: false, message: "La référence ne doit pas dépasser 30 secondes." };
  }
  return { valid: true };
}

function formatDuration(durationMs: number): string {
  const totalSeconds = Math.floor(durationMs / 1_000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}
