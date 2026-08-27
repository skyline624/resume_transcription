import { useCallback, useEffect, useRef, useState } from "preact/hooks";

import { BrowserRecorder, type RecordedAudio, type RecorderPort } from "./recorder";

export type RecorderUiState = "idle" | "requesting" | "recording" | "stopped" | "error";

export interface RecorderController {
  state: RecorderUiState;
  elapsedMs: number;
  error: string | null;
  stream: MediaStream | null;
  start(): Promise<void>;
  stop(): Promise<RecordedAudio | null>;
  cancel(): void;
}

export function useRecorder(factory: () => RecorderPort = () => new BrowserRecorder()): RecorderController {
  const recorderRef = useRef<RecorderPort | null>(null);
  const startedAtRef = useRef(0);
  const [state, setState] = useState<RecorderUiState>("idle");
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  useEffect(() => {
    if (state !== "recording") return;
    const timer = window.setInterval(() => {
      setElapsedMs(Date.now() - startedAtRef.current);
    }, 100);
    return () => window.clearInterval(timer);
  }, [state]);

  useEffect(
    () => () => {
      recorderRef.current?.cancel();
      recorderRef.current = null;
    },
    [],
  );

  const start = useCallback(async () => {
    const recorder = factory();
    recorderRef.current = recorder;
    setState("requesting");
    setError(null);
    setElapsedMs(0);
    try {
      await recorder.start();
      startedAtRef.current = Date.now();
      setStream(recorder.mediaStream());
      setState("recording");
    } catch (reason) {
      recorder.cancel();
      setStream(null);
      setError(errorMessage(reason));
      setState("error");
    }
  }, [factory]);

  const stop = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder) return null;
    try {
      const audio = await recorder.stop();
      setElapsedMs(audio.durationMs);
      setStream(null);
      setState("stopped");
      return audio;
    } catch (reason) {
      setStream(null);
      setError(errorMessage(reason));
      setState("error");
      return null;
    }
  }, []);

  const cancel = useCallback(() => {
    recorderRef.current?.cancel();
    recorderRef.current = null;
    setElapsedMs(0);
    setStream(null);
    setError(null);
    setState("idle");
  }, []);

  return { state, elapsedMs, error, stream, start, stop, cancel };
}

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : "Le microphone n’est pas disponible.";
}
