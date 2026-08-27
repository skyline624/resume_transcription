import type { TranscriptionResult as Result } from "../../api/contracts";
import { routeHref } from "../../app/routes";
import { Button } from "../../ui/Button";
import {
  downloadText,
  toDialogue,
  toJson,
  toSrt,
  toText,
  toVtt,
} from "../../utils/exports/transcription";

export function TranscriptionResult({ result, historyId }: { result: Result; historyId: string | null }) {
  const baseName = "transcription";
  return (
    <section class="result-panel" aria-label="Résultat de la transcription">
      <div class="result-panel__heading">
        <div>
          <p class="eyebrow">Résultat</p>
          <h2>Transcription terminée</h2>
        </div>
        <span class="data-badge">{formatSeconds(result.duration)}</span>
      </div>
      {result.turns.length > 0 ? (
        <>
          <ol
            aria-label="Transcription par locuteur"
            class="turn-list turn-list--primary"
          >
            {result.turns.map((turn, index) => (
              <li key={`${turn.start}-${index}`}>
                <span>{turn.speaker ?? "Voix"}</span>
                <time>{formatSeconds(turn.start)}</time>
                <p>{turn.text}</p>
              </li>
            ))}
          </ol>
          <details class="continuous-transcript">
            <summary>Texte continu</summary>
            <p class="transcript-text">{result.text}</p>
          </details>
        </>
      ) : (
        <p class="transcript-text">{result.text}</p>
      )}
      <div class="result-actions" aria-label="Exports">
        <Button onClick={() => void navigator.clipboard?.writeText(toDialogue(result))}>Copier</Button>
        <Button onClick={() => downloadText(`${baseName}.txt`, toText(result), "text/plain;charset=utf-8")}>Texte</Button>
        <Button onClick={() => downloadText(`${baseName}-dialogue.txt`, toDialogue(result), "text/plain;charset=utf-8")}>Dialogue</Button>
        <Button onClick={() => downloadText(`${baseName}.srt`, toSrt(result), "application/x-subrip")}>SRT</Button>
        <Button onClick={() => downloadText(`${baseName}.vtt`, toVtt(result), "text/vtt")}>VTT</Button>
        <Button onClick={() => downloadText(`${baseName}.json`, toJson(result), "application/json")}>JSON</Button>
      </div>
      {historyId ? (
        <a class="button button--primary action-link" href={`${routeHref("summarize")}?history=${encodeURIComponent(historyId)}`}>
          Résumer cette transcription
        </a>
      ) : null}
    </section>
  );
}

function formatSeconds(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}:${remainder.toString().padStart(2, "0")}`;
}
