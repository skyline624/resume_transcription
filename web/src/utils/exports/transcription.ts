import type { TranscriptionResult } from "../../api/contracts";

export function toText(result: TranscriptionResult): string {
  return `${result.text.trim()}\n`;
}

export function toDialogue(result: TranscriptionResult): string {
  if (result.turns.length === 0) return toText(result);
  return result.turns
    .map((turn) => `${turn.speaker ? `${turn.speaker} : ` : ""}${turn.text.trim()}`)
    .join("\n") + "\n";
}

export function toSrt(result: TranscriptionResult): string {
  return result.turns
    .map(
      (turn, index) =>
        `${index + 1}\n${formatTimestamp(turn.start, ",")} --> ${formatTimestamp(turn.end, ",")}\n${cueText(turn.speaker, turn.text)}\n`,
    )
    .join("");
}

export function toVtt(result: TranscriptionResult): string {
  const cues = result.turns
    .map(
      (turn) =>
        `${formatTimestamp(turn.start, ".")} --> ${formatTimestamp(turn.end, ".")}\n${cueText(turn.speaker, turn.text)}\n`,
    )
    .join("\n");
  return `WEBVTT\n\n${cues}`;
}

export function toJson(result: TranscriptionResult): string {
  return `${JSON.stringify(result, null, 2)}\n`;
}

export function downloadText(filename: string, contents: string, type: string): void {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function cueText(speaker: string | null, text: string): string {
  const cleaned = text.replace(/[\r\n]+/g, " ").trim();
  return speaker ? `${speaker} : ${cleaned}` : cleaned;
}

function formatTimestamp(seconds: number, separator: "," | "."): string {
  const milliseconds = Math.max(0, Math.round(seconds * 1_000));
  const hours = Math.floor(milliseconds / 3_600_000);
  const minutes = Math.floor((milliseconds % 3_600_000) / 60_000);
  const remainingSeconds = Math.floor((milliseconds % 60_000) / 1_000);
  const millis = milliseconds % 1_000;
  return `${pad(hours)}:${pad(minutes)}:${pad(remainingSeconds)}${separator}${millis.toString().padStart(3, "0")}`;
}

function pad(value: number): string {
  return value.toString().padStart(2, "0");
}
