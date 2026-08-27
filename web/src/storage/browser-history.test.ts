// @vitest-environment node
import "fake-indexeddb/auto";

import { beforeEach, describe, expect, it } from "vitest";

import { BrowserHistoryRepository } from "./browser-history";

async function deleteDatabase(): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const request = indexedDB.deleteDatabase("resume-transcription");
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error("Database deletion was blocked"));
  });
}

function audioBlob(contents: string): Promise<Blob> {
  return new Response(contents, { headers: { "content-type": "audio/mpeg" } }).blob();
}

beforeEach(deleteDatabase);

describe("BrowserHistoryRepository", () => {
  it("stores result text without ever accepting source audio", async () => {
    const repository = new BrowserHistoryRepository();
    const saved = await repository.add({
      kind: "transcription",
      title: "réunion.wav",
      parameters: { diarize: true },
      resultText: "Bonjour",
      metadata: { duration: 3.2 },
    });

    expect(saved.audio).toBeUndefined();
    expect((await repository.get(saved.id))?.resultText).toBe("Bonjour");
  });

  it("keeps generated speech only after an explicit request", async () => {
    const repository = new BrowserHistoryRepository();
    const saved = await repository.add({
      kind: "speech",
      title: "Bonjour",
      parameters: { voice: "Ryan" },
      resultText: "Bonjour",
      metadata: {},
    });

    const result = await repository.keepAudio(
      saved.id,
      await audioBlob("audio"),
    );

    expect(result.evictedIds).toEqual([]);
    expect((await repository.get(saved.id))?.audio?.size).toBe(5);
  });

  it("proposes old audio eviction and changes nothing before confirmation", async () => {
    const repository = new BrowserHistoryRepository();
    await repository.setLimits({ maxEntries: 10, maxAudioBytes: 6 });
    const oldest = await repository.add({
      kind: "speech",
      title: "ancien",
      parameters: {},
      resultText: "ancien",
      metadata: {},
    });
    await repository.keepAudio(oldest.id, await audioBlob("1234"));
    const newest = await repository.add({
      kind: "speech",
      title: "nouveau",
      parameters: {},
      resultText: "nouveau",
      metadata: {},
    });

    const proposal = await repository.keepAudio(newest.id, await audioBlob("5678"));

    expect(proposal.evictedIds).toEqual([oldest.id]);
    expect((await repository.get(oldest.id))?.audio?.size).toBe(4);
    expect((await repository.get(newest.id))?.audio).toBeUndefined();

    await repository.confirmAudioEviction(
      newest.id,
      await audioBlob("5678"),
      proposal.evictedIds,
    );

    expect((await repository.get(oldest.id))?.audio).toBeUndefined();
    expect((await repository.get(newest.id))?.audio?.size).toBe(4);
  });

  it("prunes the oldest entries and can clear all browser history", async () => {
    const repository = new BrowserHistoryRepository();
    await repository.setLimits({ maxEntries: 2, maxAudioBytes: 100 });
    const first = await repository.add({
      kind: "summary",
      title: "premier",
      parameters: {},
      resultText: "1",
      metadata: {},
    });
    await repository.add({
      kind: "summary",
      title: "deuxième",
      parameters: {},
      resultText: "2",
      metadata: {},
    });
    await repository.add({
      kind: "summary",
      title: "troisième",
      parameters: {},
      resultText: "3",
      metadata: {},
    });

    expect(await repository.get(first.id)).toBeUndefined();
    expect((await repository.list()).map((entry) => entry.title)).toEqual([
      "troisième",
      "deuxième",
    ]);

    await repository.clear();
    expect(await repository.list()).toEqual([]);
  });
});
