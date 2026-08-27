import type {
  HistoryDraft,
  HistoryEntry,
  HistoryLimits,
  HistoryRepository,
} from "./history";

const DATABASE_NAME = "resume-transcription";
const DATABASE_VERSION = 1;
const ENTRIES_STORE = "entries";
const SETTINGS_STORE = "settings";
const LIMITS_KEY = "limits";

const DEFAULT_LIMITS: HistoryLimits = {
  maxEntries: 100,
  maxAudioBytes: 250 * 1024 * 1024,
};

interface SettingRecord {
  key: string;
  value: HistoryLimits;
}

let lastCreatedAt = 0;

export class HistoryQuotaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HistoryQuotaError";
  }
}

export class BrowserHistoryRepository implements HistoryRepository {
  async add(draft: HistoryDraft): Promise<HistoryEntry> {
    const entry: HistoryEntry = {
      ...draft,
      id: globalThis.crypto.randomUUID(),
      createdAt: nextCreatedAt(),
    };
    const db = await openDatabase();
    try {
      const transaction = db.transaction([ENTRIES_STORE, SETTINGS_STORE], "readwrite");
      const entriesStore = transaction.objectStore(ENTRIES_STORE);
      const settingsStore = transaction.objectStore(SETTINGS_STORE);
      const entries = await requestResult(entriesStore.getAll() as IDBRequest<HistoryEntry[]>);
      const limits = await readLimits(settingsStore);
      entriesStore.put(entry);

      const excess = entries.length + 1 - limits.maxEntries;
      if (excess > 0) {
        for (const oldEntry of sortOldestFirst(entries).slice(0, excess)) {
          entriesStore.delete(oldEntry.id);
        }
      }
      await transactionDone(transaction);
      return entry;
    } finally {
      db.close();
    }
  }

  async list(): Promise<HistoryEntry[]> {
    const entries = await this.allEntries();
    return sortNewestFirst(entries);
  }

  async get(id: string): Promise<HistoryEntry | undefined> {
    const db = await openDatabase();
    try {
      const transaction = db.transaction(ENTRIES_STORE, "readonly");
      const entry = await requestResult(
        transaction.objectStore(ENTRIES_STORE).get(id) as IDBRequest<HistoryEntry | undefined>,
      );
      await transactionDone(transaction);
      return entry;
    } finally {
      db.close();
    }
  }

  async remove(id: string): Promise<void> {
    const db = await openDatabase();
    try {
      const transaction = db.transaction(ENTRIES_STORE, "readwrite");
      transaction.objectStore(ENTRIES_STORE).delete(id);
      await transactionDone(transaction);
    } finally {
      db.close();
    }
  }

  async keepAudio(id: string, audio: Blob): Promise<{ evictedIds: string[] }> {
    const [entries, limits] = await Promise.all([this.allEntries(), this.getLimits()]);
    const target = entries.find((entry) => entry.id === id);
    if (!target) throw new Error("Entrée d’historique introuvable.");

    const others = entries.filter((entry) => entry.id !== id);
    let requiredBytes = audio.size + audioBytes(others) - limits.maxAudioBytes;
    if (requiredBytes <= 0) {
      await this.writeAudio(id, audio);
      return { evictedIds: [] };
    }

    const evictedIds: string[] = [];
    for (const entry of sortOldestFirst(others)) {
      if (!entry.audio) continue;
      evictedIds.push(entry.id);
      requiredBytes -= entry.audio.size;
      if (requiredBytes <= 0) break;
    }

    if (requiredBytes > 0) {
      throw new HistoryQuotaError("Le fichier audio dépasse la limite de stockage locale.");
    }
    return { evictedIds };
  }

  async confirmAudioEviction(
    id: string,
    audio: Blob,
    evictedIds: string[],
  ): Promise<void> {
    const db = await openDatabase();
    try {
      const transaction = db.transaction([ENTRIES_STORE, SETTINGS_STORE], "readwrite");
      const entriesStore = transaction.objectStore(ENTRIES_STORE);
      const settingsStore = transaction.objectStore(SETTINGS_STORE);
      const entries = await requestResult(entriesStore.getAll() as IDBRequest<HistoryEntry[]>);
      const limits = await readLimits(settingsStore);
      const target = entries.find((entry) => entry.id === id);
      if (!target) throw new Error("Entrée d’historique introuvable.");

      const allowedEvictions = new Set(evictedIds.filter((entryId) => entryId !== id));
      const updated = entries.map((entry) => {
        if (!allowedEvictions.has(entry.id) || !entry.audio) return entry;
        const withoutAudio = { ...entry };
        delete withoutAudio.audio;
        return withoutAudio;
      });
      const retainedOthers = updated.filter((entry) => entry.id !== id);
      if (audio.size + audioBytes(retainedOthers) > limits.maxAudioBytes) {
        throw new HistoryQuotaError("Les suppressions confirmées ne libèrent pas assez d’espace.");
      }

      for (const entry of updated) {
        if (allowedEvictions.has(entry.id)) entriesStore.put(entry);
      }
      entriesStore.put({ ...target, audio });
      await transactionDone(transaction);
    } finally {
      db.close();
    }
  }

  async clear(): Promise<void> {
    const db = await openDatabase();
    try {
      const transaction = db.transaction(ENTRIES_STORE, "readwrite");
      transaction.objectStore(ENTRIES_STORE).clear();
      await transactionDone(transaction);
    } finally {
      db.close();
    }
  }

  async getLimits(): Promise<HistoryLimits> {
    const db = await openDatabase();
    try {
      const transaction = db.transaction(SETTINGS_STORE, "readonly");
      const limits = await readLimits(transaction.objectStore(SETTINGS_STORE));
      await transactionDone(transaction);
      return limits;
    } finally {
      db.close();
    }
  }

  async setLimits(limits: HistoryLimits): Promise<void> {
    validateLimits(limits);
    const db = await openDatabase();
    try {
      const transaction = db.transaction(SETTINGS_STORE, "readwrite");
      transaction.objectStore(SETTINGS_STORE).put({ key: LIMITS_KEY, value: limits });
      await transactionDone(transaction);
    } finally {
      db.close();
    }
  }

  private async allEntries(): Promise<HistoryEntry[]> {
    const db = await openDatabase();
    try {
      const transaction = db.transaction(ENTRIES_STORE, "readonly");
      const entries = await requestResult(
        transaction.objectStore(ENTRIES_STORE).getAll() as IDBRequest<HistoryEntry[]>,
      );
      await transactionDone(transaction);
      return entries;
    } finally {
      db.close();
    }
  }

  private async writeAudio(id: string, audio: Blob): Promise<void> {
    const db = await openDatabase();
    try {
      const transaction = db.transaction(ENTRIES_STORE, "readwrite");
      const store = transaction.objectStore(ENTRIES_STORE);
      const entry = await requestResult(store.get(id) as IDBRequest<HistoryEntry | undefined>);
      if (!entry) throw new Error("Entrée d’historique introuvable.");
      store.put({ ...entry, audio });
      await transactionDone(transaction);
    } finally {
      db.close();
    }
  }
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      const entries = db.createObjectStore(ENTRIES_STORE, { keyPath: "id" });
      entries.createIndex("createdAt", "createdAt");
      db.createObjectStore(SETTINGS_STORE, { keyPath: "key" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error ?? new Error("Transaction annulée."));
  });
}

async function readLimits(store: IDBObjectStore): Promise<HistoryLimits> {
  const record = await requestResult(
    store.get(LIMITS_KEY) as IDBRequest<SettingRecord | undefined>,
  );
  return record?.value ?? DEFAULT_LIMITS;
}

function audioBytes(entries: HistoryEntry[]): number {
  return entries.reduce((total, entry) => total + (entry.audio?.size ?? 0), 0);
}

function sortOldestFirst(entries: HistoryEntry[]): HistoryEntry[] {
  return [...entries].sort((left, right) => left.createdAt.localeCompare(right.createdAt));
}

function sortNewestFirst(entries: HistoryEntry[]): HistoryEntry[] {
  return [...entries].sort((left, right) => right.createdAt.localeCompare(left.createdAt));
}

function nextCreatedAt(): string {
  const now = Date.now();
  lastCreatedAt = Math.max(now, lastCreatedAt + 1);
  return new Date(lastCreatedAt).toISOString();
}

function validateLimits(limits: HistoryLimits): void {
  if (!Number.isInteger(limits.maxEntries) || limits.maxEntries < 1) {
    throw new RangeError("maxEntries doit être un entier positif.");
  }
  if (!Number.isInteger(limits.maxAudioBytes) || limits.maxAudioBytes < 0) {
    throw new RangeError("maxAudioBytes doit être un entier positif ou nul.");
  }
}
