# Local Web Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Livrer dans l'image Docker unique une interface Preact locale couvrant toute l'API audio, le microphone et un historique IndexedDB privé.

**Architecture:** Vite construit une SPA Preact à navigation par ancres ; FastAPI sert uniquement le build statique et conserve tous les contrats API. Les fonctions métier du navigateur sont isolées en client HTTP, dépôt IndexedDB et service de capture afin que les écrans restent testables sans GPU.

**Tech Stack:** Preact 10.29.8, TypeScript 7.0.2, Vite 8.2.2, Vitest 4.1.11, Testing Library Preact 3.2.4, jsdom 30.0.1, fake-indexeddb 6.2.5, FastAPI, Docker multistage Node 24/PyTorch.

**Spec:** `docs/superpowers/specs/2026-08-27-web-interface-design.md`

## Global Constraints

- Un seul conteneur, un seul port `127.0.0.1:8000`, aucun serveur Node en production.
- Aucun compte, aucune authentification et aucune base de données serveur.
- Aucun CDN, aucune police distante, aucune télémétrie et aucun secret dans le bundle.
- Routes par ancres uniquement ; ne jamais ajouter de fallback HTTP universel.
- Ne jamais persister automatiquement un fichier importé, une prise micro ou une référence vocale.
- Les blobs TTS ne sont persistés qu'après l'action explicite « Conserver ».
- Preact et TypeScript strict ; composants sans `fetch` ni IndexedDB directs.
- Interface clavier, contraste WCAG AA, `prefers-reduced-motion` et largeur minimale 360 px.
- TDD obligatoire : observer chaque test échouer avant l'implémentation correspondante.
- Toute commande de ce dépôt reste préfixée par `rtk`, conformément à `C:\Users\pc\.codex\RTK.md`.
- Toutes les commandes `rtk npm` s'exécutent depuis `web/`, sauf indication explicite contraire.

Versions vérifiées le 2026-08-27 sur les registres officiels npm :
[Preact](https://www.npmjs.com/package/preact),
[Vite](https://www.npmjs.com/package/vite),
[Vitest](https://www.npmjs.com/package/vitest),
[@preact/preset-vite](https://www.npmjs.com/package/@preact/preset-vite),
[TypeScript](https://www.npmjs.com/package/typescript),
[jsdom](https://www.npmjs.com/package/jsdom) et
[fake-indexeddb](https://www.npmjs.com/package/fake-indexeddb).

## File Map

- `web/src/api/` contient uniquement contrats et transport HTTP.
- `web/src/app/` contient composition, routes et injection des services.
- `web/src/features/<feature>/` regroupe écran, logique API et tests d'une capacité.
- `web/src/media/` possède les ressources micro et URLs audio.
- `web/src/storage/` est l'unique accès à IndexedDB.
- `web/src/ui/` reçoit les composants visuels sans logique métier.
- `web/src/utils/exports/` dérive les formats sans nouvel appel GPU.
- `src/transcription_server/api/web.py` est l'unique intégration backend du build.
- `docker/Dockerfile` construit le frontend puis copie seulement `dist/`.

---

### Task 1: Socle Vite, coque et navigation par ancres

**Files:**
- Create: `web/package.json`
- Create: `web/package-lock.json`
- Create: `web/tsconfig.json`
- Create: `web/vite.config.ts`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/app/App.tsx`
- Create: `web/src/app/routes.ts`
- Create: `web/src/styles/tokens.css`
- Create: `web/src/styles/app.css`
- Create: `web/src/ui/Button.tsx`
- Create: `web/src/ui/Field.tsx`
- Create: `web/src/ui/OperationStatus.tsx`
- Create: `web/src/ui/ConfirmDialog.tsx`
- Create: `web/src/test/setup.ts`
- Test: `web/src/app/App.test.tsx`
- Test: `web/src/ui/ConfirmDialog.test.tsx`

**Interfaces:**
- Produces: `RouteName`, `readRoute(hash: string): RouteName`, `routeHref(route: RouteName): string`.
- Produces: la coque responsive et les cinq points de montage d'écran.
- Produces: `Button`, `Field`, `OperationStatus` et `ConfirmDialog` communs.

- [ ] **Step 1: Créer la chaîne de test reproductible**

Créer `web/package.json` avec exactement ces scripts et versions vérifiées :

```json
{
  "name": "resume-transcription-web",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "test": "vitest run",
    "test:watch": "vitest",
    "typecheck": "tsc --noEmit",
    "build": "tsc --noEmit && vite build"
  },
  "dependencies": {
    "preact": "10.29.8"
  },
  "devDependencies": {
    "@preact/preset-vite": "2.10.6",
    "@testing-library/preact": "3.2.4",
    "fake-indexeddb": "6.2.5",
    "jsdom": "30.0.1",
    "typescript": "7.0.2",
    "vite": "8.2.2",
    "vitest": "4.1.11"
  }
}
```

Configurer TypeScript avec `strict`, `noUncheckedIndexedAccess`,
`jsxImportSource: "preact"`, DOM/ES2024 et aucun emit. Configurer Vite avec le
preset Preact, un bloc test `environment: "jsdom"`/`setupFiles`, et
un proxy de développement pour `/health`, `/transcribe`, `/summarize` et `/v1`
vers `http://127.0.0.1:8000`. Générer le lock avec :

```ts
import preact from "@preact/preset-vite";
import { defineConfig } from "vite";

const api = "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [preact()],
  server: {
    proxy: {
      "/health": api,
      "/transcribe": api,
      "/summarize": api,
      "/v1": api,
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
```

Run: `rtk npm install --package-lock-only`

- [ ] **Step 2: Écrire le test rouge de navigation**

```tsx
import { fireEvent, render, screen } from "@testing-library/preact";
import { afterEach, describe, expect, it } from "vitest";
import { App } from "./App";

describe("App", () => {
  afterEach(() => { window.location.hash = ""; });

  it("ouvre Transcrire et expose les cinq destinations", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "Transcrire" })).toBeTruthy();
    for (const name of ["Transcrire", "Résumer", "Synthétiser", "Voix", "Historique"]) {
      expect(screen.getByRole("link", { name })).toBeTruthy();
    }
  });

  it("suit une ancre sans requête serveur", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("link", { name: "Voix" }));
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    expect(window.location.hash).toBe("#/voices");
    expect(screen.getByRole("heading", { name: "Voix" })).toBeTruthy();
  });
});
```

Dans `ConfirmDialog.test.tsx`, écrire également :

```tsx
it("place le focus sur Annuler", () => {
  render(
    <ConfirmDialog
      open title="Supprimer ?" confirmLabel="Supprimer" danger
      onConfirm={() => undefined} onCancel={() => undefined}
    >Cette action est définitive.</ConfirmDialog>,
  );
  expect(document.activeElement).toBe(screen.getByRole("button", { name: "Annuler" }));
});
```

- [ ] **Step 3: Vérifier l'échec**

Run: `rtk npm test -- --run src/app/App.test.tsx src/ui/ConfirmDialog.test.tsx`

Expected: FAIL, `App`, `routes` et `ConfirmDialog` n'existent pas.

- [ ] **Step 4: Implémenter le routeur et la coque minimale**

```ts
export type RouteName = "transcribe" | "summarize" | "speech" | "voices" | "history";

const paths: Record<RouteName, string> = {
  transcribe: "#/transcribe",
  summarize: "#/summarize",
  speech: "#/speech",
  voices: "#/voices",
  history: "#/history",
};

export function readRoute(hash: string): RouteName {
  const path = hash.split("?", 1)[0];
  const found = Object.entries(paths).find(([, value]) => value === path);
  return found?.[0] as RouteName ?? "transcribe";
}

export const routeHref = (route: RouteName): string => paths[route];
```

`App.tsx` écoute `hashchange`, rend un `<header>`, un `<nav aria-label="Navigation principale">`
et un `<main>` contenant le titre de la route active. Utiliser les libellés
français validés et `aria-current="page"` sur le lien actif.

Définir dans `tokens.css` les six couleurs validées, les trois piles de police,
les espacements 4/8/12/16/24/32 px, un rayon unique de 10 px et un focus de
3 px `#176B75`. Dans `app.css`, utiliser une grille `auto 1fr`, une colonne de
navigation de 13rem, puis une barre inférieure sous 720 px. Ajouter un bloc
`@media (prefers-reduced-motion: reduce)` qui neutralise transitions et scroll
animé.

Créer les quatre primitives UI sans logique métier. `Button` accepte
`variant: "primary" | "secondary" | "danger"`; `Field` lie libellé, aide et
erreur par identifiants ARIA ; `OperationStatus` reçoit `{ active, label,
startedAt }` et annonce libellé/durée ; `ConfirmDialog` reçoit `{ open, title,
confirmLabel, danger, onConfirm, onCancel, children }`, place le focus sur
Annuler à l'ouverture et le restitue au déclencheur à la fermeture.

- [ ] **Step 5: Vérifier le socle**

Run: `rtk npm test -- --run src/app/App.test.tsx src/ui/ConfirmDialog.test.tsx`

Expected: 2 tests PASS.

Run: `rtk npm run typecheck`

Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
rtk git add web
rtk git commit -m "feat: scaffold Preact web interface"
```

### Task 2: Client HTTP typé et normalisation des erreurs

**Files:**
- Create: `web/src/api/contracts.ts`
- Create: `web/src/api/http.ts`
- Test: `web/src/api/http.test.ts`

**Interfaces:**
- Produces: `ApiFailure`, `HttpClient.getJson<T>()`, `postJson<T>()`, `postForm<T>()`, `postBlob()` et `delete()`.
- Produces: les contrats `Health`, `TranscriptionResult`, `SummaryResult`, `Voice`, `SpeechRequest`, `AudioResult` et les types d'entrée multipart.

- [ ] **Step 1: Écrire les tests rouges des deux enveloppes d'erreur**

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiFailure, HttpClient } from "./http";

afterEach(() => vi.unstubAllGlobals());

describe("HttpClient", () => {
  it("normalise une erreur OpenAI", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ error: { message: "Voix inconnue.", code: "voice_not_found", param: "voice" } }),
      { status: 404, headers: { "content-type": "application/json" } },
    )));
    await expect(new HttpClient().getJson("/v1/voices/x")).rejects.toMatchObject({
      status: 404, code: "voice_not_found", param: "voice", message: "Voix inconnue.",
    });
  });

  it("normalise le detail FastAPI", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: "Le fichier est invalide." }),
      { status: 400, headers: { "content-type": "application/json" } },
    )));
    await expect(new HttpClient().postForm("/transcribe", new FormData())).rejects.toBeInstanceOf(ApiFailure);
  });
});
```

- [ ] **Step 2: Vérifier l'échec**

Run: `rtk npm test -- --run src/api/http.test.ts`

Expected: FAIL, module `./http` absent.

- [ ] **Step 3: Implémenter le transport commun**

```ts
export class ApiFailure extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
    readonly param?: string,
  ) { super(message); this.name = "ApiFailure"; }
}

async function failure(response: Response): Promise<ApiFailure> {
  const fallback = `Le serveur a répondu ${response.status}.`;
  try {
    const body = await response.json() as {
      error?: { message?: string; code?: string; param?: string };
      detail?: string | { message?: string; code?: string; param?: string };
    };
    const detail = typeof body.detail === "string" ? { message: body.detail } : body.detail;
    const source = body.error ?? detail;
    return new ApiFailure(source?.message ?? fallback, response.status, source?.code, source?.param);
  } catch {
    return new ApiFailure(fallback, response.status);
  }
}
```

Toutes les méthodes appellent un unique `request(path, init)`, lèvent
`failure(response)` pour `!response.ok`, ne définissent jamais manuellement
`content-type` pour `FormData`, et vérifient `content-type` avant `json()`.
`postBlob()` retourne `{ blob, contentType, filename? }`, le nom venant de
`Content-Disposition` lorsqu'il existe.

Déclarer dans `contracts.ts` les champs exactement tels que les schémas Python
les exposent, y compris `turns[].words`, `tts.vram_allocated_mib`, les trois
modes Qwen, les six formats audio et `Voice.kind: "builtin" | "clone"`.
`AudioResult` vaut `{ blob: Blob; contentType: string; filename?: string }`.

- [ ] **Step 4: Vérifier transport et types**

Run: `rtk npm test -- --run src/api/http.test.ts`

Expected: 2 tests PASS.

Run: `rtk npm run typecheck`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
rtk git add web/src/api
rtk git commit -m "feat: add typed web API transport"
```

### Task 3: Dépôt IndexedDB privé et rétention

**Files:**
- Create: `web/src/storage/history.ts`
- Create: `web/src/storage/browser-history.ts`
- Test: `web/src/storage/browser-history.test.ts`

**Interfaces:**
- Produces: `HistoryEntry`, `HistoryDraft`, `HistoryLimits`, `HistoryRepository`.
- Produces: `BrowserHistoryRepository.add()`, `list()`, `get()`, `remove()`, `keepAudio()`, `confirmAudioEviction()`, `clear()`, `getLimits()` et `setLimits()`.

- [ ] **Step 1: Écrire les tests rouges de confidentialité et rétention**

```ts
import "fake-indexeddb/auto";
import { beforeEach, expect, it } from "vitest";
import { BrowserHistoryRepository } from "./browser-history";

beforeEach(async () => {
  await new Promise<void>((resolve, reject) => {
    const request = indexedDB.deleteDatabase("resume-transcription");
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
});

it("enregistre le texte mais aucun audio source", async () => {
  const repository = new BrowserHistoryRepository();
  const saved = await repository.add({
    kind: "transcription", title: "réunion.wav", parameters: { diarize: true },
    resultText: "Bonjour", metadata: { duration: 3.2 },
  });
  expect(saved.audio).toBeUndefined();
  expect((await repository.get(saved.id))?.resultText).toBe("Bonjour");
});

it("ne conserve un blob TTS que sur demande explicite", async () => {
  const repository = new BrowserHistoryRepository();
  const saved = await repository.add({
    kind: "speech", title: "Bonjour", parameters: { voice: "Ryan" },
    resultText: "Bonjour", metadata: {},
  });
  await repository.keepAudio(saved.id, new Blob(["audio"], { type: "audio/mpeg" }));
  expect((await repository.get(saved.id))?.audio?.size).toBe(5);
});
```

- [ ] **Step 2: Vérifier l'échec**

Run: `rtk npm test -- --run src/storage/browser-history.test.ts`

Expected: FAIL, dépôt absent.

- [ ] **Step 3: Définir le contrat et le schéma version 1**

```ts
export type HistoryKind = "transcription" | "summary" | "speech";
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
export interface HistoryEntry {
  id: string;
  createdAt: string;
  kind: HistoryKind;
  title: string;
  parameters: Record<string, string | number | boolean | null>;
  resultText: string;
  metadata: Record<string, JsonValue>;
  audio?: Blob;
}
export type HistoryDraft = Omit<HistoryEntry, "id" | "createdAt" | "audio">;
export interface HistoryLimits { maxEntries: number; maxAudioBytes: number; }
export interface HistoryRepository {
  add(draft: HistoryDraft): Promise<HistoryEntry>;
  list(): Promise<HistoryEntry[]>;
  get(id: string): Promise<HistoryEntry | undefined>;
  remove(id: string): Promise<void>;
  keepAudio(id: string, audio: Blob): Promise<{ evictedIds: string[] }>;
  confirmAudioEviction(id: string, audio: Blob, evictedIds: string[]): Promise<void>;
  clear(): Promise<void>;
  getLimits(): Promise<HistoryLimits>;
  setLimits(limits: HistoryLimits): Promise<void>;
}
```

Ouvrir `resume-transcription` version 1 avec les stores `entries` (`keyPath:
"id"`, index `createdAt`) et `settings`. Les limites par défaut sont
`{ maxEntries: 100, maxAudioBytes: 262_144_000 }`. `add()` génère
`crypto.randomUUID()`, applique la limite d'entrées du plus ancien au plus
récent et ne possède aucun paramètre de fichier source. `keepAudio()` calcule
la somme des `Blob.size`, retourne les identifiants qu'il faudrait évincer et
n'écrit rien si la limite serait dépassée ; une seconde méthode
`confirmAudioEviction(id, audio, evictedIds)` réalise l'écriture après accord UI.

- [ ] **Step 4: Ajouter les tests de dépassement et purge**

Tester que `keepAudio()` propose les plus anciens audios, que
`confirmAudioEviction()` ne supprime que leurs blobs et que `clear()` rend
`list()` vide. Utiliser des blobs de 4 octets et une limite de 6 octets pour
éviter des allocations massives.

- [ ] **Step 5: Vérifier le dépôt**

Run: `rtk npm test -- --run src/storage/browser-history.test.ts`

Expected: tous les tests PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add web/src/storage
rtk git commit -m "feat: persist private browser history"
```

### Task 4: Capture microphone, aperçu et forme d'onde fonctionnelle

**Files:**
- Create: `web/src/media/recorder.ts`
- Create: `web/src/media/use-recorder.ts`
- Create: `web/src/media/AudioSourcePicker.tsx`
- Create: `web/src/media/Waveform.tsx`
- Test: `web/src/media/recorder.test.ts`
- Test: `web/src/media/AudioSourcePicker.test.tsx`

**Interfaces:**
- Produces: `RecordedAudio { blob, filename, durationMs }`.
- Produces: `RecorderPort.start()`, `stop()`, `cancel()` et `state()`.
- Produces: `AudioSourcePicker({ value, onChange, referenceMode })`.

- [ ] **Step 1: Écrire le test rouge de libération du micro**

```ts
import { expect, it, vi } from "vitest";
import { BrowserRecorder, type MediaRecorderPort } from "./recorder";

class FakeMediaRecorder implements MediaRecorderPort {
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  start = vi.fn();
  stop = vi.fn();

  finish(blob: Blob): void {
    this.ondataavailable?.({ data: blob });
    this.onstop?.();
  }
}

it("arrête toutes les pistes après la capture", async () => {
  const stopTrack = vi.fn();
  const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream;
  const fakeMediaRecorder = new FakeMediaRecorder();
  let now = 1_000;
  const recorder = new BrowserRecorder(
    async () => stream,
    () => now,
    () => fakeMediaRecorder as MediaRecorderPort,
  );
  await recorder.start();
  const pending = recorder.stop();
  now = 4_500;
  fakeMediaRecorder.finish(new Blob(["voice"], { type: "audio/webm" }));
  const result = await pending;
  expect(result.durationMs).toBe(3_500);
  expect(stopTrack).toHaveBeenCalledOnce();
});
```

- [ ] **Step 2: Vérifier l'échec**

Run: `rtk npm test -- --run src/media/recorder.test.ts`

Expected: FAIL, service absent.

- [ ] **Step 3: Implémenter le service de capture**

Définir `MediaRecorderPort` avec les callbacks `ondataavailable`, `onstop` et
les méthodes `start()`/`stop()`, puis injecter sa fabrique dans
`BrowserRecorder` comme le montre le test.

Choisir le premier MIME supporté dans `audio/webm;codecs=opus`, `audio/ogg;codecs=opus`,
puis `audio/webm`. `start()` appelle `getUserMedia({ audio: true })`, crée
`MediaRecorder`, mémorise les chunks non vides et l'instant initial. `stop()`
attend l'événement `stop`, assemble le blob, nomme le fichier
`enregistrement-<timestamp>.webm|ogg` et arrête les pistes dans `finally`.
`cancel()` arrête recorder et pistes, vide les chunks et ne produit aucun blob.

Le hook `useRecorder` expose `idle | requesting | recording | stopped | error`,
la durée écoulée et les actions. `AudioSourcePicker` rend un input fichier et
les commandes micro ; il révoque toute URL d'objet remplacée ou démontée.
En `referenceMode`, il refuse côté UI une durée hors de 3 000 à 30 000 ms.

`Waveform` utilise `AudioContext.createMediaStreamSource()` et
`AnalyserNode.getByteTimeDomainData()` pendant la capture. Son canvas possède
un équivalent textuel `aria-label="Niveau du microphone"`; la boucle
`requestAnimationFrame` et l'AudioContext sont arrêtés au démontage.

- [ ] **Step 4: Tester annulation, URL et bornes de référence**

Avec Testing Library, vérifier qu'« Annuler » appelle `cancel()`, que le bouton
d'envoi reste désactivé sous 3 secondes en mode référence et que
`URL.revokeObjectURL` est appelé lors du remplacement d'un média.

- [ ] **Step 5: Vérifier les médias**

Run: `rtk npm test -- --run src/media`

Expected: tous les tests PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add web/src/media
rtk git commit -m "feat: capture browser microphone audio"
```

### Task 5: Santé du serveur et services injectables

**Files:**
- Create: `web/src/app/services.ts`
- Create: `web/src/features/health/health-api.ts`
- Create: `web/src/features/health/use-health.ts`
- Create: `web/src/features/health/HealthBar.tsx`
- Create: `web/src/features/health/HealthPanel.tsx`
- Create: `web/src/test/fakes.tsx`
- Test: `web/src/features/health/HealthBar.test.tsx`
- Modify: `web/src/app/App.tsx`

**Interfaces:**
- Produces: `AppServices { http, history, recorderFactory }` et `ServicesProvider`.
- Produces: `getHealth(http: HttpClient): Promise<Health>`.
- Produces: `useHealth()` avec état, dernière mise à jour et rafraîchissement.

- [ ] **Step 1: Écrire le test rouge du bandeau dégradé**

```tsx
it("explique un worker Qwen indisponible", async () => {
  render(<HealthBar />, { wrapper: fakeServices({ health: {
    status: "degraded", device: "cuda", gpu: { name: "RTX 3090" },
    tts: { enabled: true, worker: false, state: "error", last_error: "worker_unreachable" },
  } }).wrapper });
  expect(await screen.findByText("Synthèse indisponible")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Détails du serveur" }));
  expect(screen.getByText("worker_unreachable")).toBeTruthy();
});
```

- [ ] **Step 2: Vérifier l'échec**

Run: `rtk npm test -- --run src/features/health/HealthBar.test.tsx`

Expected: FAIL, composant absent.

- [ ] **Step 3: Implémenter injection, polling et affichage**

`ServicesProvider` fournit les trois ports sans singleton caché. La production
instancie `new HttpClient()`, `new BrowserHistoryRepository()` et
`new BrowserRecorder()` dans `main.tsx`.

Créer dans `src/test/fakes.tsx` `fakeServices(options?: FakeOptions)` qui rend
`{ wrapper, http, history, recorderFactory }`. `FakeOptions` accepte `health`,
`historyEntry`, `historyEntries`, `voices`, `speechBlob`; les méthodes sont des
spies Vitest et `http.lastForm` expose le dernier multipart. `health` et
`historyEntry` sont des `DeepPartial` complétés par le helper ;
`historyEntries` est une liste complète de `HistoryEntry`. Ce helper est le
seul faux partagé consommé par les Tasks 6 à 10.

```ts
export type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};
```

`useHealth()` appelle `/health` immédiatement, toutes les 10 secondes si
`document.visibilityState === "visible"`, toutes les 30 secondes sinon, et à
chaque retour visible. Nettoyer timer et écouteur au démontage. Une panne garde
le dernier résultat et expose l'erreur séparément.

Le bandeau nomme `Opérationnel`, `Dégradé` ou `Hors ligne`. Le panneau détaille
GPU, VRAM, Parakeet, diarisation, résumé, Qwen, checkpoint chargé et modèles
téléchargés. Aucun état ne dépend uniquement d'une couleur.

- [ ] **Step 4: Vérifier polling et accessibilité**

Run: `rtk npm test -- --run src/features/health`

Expected: tests PASS avec timers Vitest simulés.

- [ ] **Step 5: Commit**

```bash
rtk git add web/src/app web/src/features/health web/src/main.tsx
rtk git commit -m "feat: show live server health in web UI"
```

### Task 6: Parcours de transcription et exports locaux

**Files:**
- Create: `web/src/features/transcription/transcription-api.ts`
- Create: `web/src/features/transcription/TranscriptionPage.tsx`
- Create: `web/src/features/transcription/TranscriptionOptions.tsx`
- Create: `web/src/features/transcription/TranscriptionResult.tsx`
- Create: `web/src/utils/exports/transcription.ts`
- Test: `web/src/features/transcription/TranscriptionPage.test.tsx`
- Test: `web/src/utils/exports/transcription.test.ts`
- Modify: `web/src/app/App.tsx`

**Interfaces:**
- Produces: `transcribe(http, input): Promise<TranscriptionResult>`.
- Produces: `toText()`, `toDialogue()`, `toSrt()`, `toVtt()` et `toJson()`.
- Stores: une `HistoryEntry` automatique de type `transcription`.

- [ ] **Step 1: Écrire les tests rouges du formulaire et du SRT**

```tsx
it("envoie un fichier avec diarisation et l'historise", async () => {
  const services = fakeServices();
  render(<TranscriptionPage />, { wrapper: services.wrapper });
  fireEvent.change(screen.getByLabelText("Fichier audio"), {
    target: { files: [new File(["audio"], "réunion.wav", { type: "audio/wav" })] },
  });
  fireEvent.click(screen.getByLabelText("Séparer les locuteurs"));
  fireEvent.click(screen.getByRole("button", { name: "Transcrire" }));
  expect(await screen.findByText("Bonjour à tous")).toBeTruthy();
  expect(services.history.add).toHaveBeenCalledWith(expect.objectContaining({ kind: "transcription" }));
});
```

```ts
expect(toSrt(resultWithOneTurn)).toBe(
  "1\n00:00:01,250 --> 00:00:03,500\nSPEAKER_00 : Bonjour\n",
);
```

- [ ] **Step 2: Vérifier les échecs**

Run: `rtk npm test -- --run src/features/transcription src/utils/exports`

Expected: FAIL, modules absents.

- [ ] **Step 3: Implémenter l'appel et les règles du formulaire**

Construire `FormData` avec `file`, `response_format=json`, `word_timestamps`,
`language` seulement si non automatique, `diarize`, `channels`, puis les
champs locuteurs renseignés. Lorsqu'un nombre exact est saisi, vider min/max ;
lorsqu'une borne est saisie, vider le nombre exact. Utiliser
`AudioSourcePicker` pour fichier et micro.

Pendant l'appel, afficher `Transcription en cours — mm:ss` dans une région
`aria-live="polite"`, désactiver seulement la double soumission et laisser la
navigation disponible. À succès, historiser texte, paramètres, durée,
locuteurs et réponse JSON, jamais le `File`.

Les exports formatent les temps sans dépendance, échappent les retours de ligne
et créent les téléchargements avec le MIME et l'extension corrects. Le bouton
« Résumer cette transcription » navigue vers
`#/summarize?history=<encodeURIComponent(id)>`.

- [ ] **Step 4: Vérifier transcription et exports**

Run: `rtk npm test -- --run src/features/transcription src/utils/exports`

Expected: tous les tests PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add web/src/features/transcription web/src/utils/exports web/src/app/App.tsx
rtk git commit -m "feat: add web transcription workflow"
```

### Task 7: Parcours de résumé depuis audio, texte ou historique

**Files:**
- Create: `web/src/features/summary/summary-api.ts`
- Create: `web/src/features/summary/SummaryPage.tsx`
- Test: `web/src/features/summary/SummaryPage.test.tsx`
- Modify: `web/src/app/App.tsx`

**Interfaces:**
- Consumes: `HttpClient`, `HistoryRepository`, `AudioSourcePicker` et le paramètre `history` de l'ancre.
- Produces: `summarize(http, input): Promise<SummaryResult>`.
- Stores: une entrée automatique `summary`.

- [ ] **Step 1: Écrire le test rouge de reprise d'historique**

```tsx
it("charge une transcription locale sans renvoyer son audio", async () => {
  window.location.hash = "#/summarize?history=entry-1";
  const services = fakeServices({ historyEntry: {
    id: "entry-1", kind: "transcription", resultText: "Décision validée.",
  } });
  render(<SummaryPage />, { wrapper: services.wrapper });
  expect(await screen.findByDisplayValue("Décision validée.")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Rédiger le résumé" }));
  expect(services.http.lastForm?.get("transcript")).toBe("Décision validée.");
  expect(services.http.lastForm?.has("file")).toBe(false);
});
```

- [ ] **Step 2: Vérifier l'échec**

Run: `rtk npm test -- --run src/features/summary/SummaryPage.test.tsx`

Expected: FAIL, écran absent.

- [ ] **Step 3: Implémenter les trois sources exclusives**

Utiliser un sélecteur `Audio | Texte | Historique`. Un changement de source
efface les valeurs des deux autres. Pour audio, envoyer `file`, `format`,
`diarize`, `channels`, `language`, `response_format=json`. Pour texte ou
historique, envoyer seulement `transcript`, `format` et `response_format=json`.

Lire l'identifiant depuis `new URLSearchParams(location.hash.split("?", 2)[1])`.
Si l'entrée manque, afficher « Cette transcription n'existe plus dans
l'historique local » et laisser le champ texte vide. Désactiver l'action si
`health.summary_enabled === false` avec l'instruction de configuration fournie
par le README.

À succès, historiser le résumé, le format et le modèle ; proposer copier et
télécharger en `.md`.

- [ ] **Step 4: Vérifier le parcours**

Run: `rtk npm test -- --run src/features/summary`

Expected: tests PASS pour les trois sources, service désactivé et erreur 503.

- [ ] **Step 5: Commit**

```bash
rtk git add web/src/features/summary web/src/app/App.tsx
rtk git commit -m "feat: add web summary workflow"
```

### Task 8: Synthèse Qwen et clonage ponctuel

**Files:**
- Create: `web/src/features/speech/speech-api.ts`
- Create: `web/src/features/speech/SpeechPage.tsx`
- Create: `web/src/features/speech/SpeechResult.tsx`
- Create: `web/src/features/speech/OneShotClone.tsx`
- Test: `web/src/features/speech/SpeechPage.test.tsx`
- Test: `web/src/features/speech/OneShotClone.test.tsx`
- Modify: `web/src/app/App.tsx`

**Interfaces:**
- Produces: `createSpeech(http, request): Promise<AudioResult>`.
- Produces: `cloneOnce(http, input): Promise<AudioResult>`.
- Consumes: `GET /v1/voices`, `AudioSourcePicker`, santé TTS et historique.

- [ ] **Step 1: Écrire les tests rouges des champs conditionnels**

```tsx
it("exige une instruction seulement pour VoiceDesign", () => {
  render(<SpeechPage />, { wrapper: fakeServices().wrapper });
  fireEvent.change(screen.getByLabelText("Mode vocal"), {
    target: { value: "qwen3-tts-voice-design" },
  });
  expect(screen.getByLabelText("Description de la voix")).toBeTruthy();
  expect(screen.queryByLabelText("Voix")).toBeNull();
});

it("ne conserve pas automatiquement le blob généré", async () => {
  const services = fakeServices({ speechBlob: new Blob(["mp3"], { type: "audio/mpeg" }) });
  render(<SpeechPage />, { wrapper: services.wrapper });
  fireEvent.input(screen.getByLabelText("Texte à prononcer"), { target: { value: "Bonjour" } });
  fireEvent.click(screen.getByRole("button", { name: "Créer l’audio" }));
  await screen.findByRole("button", { name: "Conserver" });
  expect(services.history.keepAudio).not.toHaveBeenCalled();
});
```

- [ ] **Step 2: Vérifier les échecs**

Run: `rtk npm test -- --run src/features/speech`

Expected: FAIL, écrans absents.

- [ ] **Step 3: Implémenter la synthèse persistante**

Charger `/v1/voices` à l'entrée de page. Construire le JSON avec `model`,
`input`, `voice`, `instructions`, `response_format`, `speed`, `language` en
omettant les champs incompatibles. Afficher le compteur `n / 4096`.

CustomVoice filtre `kind=builtin`; clone persistant filtre `kind=clone` ;
VoiceDesign montre uniquement l'instruction. Le résultat crée une URL d'objet,
un lecteur et un téléchargement. Historiser texte et paramètres sans blob.
« Conserver » appelle `keepAudio()` ; si des identifiants sont proposés,
demander confirmation avant `confirmAudioEviction()`.

- [ ] **Step 4: Implémenter le clone ponctuel**

`OneShotClone` envoie multipart `file`, `input`, `consent`, `transcript`
facultatif, `language`, `response_format`, `speed`. La case de consentement est
fausse à chaque montage et après chaque succès. Ne passer ni le fichier ni sa
transcription de référence à `history.add()` ; seule la phrase synthétisée et
les paramètres non sensibles peuvent être historisés.

- [ ] **Step 5: Vérifier synthèse et confidentialité**

Run: `rtk npm test -- --run src/features/speech`

Expected: tests PASS pour trois modes, 4096 caractères, consentement, erreur
Qwen, conservation explicite et révocation d'URL.

- [ ] **Step 6: Commit**

```bash
rtk git add web/src/features/speech web/src/app/App.tsx
rtk git commit -m "feat: add Qwen speech web workflows"
```

### Task 9: Gestion des voix consenties

**Files:**
- Create: `web/src/features/voices/voices-api.ts`
- Create: `web/src/features/voices/VoicesPage.tsx`
- Create: `web/src/features/voices/VoiceEnrollment.tsx`
- Test: `web/src/features/voices/VoicesPage.test.tsx`
- Test: `web/src/features/voices/VoiceEnrollment.test.tsx`
- Modify: `web/src/app/App.tsx`

**Interfaces:**
- Produces: `listVoices()`, `createVoice()` et `deleteVoice()`.
- Consumes: `AudioSourcePicker` en `referenceMode` et `Voice` du contrat API.

- [ ] **Step 1: Écrire les tests rouges du consentement et de la suppression**

```tsx
it("interdit l'inscription sans consentement", () => {
  render(<VoiceEnrollment onCreated={() => undefined} />, { wrapper: fakeServices().wrapper });
  const button = screen.getByRole("button", { name: "Enregistrer la voix" }) as HTMLButtonElement;
  expect(button.disabled).toBe(true);
});

it("ne propose jamais de supprimer une voix prédéfinie", async () => {
  render(<VoicesPage />, { wrapper: fakeServices({ voices: [
    { id: "Ryan", name: "Ryan", kind: "builtin" },
    { id: "clone-1", name: "Ma voix", kind: "clone", language: "fr", duration: 8 },
  ] }).wrapper });
  expect(await screen.findByText("Ryan")).toBeTruthy();
  expect(screen.getAllByRole("button", { name: /Supprimer/ })).toHaveLength(1);
});
```

- [ ] **Step 2: Vérifier les échecs**

Run: `rtk npm test -- --run src/features/voices`

Expected: FAIL, composants absents.

- [ ] **Step 3: Implémenter liste et inscription**

Grouper `kind=builtin` et `kind=clone`. Afficher pour un clone nom, langue,
durée, date et origine de transcription. Le formulaire envoie `file`, `name`,
`language`, `transcript` si non vide et `consent=true`. Après succès, vider le
média, le texte, le consentement et recharger la liste.

Afficher avant l'action : « Vous confirmez avoir le droit d'utiliser cette
voix. La référence sera stockée localement dans le volume Docker. » Ne jamais
précocher la case.

- [ ] **Step 4: Implémenter la suppression confirmée**

Ouvrir un `<dialog>` nommé « Supprimer Ma voix ? ». Le bouton destructif appelle
`DELETE /v1/voices/clone-1`, ferme le dialogue et recharge. Annuler ne lance
aucun appel. Une erreur `builtin_voice` ne doit être atteignable que si le
serveur contredit la liste et s'affiche comme erreur technique.

- [ ] **Step 5: Vérifier les voix**

Run: `rtk npm test -- --run src/features/voices`

Expected: tests PASS pour fichier, micro, transcription facultative, création,
annulation et suppression.

- [ ] **Step 6: Commit**

```bash
rtk git add web/src/features/voices web/src/app/App.tsx
rtk git commit -m "feat: manage consented voices in web UI"
```

### Task 10: Écran Historique et reprises inter-écrans

**Files:**
- Create: `web/src/features/history/HistoryPage.tsx`
- Create: `web/src/features/history/HistoryDetail.tsx`
- Create: `web/src/features/history/HistorySettings.tsx`
- Test: `web/src/features/history/HistoryPage.test.tsx`
- Modify: `web/src/app/App.tsx`
- Modify: `web/src/features/speech/SpeechPage.tsx`

**Interfaces:**
- Consumes: toutes les méthodes `HistoryRepository`.
- Produces: navigation vers `#/summarize?history=<id>` et `#/speech?history=<id>`.

- [ ] **Step 1: Écrire le test rouge des reprises**

```tsx
it("reprend une transcription pour le résumé", async () => {
  const transcriptionEntry = {
    id: "entry-1", createdAt: "2026-08-27T10:00:00Z", kind: "transcription" as const,
    title: "réunion.wav", parameters: {}, resultText: "Décision validée.", metadata: {},
  };
  render(<HistoryPage />, { wrapper: fakeServices({ historyEntries: [transcriptionEntry] }).wrapper });
  fireEvent.click(await screen.findByRole("button", { name: "Ouvrir réunion.wav" }));
  fireEvent.click(screen.getByRole("link", { name: "Résumer ce texte" }));
  expect(window.location.hash).toBe("#/summarize?history=entry-1");
});
```

- [ ] **Step 2: Vérifier l'échec**

Run: `rtk npm test -- --run src/features/history/HistoryPage.test.tsx`

Expected: FAIL, écran absent.

- [ ] **Step 3: Implémenter liste, détail et actions**

Trier du plus récent au plus ancien, filtrer par type et rendre des cartes sous
720 px. Le détail affiche paramètres, résultat, métadonnées et lecteur si un
blob existe. Les actions disponibles dépendent du type : résumé depuis tout
texte, synthèse depuis tout texte, téléchargement audio seulement si blob,
suppression avec confirmation.

Dans `SpeechPage`, lire `history` depuis l'ancre et préremplir uniquement
`resultText`, jamais paramètres de clone ou média. Si l'entrée n'existe plus,
afficher un message non bloquant.

- [ ] **Step 4: Implémenter limites et purge**

`HistorySettings` valide `maxEntries` entre 10 et 1 000 et `maxAudioBytes`
entre 10 MiB et le plus petit de 2 GiB ou 80 % de `navigator.storage.estimate().quota`.
« Effacer l'historique local » exige une confirmation puis appelle `clear()`.
Afficher espace audio utilisé, limite et nombre d'opérations.

- [ ] **Step 5: Vérifier l'historique complet**

Run: `rtk npm test -- --run src/features/history src/features/summary src/features/speech`

Expected: tests PASS pour filtres, reprises, audio absent, suppression, limites
et purge.

- [ ] **Step 6: Commit**

```bash
rtk git add web/src/features/history web/src/features/speech web/src/app/App.tsx
rtk git commit -m "feat: browse and reuse local web history"
```

### Task 11: Distribution statique FastAPI et politique navigateur

**Files:**
- Create: `src/transcription_server/api/web.py`
- Modify: `src/transcription_server/config.py`
- Modify: `src/transcription_server/app.py`
- Modify: `tests/conftest.py`
- Modify: `tests/unit/test_config.py`
- Test: `tests/unit/test_web_ui.py`

**Interfaces:**
- Produces: `mount_web_ui(app: FastAPI, dist_path: Path) -> None`.
- Adds: `Settings.web_dist_path: Path = Path("/app/web-dist")`.

- [ ] **Step 1: Écrire les tests Python rouges**

```python
from fastapi.testclient import TestClient

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import NullDiarizationEngine


def make_client(web_dist_path):
    settings = Settings(
        _env_file=None,
        enable_diarization=False,
        enable_summary=False,
        enable_tts=False,
        device="cpu",
        web_dist_path=web_dist_path,
    )
    app = create_app(settings, StubAsrEngine([]), NullDiarizationEngine())
    return TestClient(app)


def test_racine_sert_le_build_sans_masquer_docs_ni_api(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<main>VoxLab</main>", encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("export {};", encoding="utf-8")
    client = make_client(tmp_path)

    assert client.get("/").text == "<main>VoxLab</main>"
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/v1/inconnue").status_code == 404
    assert "default-src 'self'" in client.get("/").headers["content-security-policy"]


def test_build_absent_laisse_api_demarrer(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/").status_code == 404
    assert client.get("/health").status_code == 200


def test_chemin_du_build_web_est_configurable(monkeypatch, tmp_path):
    monkeypatch.setenv("WEB_DIST_PATH", str(tmp_path))
    assert Settings(_env_file=None).web_dist_path == tmp_path
```

Placer les deux premiers tests dans `test_web_ui.py` et le dernier dans
`test_config.py`.

- [ ] **Step 2: Vérifier l'échec**

Run: `rtk pytest tests/unit/test_web_ui.py tests/unit/test_config.py -q`

Expected: FAIL, réglage et montage absents.

- [ ] **Step 3: Implémenter le montage exact**

```python
_CSP = (
    "default-src 'self'; base-uri 'self'; form-action 'self'; "
    "connect-src 'self'; img-src 'self' data: blob:; "
    "media-src 'self' blob:; script-src 'self'; style-src 'self'; "
    "worker-src 'self' blob:; object-src 'none'; frame-ancestors 'none'"
)

def mount_web_ui(app: FastAPI, dist_path: Path) -> None:
    index = dist_path / "index.html"
    assets = dist_path / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="web-assets")

    @app.get("/", include_in_schema=False)
    async def web_index():
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Interface web non construite.")
        return FileResponse(index, headers={"Content-Security-Policy": _CSP})
```

Appeler `mount_web_ui()` après l'enregistrement des routeurs. Ne créer aucune
route `/{path:path}`. Le setting accepte `WEB_DIST_PATH` mais ne doit jamais
apparaître dans `/health`. Ajouter `WEB_DIST_PATH` à
`VARIABLES_DE_CONFIGURATION` dans `tests/conftest.py` pour préserver
l'isolation des tests.

- [ ] **Step 4: Vérifier backend et non-régression**

Run: `rtk pytest tests/unit/test_web_ui.py tests/unit/test_config.py tests/unit/test_native_routes.py tests/unit/test_openai_routes.py -q`

Expected: tests PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/transcription_server/api/web.py src/transcription_server/config.py src/transcription_server/app.py tests/conftest.py tests/unit/test_config.py tests/unit/test_web_ui.py
rtk git commit -m "feat: serve web interface from FastAPI"
```

### Task 12: Image Docker multistage et documentation utilisateur

**Files:**
- Modify: `docker/Dockerfile`
- Modify: `.dockerignore`
- Modify: `tests/integration/test_container_layout.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `web/package-lock.json`, scripts `test` et `build`.
- Produces: `/app/web-dist` dans l'étape finale, sans `node` ni `npm`.

- [ ] **Step 1: Écrire le test structurel rouge**

```python
def test_frontend_est_construit_dans_une_etape_node_ephemere():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM node:24-alpine AS web-builder" in dockerfile
    assert "RUN npm test" in dockerfile
    assert "RUN npm run build" in dockerfile
    assert "COPY --from=web-builder /web/dist /app/web-dist" in dockerfile
    final_stage = dockerfile.split("FROM pytorch/", 1)[1]
    assert "apt-get install -y nodejs" not in final_stage
    assert "apt-get install -y npm" not in final_stage
```

- [ ] **Step 2: Vérifier l'échec**

Run: `rtk pytest tests/integration/test_container_layout.py -q`

Expected: FAIL, stage Node absent.

- [ ] **Step 3: Ajouter l'étape Vite**

Placer avant l'étape PyTorch :

```dockerfile
FROM node:24-alpine AS web-builder
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm test
RUN npm run build
```

Puis, dans l'étape finale après `COPY src/ ./src/` :

```dockerfile
COPY --from=web-builder /web/dist /app/web-dist
```

Vérifier que `.dockerignore` conserve `web/src`, `web/index.html`, les configs
et le lock, mais ignore `web/node_modules`, `web/dist` et `web/coverage`.

- [ ] **Step 4: Documenter usage et développement**

Dans `README.md`, ajouter :

- ouverture de `http://127.0.0.1:8000/` ;
- description des cinq espaces et de la permission micro ;
- caractère local de l'historique et action de purge ;
- commande développement `cd web`, `npm ci`, `npm run dev` avec API sur 8000 ;
- commandes `npm test`, `npm run typecheck`, `npm run build` ;
- rappel que l'exposition réseau nécessite authentification et TLS.

- [ ] **Step 5: Vérifier build frontend et structure**

Run: `rtk npm test`

Run: `rtk npm run typecheck`

Run: `rtk npm run build`

Run from repository root: `rtk pytest tests/integration/test_container_layout.py -q`

Expected: toutes les commandes exit 0.

- [ ] **Step 6: Commit**

```bash
rtk git add docker/Dockerfile .dockerignore tests/integration/test_container_layout.py README.md
rtk git commit -m "feat: package web UI in Docker image"
```

### Task 13: Polissage accessible et validation de bout en bout

**Files:**
- Modify: `web/src/styles/app.css`
- Create: `web/src/styles/forms.css`
- Create: `web/src/styles/media.css`
- Test: `web/src/app/accessibility.test.tsx`
- Modify: `docs/superpowers/specs/2026-08-27-web-interface-design.md` only if a verified implementation constraint requires clarification.

**Interfaces:**
- Produces: interface finale à 360, 720 et 1440 px avec parcours clavier complet.

- [ ] **Step 1: Ajouter les assertions d'accessibilité stables**

Tester pour chaque route : un seul `<h1>`, navigation nommée, libellé pour
chaque contrôle, focus envoyé au titre après navigation, région `aria-live`
pour état long et aucune action icon-only sans nom. Tester aussi que la case de
consentement est décochée après remontage.

- [ ] **Step 2: Vérifier les échecs avant polissage**

Run: `rtk npm test -- --run src/app/accessibility.test.tsx`

Expected: au moins l'envoi du focus et les noms accessibles incomplets échouent.

- [ ] **Step 3: Finaliser le système visuel validé**

Appliquer strictement les tokens de la spec. Conserver une seule action
primaire pétrole par écran, réserver le rouge à l'enregistrement/destruction,
limiter les ombres à `0 8px 24px rgb(23 36 40 / 8%)`, transformer listes et
tableaux en cartes sous 720 px et fixer la barre mobile sans masquer le dernier
contrôle. Les durées et métriques utilisent la pile monospace ; les textes de
travail utilisent la pile corps.

La forme d'onde est le seul mouvement continu. Sous
`prefers-reduced-motion: reduce`, la remplacer par un niveau instantané sans
transition. Ajouter `:focus-visible` sur liens, boutons, champs et dialogue.

- [ ] **Step 4: Exécuter toutes les suites avant Docker**

Run: `rtk npm test`

Expected: frontend PASS.

Run: `rtk npm run typecheck`

Expected: exit 0.

Run from repository root: `rtk pytest -q`

Expected: suite Python PASS, GPU désélectionné par configuration.

- [ ] **Step 5: Construire et vérifier l'image finale**

Run: `rtk docker compose build`

Run: `rtk docker compose up -d --force-recreate`

Run: `rtk curl --retry 40 --retry-delay 3 --retry-connrefused http://127.0.0.1:8000/health`

Run: `rtk curl http://127.0.0.1:8000/`

Run: `rtk docker exec resume_transcription-transcription-1 sh -lc "! command -v node && ! command -v npm"`

Expected: image construite, conteneur healthy, `/` contient le point de montage
Preact, `node` et `npm` absents.

- [ ] **Step 6: Vérifier les parcours réels dans le navigateur**

À 1440 px puis 360 px :

1. importer et enregistrer un audio, transcrire avec diarisation ;
2. exporter SRT sans second appel `/transcribe` ;
3. résumer l'entrée depuis l'historique ;
4. générer CustomVoice et conserver explicitement le blob ;
5. inscrire une référence consentie de 3 à 30 secondes, l'utiliser puis la supprimer ;
6. effectuer un clone ponctuel et vérifier que la référence n'apparaît pas dans IndexedDB ;
7. recharger la page, vérifier l'historique puis le purger ;
8. parcourir les cinq écrans au clavier et avec mouvement réduit.

Inspecter Network pour confirmer zéro requête CDN/télémétrie et vérifier que
`/docs`, `/openapi.json` et un 404 `/v1/inconnue` restent corrects.

- [ ] **Step 7: Commit final de polissage**

```bash
rtk git add web/src/styles web/src/app/accessibility.test.tsx docs/superpowers/specs/2026-08-27-web-interface-design.md
rtk git commit -m "test: validate accessible web interface"
```

### Task 14: Vérification, intégration et livraison

**Files:**
- Modify: aucun fichier attendu ; tout correctif découvert suit un cycle rouge/vert et un commit dédié.

**Interfaces:**
- Produces: `main` propre, image saine et documentation cohérente.

- [ ] **Step 1: Vérification finale reproductible**

Run from `web/`: `rtk npm test`

Run from `web/`: `rtk npm run typecheck`

Run from `web/`: `rtk npm run build`

Run: `rtk pytest -q` from repository root.

Run: `rtk proxy git diff --check`

Run: `rtk docker compose config -q`

Expected: toutes les commandes exit 0.

- [ ] **Step 2: Contrôler sécurité et distribution**

Vérifier avec les DevTools qu'aucun token, chemin `/app/voices`, texte de
référence ou média source n'est présent dans console, Network ou IndexedDB.
Vérifier avec `rtk docker port resume_transcription-transcription-1` que seul
`127.0.0.1:8000` est publié et avec `rtk docker compose ps` que le service est
healthy.

- [ ] **Step 3: Finaliser la branche selon la consigne utilisateur**

Utiliser `superpowers:verification-before-completion`, puis
`superpowers:finishing-a-development-branch`. Si l'exécution a utilisé une
branche temporaire, fusionner sur `main`, relancer les suites sur le résultat,
supprimer le worktree et la branche temporaire, puis pousser `main`. Ne laisser
aucune branche de travail locale ou distante.
