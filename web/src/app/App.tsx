import { useEffect, useState } from "preact/hooks";

import { HealthBar } from "../features/health/HealthBar";
import { readRoute, routeHref, type RouteName } from "./routes";

const navigation: Array<{ route: RouteName; label: string; hint: string }> = [
  { route: "transcribe", label: "Transcrire", hint: "Audio vers texte" },
  { route: "summarize", label: "Résumer", hint: "Texte vers synthèse" },
  { route: "speech", label: "Synthétiser", hint: "Texte vers voix" },
  { route: "voices", label: "Voix", hint: "Références consenties" },
  { route: "history", label: "Historique", hint: "Résultats locaux" },
];

const introductions: Record<RouteName, string> = {
  transcribe: "Transformez un fichier ou une prise micro en texte exploitable.",
  summarize: "Rédigez un compte-rendu depuis un audio ou une transcription.",
  speech: "Créez une voix, ajustez-la et écoutez le résultat.",
  voices: "Gérez les voix prédéfinies et vos références consenties.",
  history: "Retrouvez les résultats conservés dans ce navigateur.",
};

export function App() {
  const [route, setRoute] = useState<RouteName>(() =>
    readRoute(window.location.hash),
  );

  useEffect(() => {
    const update = () => setRoute(readRoute(window.location.hash));
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);

  const active = navigation.find((item) => item.route === route) ?? navigation[0]!;

  return (
    <div class="app-shell">
      <header class="topbar">
        <a class="brand" href={routeHref("transcribe")} aria-label="Atelier audio — Accueil">
          <span class="brand__mark" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
            <i />
          </span>
          <span>
            <strong>Atelier audio</strong>
            <small>Parakeet + Qwen</small>
          </span>
        </a>
        <HealthBar />
      </header>

      <nav class="main-nav" aria-label="Navigation principale">
        {navigation.map((item) => (
          <a
            key={item.route}
            href={routeHref(item.route)}
            class={`nav-link${item.route === route ? " nav-link--active" : ""}`}
            aria-current={item.route === route ? "page" : undefined}
            aria-label={item.label}
            onClick={() => {
              window.location.hash = routeHref(item.route);
              setRoute(item.route);
            }}
          >
            <span>{item.label}</span>
            <small>{item.hint}</small>
          </a>
        ))}
      </nav>

      <main class="workspace">
        <div class="workspace__heading">
          <p class="eyebrow">Outil local</p>
          <h1 tabIndex={-1}>{active.label}</h1>
          <p>{introductions[route]}</p>
        </div>
        <section class="work-surface" aria-label={`Espace ${active.label}`}>
          <p class="empty-guidance">Choisissez une source pour commencer.</p>
        </section>
      </main>
    </div>
  );
}
