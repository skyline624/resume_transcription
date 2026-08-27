import { render } from "preact";

import { HttpClient } from "./api/http";
import { App } from "./app/App";
import { ServicesProvider } from "./app/services";
import { BrowserRecorder } from "./media/recorder";
import { BrowserHistoryRepository } from "./storage/browser-history";
import "./styles/app.css";

const root = document.getElementById("app");
if (!root) throw new Error("Point de montage #app introuvable.");
render(
  <ServicesProvider
    services={{
      http: new HttpClient(),
      history: new BrowserHistoryRepository(),
      recorderFactory: () => new BrowserRecorder(),
    }}
  >
    <App />
  </ServicesProvider>,
  root,
);
