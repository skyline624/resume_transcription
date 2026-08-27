import { render } from "preact";

import { App } from "./app/App";
import "./styles/app.css";

const root = document.getElementById("app");
if (!root) throw new Error("Point de montage #app introuvable.");
render(<App />, root);
