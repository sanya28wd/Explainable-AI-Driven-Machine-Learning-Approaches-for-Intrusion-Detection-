import { createRoot } from "react-dom/client";
import Page from "./app/page";
import "./app/globals.css";
import "./app/overrides.css";

const originalFetch = window.fetch.bind(window);

function pageFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  if (typeof input === "string" && input.startsWith("/research/")) {
    return originalFetch(`${import.meta.env.BASE_URL}${input.slice(1)}`, init);
  }
  return originalFetch(input, init);
}

window.fetch = pageFetch;

const container = document.getElementById("root");
if (!container) {
  throw new Error("The GitHub Pages root element is missing.");
}

createRoot(container).render(<Page />);
