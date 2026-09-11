import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: process.env.GITHUB_ACTIONS === "true"
    ? "/Explainable-AI-Driven-Machine-Learning-Approaches-for-Intrusion-Detection-/"
    : "/",
  plugins: [react()],
  build: {
    emptyOutDir: true,
    outDir: "pages-dist",
  },
});
