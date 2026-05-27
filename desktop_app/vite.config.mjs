import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs/promises";

// vtk.js ships raw GLSL shader files — Vite/Rollup can't parse them as JS.
// This plugin intercepts .glsl imports and re-exports the source as a string.
function vtkShaderLoader() {
  return {
    name: "vtk-shader-loader",
    async load(id) {
      if (!id.endsWith(".glsl")) return null;
      const source = await fs.readFile(id, "utf8");
      return `export default ${JSON.stringify(source)};`;
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [vtkShaderLoader(), react()],
  css: {
    postcss: "./postcss.config.js",
  },
  server: {
    port: 5173,
    strictPort: true
  }
});
