/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Clinical palette — calm, not eye-straining
        surface:  { DEFAULT: "#0d1625", 2: "#111e32", 3: "#162038" },
        border:   { DEFAULT: "#1e2f48", 2: "#243650" },
        accent:   { DEFAULT: "#2563eb", soft: "#1d4ed8" },
        success:  "#16a34a",
        danger:   "#dc2626",
        warn:     "#d97706",
        muted:    "#4b6380",
        text:     { DEFAULT: "#e2e8f0", 2: "#94a3b8", 3: "#64748b" },
        brain:    "#3b82f6",
        lung:     "#f97316",
        liver:    "#10b981",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        ui:   ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
