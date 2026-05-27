# NeuroLens Clinical Workspace

Windows desktop shell for the local brain MRI decision-support prototype.

## Run

From `desktop_app/`:

```powershell
npm install
npm run app
```

The Electron process opens the desktop UI and attempts to start the local FastAPI server from:

```text
..\ .venv\Scripts\python.exe -m uvicorn ai_assistant.api.main:app --host 127.0.0.1 --port 8000
```

## Notes

- The app is English-only.
- Clinical mode is case-locked.
- Questions are disabled until a case is selected.
- Answers without evidence are promoted to the quality panel.
- Definitive diagnosis and treatment decisions are explicitly disallowed in the visible UX.
