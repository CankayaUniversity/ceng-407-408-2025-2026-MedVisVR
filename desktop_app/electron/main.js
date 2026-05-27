const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const fs = require("fs");
const path = require("path");

let mainWindow = null;
let apiProcess = null;
let rendererLogPath = null;

function repoRoot() {
  if (app.isPackaged) {
    const root = path.resolve(path.dirname(process.execPath), "..", "..", "..");
    fs.appendFileSync(
      path.join(path.dirname(process.execPath), "lumina_root.log"),
      `[${new Date().toISOString()}] execPath=${process.execPath}\nrepoRoot=${root}\n\n`
    );
    return root;
  }
  return path.resolve(__dirname, "..", "..");
}

function apiPythonPath() {
  return path.join(repoRoot(), ".venv", "Scripts", "python.exe");
}

function preferredRunsRoot() {
  const candidate = path.join(repoRoot(), "outputs", "runs_first10_reportllm_relaxed_v1");
  if (fs.existsSync(candidate)) return candidate;
  return path.join(repoRoot(), "outputs", "runs");
}

function appendRendererLog(message) {
  if (!rendererLogPath) return;
  fs.appendFileSync(rendererLogPath, `${new Date().toISOString()} ${message}\n`);
}

function isApiReachable() {
  return new Promise((resolve) => {
    const req = http.get("http://127.0.0.1:8000/health", (res) => {
      resolve(res.statusCode === 200);
      res.resume();
    });
    req.on("error", () => resolve(false));
    req.setTimeout(1000, () => { req.destroy(); resolve(false); });
  });
}

function startBackendIfAvailable() {
  const pythonExe = apiPythonPath();
  if (!fs.existsSync(pythonExe)) return;

  const logsDir = path.join(repoRoot(), "tmp");
  if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });

  const out = fs.openSync(path.join(logsDir, "desktop_api_stdout.log"), "a");
  const err = fs.openSync(path.join(logsDir, "desktop_api_stderr.log"), "a");

  apiProcess = spawn(
    pythonExe,
    ["-m", "uvicorn", "ai_assistant.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
    {
      cwd: repoRoot(),
      windowsHide: true,
      env: {
        ...process.env,
        AI_CONTEXTS_ROOT: path.join(repoRoot(), "outputs", "cases"),
        AI_RUNS_ROOT: preferredRunsRoot(),
        AI_WORKSPACE_ROOT: repoRoot(),
        ...(process.env.NNUNET_V1_ENV ? {} : (() => {
          const bundled = path.join(repoRoot(), "nnunet_v1_env");
          const sibling = path.join(repoRoot(), "..", "nnunet_v1_env");
          if (fs.existsSync(bundled)) return { NNUNET_V1_ENV: bundled };
          if (fs.existsSync(sibling)) return { NNUNET_V1_ENV: sibling };
          return {};
        })()),
      },
      stdio: ["ignore", out, err]
    }
  );
  apiProcess.on("exit", () => { apiProcess = null; });
}

function resolveIcon() {
  const candidates = [
    path.join(__dirname, "..", "build", "lumina.ico"),
    path.join(repoRoot(), "desktop_app", "build", "lumina.ico"),
    path.join(path.dirname(process.execPath), "resources", "build", "lumina.ico"),
    path.join(__dirname, "..", "assets", "icon.ico"),
    path.join(repoRoot(), "desktop_app", "assets", "icon.ico"),
  ];
  return candidates.find(p => fs.existsSync(p)) || undefined;
}

let splashWindow = null;

function createSplash() {
  const splashPath = path.join(__dirname, "splash.html");
  if (!fs.existsSync(splashPath)) return null;

  splashWindow = new BrowserWindow({
    width: 520,
    height: 340,
    frame: false,
    transparent: false,
    resizable: false,
    alwaysOnTop: true,
    center: true,
    backgroundColor: "#060a10",
    icon: resolveIcon(),
    webPreferences: { contextIsolation: true, nodeIntegration: false }
  });
  splashWindow.loadFile(splashPath);
  splashWindow.on("closed", () => { splashWindow = null; });
  return splashWindow;
}

function createWindow() {
  const logsDir = path.join(repoRoot(), "tmp");
  if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });
  rendererLogPath = path.join(logsDir, "desktop_renderer.log");

  mainWindow = new BrowserWindow({
    show: false,
    width: 1560,
    height: 980,
    minWidth: 1360,
    minHeight: 860,
    backgroundColor: "#0b1220",
    title: "CARVIS — Clinical AI Platform",
    autoHideMenuBar: true,
    icon: resolveIcon(),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  mainWindow.webContents.on("did-fail-load", (_e, code, desc) => appendRendererLog(`did-fail-load code=${code} desc=${desc}`));
  mainWindow.webContents.on("console-message", (_e, level, message, line, sourceId) => appendRendererLog(`console level=${level} line=${line} source=${sourceId} message=${message}`));
  mainWindow.webContents.on("render-process-gone", (_e, details) => appendRendererLog(`render-process-gone reason=${details.reason} exitCode=${details.exitCode}`));

  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }

  const SPLASH_MIN_MS = 5000;
  const splashShownAt = Date.now();

  mainWindow.once("ready-to-show", () => {
    const elapsed = Date.now() - splashShownAt;
    const remaining = Math.max(0, SPLASH_MIN_MS - elapsed);
    setTimeout(() => {
      if (splashWindow && !splashWindow.isDestroyed()) splashWindow.close();
      mainWindow.show();
      mainWindow.focus();
    }, remaining);
  });
}

ipcMain.handle("save-pdf", async (_event, { html, defaultName }) => {
  const { filePath, canceled } = await dialog.showSaveDialog(mainWindow, {
    title: "Save Clinical Report as PDF",
    defaultPath: path.join(app.getPath("documents"), defaultName || "NeuroLens_Report.pdf"),
    filters: [{ name: "PDF Document", extensions: ["pdf"] }],
  });

  if (canceled || !filePath) return { success: false, reason: "canceled" };

  const pdfWin = new BrowserWindow({
    show: false,
    webPreferences: { contextIsolation: true, nodeIntegration: false }
  });

  await pdfWin.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);

  try {
    const pdfBuffer = await pdfWin.webContents.printToPDF({
      marginsType: 1,
      pageSize: "A4",
      printBackground: true,
      landscape: false,
    });
    fs.writeFileSync(filePath, pdfBuffer);
    pdfWin.destroy();
    shell.openPath(filePath);
    return { success: true, filePath };
  } catch (err) {
    pdfWin.destroy();
    return { success: false, reason: err.message };
  }
});

ipcMain.handle("print-pdf", async (_event, { html }) => {
  const pdfWin = new BrowserWindow({
    show: true,
    width: 900,
    height: 750,
    title: "Print Report",
    webPreferences: { contextIsolation: true, nodeIntegration: false }
  });
  await pdfWin.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
  pdfWin.webContents.once("did-finish-load", () => {
    setTimeout(() => pdfWin.webContents.print({}, () => {}), 500);
  });
  return { success: true };
});

ipcMain.handle("choose-nifti-file", async () => {
  const { filePaths, canceled } = await dialog.showOpenDialog(mainWindow, {
    title: "Select NIfTI File",
    filters: [
      { name: "NIfTI Files", extensions: ["nii", "gz"] },
      { name: "All Files",   extensions: ["*"] },
    ],
    properties: ["openFile"],
  });
  if (canceled || !filePaths.length) return null;
  return filePaths[0];
});

ipcMain.handle("choose-nifti-files", async () => {
  const { filePaths, canceled } = await dialog.showOpenDialog(mainWindow, {
    title: "Select BraTS NIfTI Files (FLAIR, T1, T1ce, T2)",
    filters: [
      { name: "NIfTI Files", extensions: ["nii", "gz"] },
      { name: "All Files",   extensions: ["*"] },
    ],
    properties: ["openFile", "multiSelections"],
  });
  if (canceled || !filePaths.length) return [];
  return filePaths;
});

app.whenReady().then(async () => {
  const { session } = require("electron");
  await session.defaultSession.clearCache();

  createSplash();

  const reachable = await isApiReachable();
  if (!reachable) startBackendIfAvailable();
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (apiProcess && !apiProcess.killed) apiProcess.kill();
});
