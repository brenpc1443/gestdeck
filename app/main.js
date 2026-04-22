// main.js — Proceso principal de Electron
//
// Responsabilidades:
//   1. Crear las tres ventanas (Editor / Presenter / Control).
//   2. Registrar el protocolo custom "gestdeck://" que sirve archivos
//      desde la raíz del proyecto (para slides, SVGs y assets de sesión).
//   3. Exponer IPC para diálogos nativos y operaciones de filesystem:
//      abrir .pptx, cargar/guardar config.json de sesión, y disparar
//      la conversión .pptx → PNG mediante `python -m core.slide_reader`.

const { app, BrowserWindow, ipcMain, dialog, screen, Menu, protocol } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const isDev = !app.isPackaged;
const VITE_URL = 'http://localhost:5173';

// Raíz del proyecto GestDeck — app/ está un nivel por debajo.
const PROJECT_ROOT = path.resolve(__dirname, '..');

let editorWindow = null;
let presenterWindow = null;
let controlWindow = null;

// ----- Protocolo custom "gestdeck://" ---------------------------------
// Permite al renderer cargar cualquier archivo dentro del proyecto
// (slides PNG, SVGs, imágenes de objetos, PPTX exportados) sin violar
// web-security. Uso:
//   <img src="gestdeck://sessions/demo/slides/slide-001.png" />
//   <img src="gestdeck://assets/objects/arrow.svg" />
protocol.registerSchemesAsPrivileged([
  { scheme: 'gestdeck', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } },
]);

function registerGestdeckProtocol() {
  protocol.registerFileProtocol('gestdeck', (request, callback) => {
    try {
      // gestdeck://something/here → <ROOT>/something/here
      const url = request.url.replace(/^gestdeck:\/\//, '');
      const decoded = decodeURI(url);
      const resolved = path.normalize(path.join(PROJECT_ROOT, decoded));
      // Seguridad: evita path traversal fuera del proyecto
      if (!resolved.startsWith(PROJECT_ROOT)) {
        return callback({ error: -10 /* ACCESS_DENIED */ });
      }
      callback({ path: resolved });
    } catch (e) {
      console.error('[gestdeck://] error:', e);
      callback({ error: -2 });
    }
  });
}

// ----- Ventanas --------------------------------------------------------
function loadView(win, route) {
  if (isDev) {
    win.loadURL(`${VITE_URL}/#${route}`);
  } else {
    win.loadFile(path.join(__dirname, 'dist/index.html'), { hash: route });
  }
}

function createEditor() {
  editorWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    title: 'GestDeck — Editor',
    backgroundColor: '#0b0f1a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  loadView(editorWindow, '/editor');
  editorWindow.on('closed', () => { editorWindow = null; });
}

function createPresenter() {
  const displays = screen.getAllDisplays();
  const target = displays.length > 1 ? displays[1] : displays[0];
  const { x, y, width, height } = target.bounds;
  presenterWindow = new BrowserWindow({
    x, y, width, height,
    title: 'GestDeck — Presentador',
    fullscreen: true,
    autoHideMenuBar: true,
    backgroundColor: '#000000',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  loadView(presenterWindow, '/presenter');
  presenterWindow.on('closed', () => { presenterWindow = null; });
}

function createControl() {
  controlWindow = new BrowserWindow({
    width: 520,
    height: 760,
    title: 'GestDeck — Control',
    alwaysOnTop: true,
    backgroundColor: '#0b0f1a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  loadView(controlWindow, '/control');
  controlWindow.on('closed', () => { controlWindow = null; });
}

// ----- IPC: ventanas ---------------------------------------------------
ipcMain.handle('app:open-presenter', () => {
  if (!presenterWindow) createPresenter();
  if (!controlWindow) createControl();
  return true;
});

ipcMain.handle('app:close-presenter', () => {
  if (presenterWindow) presenterWindow.close();
  if (controlWindow) controlWindow.close();
  return true;
});

ipcMain.handle('app:get-mode', () => ({ mode: 'ghost', projectRoot: PROJECT_ROOT }));

// ----- IPC: diálogos de archivo ---------------------------------------
ipcMain.handle('session:open-pptx-dialog', async () => {
  const res = await dialog.showOpenDialog(editorWindow, {
    title: 'Selecciona tu .pptx',
    filters: [{ name: 'PowerPoint', extensions: ['pptx', 'ppt'] }],
    properties: ['openFile'],
  });
  if (res.canceled || !res.filePaths.length) return null;
  return res.filePaths[0];
});

ipcMain.handle('session:open-config-dialog', async () => {
  const res = await dialog.showOpenDialog(editorWindow, {
    title: 'Abrir sesión GestDeck',
    filters: [{ name: 'Sesión', extensions: ['json'] }],
    properties: ['openFile'],
    defaultPath: path.join(PROJECT_ROOT, 'sessions'),
  });
  if (res.canceled || !res.filePaths.length) return null;
  return res.filePaths[0];
});

ipcMain.handle('session:save-dialog', async (_evt, suggestedName = 'nueva_sesion') => {
  const res = await dialog.showSaveDialog(editorWindow, {
    title: 'Guardar sesión GestDeck',
    defaultPath: path.join(PROJECT_ROOT, 'sessions', suggestedName, 'config.json'),
    filters: [{ name: 'Sesión', extensions: ['json'] }],
  });
  if (res.canceled || !res.filePath) return null;
  return res.filePath;
});

// ----- IPC: conversión PPTX → PNG -------------------------------------
ipcMain.handle('session:convert-pptx', async (_evt, pptxPath, sessionName) => {
  return new Promise((resolve, reject) => {
    const sessionDir = path.join(PROJECT_ROOT, 'sessions', sessionName);
    const slidesDir = path.join(sessionDir, 'slides');
    fs.mkdirSync(slidesDir, { recursive: true });

    const pyCmd = process.platform === 'win32'
      ? path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')
      : path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
    const pyFallback = process.platform === 'win32' ? 'python.exe' : 'python3';
    const pyExe = fs.existsSync(pyCmd) ? pyCmd : pyFallback;

    const args = ['-m', 'core.slide_reader', pptxPath, slidesDir];
    const child = spawn(pyExe, args, { cwd: PROJECT_ROOT });

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code !== 0) {
        console.error('[convert-pptx] stderr:', stderr);
        return reject(new Error(`slide_reader exit ${code}: ${stderr.slice(0, 200)}`));
      }
      // Lista de PNGs generados (relativa al project root → URL gestdeck://)
      const files = fs.readdirSync(slidesDir)
        .filter((f) => /\.(png|jpg|jpeg)$/i.test(f))
        .sort();
      const urls = files.map((f) => `gestdeck://sessions/${sessionName}/slides/${f}`);
      resolve({ urls, sessionDir: `sessions/${sessionName}`, count: urls.length, stdout });
    });
  });
});

// ----- IPC: leer/escribir config.json --------------------------------
ipcMain.handle('session:read-config', async (_evt, configPath) => {
  if (!configPath || !fs.existsSync(configPath)) return null;
  return fs.readFileSync(configPath, 'utf-8');
});

ipcMain.handle('session:write-config', async (_evt, configPath, contentStr) => {
  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  fs.writeFileSync(configPath, contentStr, 'utf-8');
  return true;
});

ipcMain.handle('session:list-slides-in-dir', async (_evt, relDir) => {
  const abs = path.join(PROJECT_ROOT, relDir);
  if (!fs.existsSync(abs)) return [];
  return fs.readdirSync(abs)
    .filter((f) => /\.(png|jpg|jpeg)$/i.test(f))
    .sort()
    .map((f) => `gestdeck://${relDir.replace(/\\/g, '/')}/${f}`);
});

ipcMain.handle('session:to-gestdeck-url', async (_evt, absPath) => {
  const rel = path.relative(PROJECT_ROOT, absPath).replace(/\\/g, '/');
  return `gestdeck://${rel}`;
});

// ----- Estado de slides compartido entre ventanas ---------------------
// El Editor lo setea cuando carga/convierte una sesión; Presenter y Control
// lo consumen para pintar la imagen del slide actual como fondo.
let sharedSlidesState = {
  slideImages: [],      // array de URLs gestdeck:// (index = slide - 1)
  sessionName: null,
  slidesDir: null,
};

function broadcastSlidesState() {
  const payload = sharedSlidesState;
  for (const w of [editorWindow, presenterWindow, controlWindow]) {
    if (w && !w.isDestroyed()) {
      w.webContents.send('gestdeck:slides-updated', payload);
    }
  }
}

ipcMain.handle('session:set-active-slides', (_evt, state) => {
  sharedSlidesState = {
    slideImages: Array.isArray(state?.slideImages) ? state.slideImages : [],
    sessionName: state?.sessionName ?? null,
    slidesDir: state?.slidesDir ?? null,
  };
  broadcastSlidesState();
  return true;
});

ipcMain.handle('session:get-active-slides', () => sharedSlidesState);

// ----- Lifecycle -------------------------------------------------------
app.whenReady().then(() => {
  registerGestdeckProtocol();
  Menu.setApplicationMenu(null);
  createEditor();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createEditor();
});
