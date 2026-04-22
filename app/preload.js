// preload.js — puente seguro Electron ↔ React.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gestdeck', {
  // Control de ventanas
  openPresenter: () => ipcRenderer.invoke('app:open-presenter'),
  closePresenter: () => ipcRenderer.invoke('app:close-presenter'),
  getMode: () => ipcRenderer.invoke('app:get-mode'),

  // Diálogos
  openPptxDialog: () => ipcRenderer.invoke('session:open-pptx-dialog'),
  openConfigDialog: () => ipcRenderer.invoke('session:open-config-dialog'),
  saveDialog: (name) => ipcRenderer.invoke('session:save-dialog', name),

  // Conversión
  convertPptx: (pptxPath, sessionName) => ipcRenderer.invoke('session:convert-pptx', pptxPath, sessionName),
  listSlidesInDir: (relDir) => ipcRenderer.invoke('session:list-slides-in-dir', relDir),

  // Config.json
  readConfig: (absPath) => ipcRenderer.invoke('session:read-config', absPath),
  writeConfig: (absPath, contentStr) => ipcRenderer.invoke('session:write-config', absPath, contentStr),
  toGestdeckUrl: (absPath) => ipcRenderer.invoke('session:to-gestdeck-url', absPath),

  // Estado de slides compartido entre ventanas
  setActiveSlides: (state) => ipcRenderer.invoke('session:set-active-slides', state),
  getActiveSlides: () => ipcRenderer.invoke('session:get-active-slides'),
  onSlidesUpdated: (cb) => {
    const listener = (_evt, state) => cb(state);
    ipcRenderer.on('gestdeck:slides-updated', listener);
    return () => ipcRenderer.removeListener('gestdeck:slides-updated', listener);
  },

  // URL del WebSocket backend
  wsUrl: process.env.GESTDECK_WS || 'ws://127.0.0.1:8765',
});
