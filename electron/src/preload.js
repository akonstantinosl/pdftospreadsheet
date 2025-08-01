const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // File operations
  selectFile: () => ipcRenderer.invoke('select-file'),
  saveFile: (defaultName, filters) => ipcRenderer.invoke('save-file', defaultName, filters),
  readFile: (filePath) => ipcRenderer.invoke('read-file', filePath),
  writeFile: (filePath, data, encoding) => ipcRenderer.invoke('write-file', filePath, data, encoding),
  
  // Python process communication
  onPythonLog: (callback) => ipcRenderer.on('python-log', callback),
  onPythonError: (callback) => ipcRenderer.on('python-error', callback),
  onPythonClosed: (callback) => ipcRenderer.on('python-closed', callback),
  
  // Remove listeners
  removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});