const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const log = require('electron-log');
const fs = require('fs');

let mainWindow;
let pythonProcess;

// Configure logging
log.transports.file.level = 'info';
log.transports.console.level = 'debug';

// Create the main application window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, '../assets/logo.png'),
    show: false
  });

  // Load the HTML file
  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    // Start Python backend
    startPythonBackend();
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
    if (pythonProcess) {
      pythonProcess.kill();
    }
  });

  // Development tools
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }
}

// Start Python Flask backend
function startPythonBackend() {
  const isDev = process.argv.includes('--dev');
  let pythonPath;
  let scriptPath;

  if (isDev) {
    // Development mode
    pythonPath = 'python';
    scriptPath = path.join(__dirname, '../python/app.py');
  } else {
    // Production mode
    if (process.platform === 'win32') {
      pythonPath = path.join(process.resourcesPath, 'python', 'python.exe');
    } else {
      pythonPath = path.join(process.resourcesPath, 'python', 'python');
    }
    scriptPath = path.join(process.resourcesPath, 'python', 'app.py');
  }

  log.info(`Starting Python backend: ${pythonPath} ${scriptPath}`);

  pythonProcess = spawn(pythonPath, [scriptPath], {
    stdio: ['pipe', 'pipe', 'pipe']
  });

  pythonProcess.stdout.on('data', (data) => {
    log.info(`Python stdout: ${data}`);
    
    // Send status updates to renderer
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('python-log', data.toString());
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    log.error(`Python stderr: ${data}`);
    
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('python-error', data.toString());
    }
  });

  pythonProcess.on('close', (code) => {
    log.info(`Python process exited with code ${code}`);
    
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('python-closed', code);
    }
  });

  pythonProcess.on('error', (err) => {
    log.error('Failed to start Python process:', err);
    
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('python-error', err.message);
    }
  });
}

// IPC handlers
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'PDF Files', extensions: ['pdf'] },
      { name: 'Image Files', extensions: ['jpg', 'jpeg', 'png'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle('save-file', async (event, defaultName, filters) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: defaultName,
    filters: filters || [
      { name: 'Excel Files', extensions: ['xlsx'] },
      { name: 'CSV Files', extensions: ['csv'] },
      { name: 'ODS Files', extensions: ['ods'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });

  if (!result.canceled) {
    return result.filePath;
  }
  return null;
});

ipcMain.handle('read-file', async (event, filePath) => {
  try {
    return await fs.promises.readFile(filePath);
  } catch (error) {
    log.error('Error reading file:', error);
    throw new Error(`Failed to read file: ${error.message}`);
  }
});

ipcMain.handle('write-file', async (event, filePath, data, encoding = 'base64') => {
  try {
    if (encoding === 'base64') {
      const buffer = Buffer.from(data, 'base64');
      fs.writeFileSync(filePath, buffer);
    } else {
      fs.writeFileSync(filePath, data, encoding);
    }
    return true;
  } catch (error) {
    log.error('Error writing file:', error);
    return false;
  }
});

// App event handlers
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});

// Handle app termination
process.on('SIGINT', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  app.quit();
});

process.on('SIGTERM', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  app.quit();
});