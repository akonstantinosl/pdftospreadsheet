// Application state
let appState = {
    selectedFile: null,
    selectedFormat: 'xlsx',
    isProcessing: false,
    pythonReady: false,
    lastResult: null
};

// DOM elements
const elements = {
    fileUploadArea: document.getElementById('fileUploadArea'),
    selectFileBtn: document.getElementById('selectFileBtn'),
    selectedFileInfo: document.getElementById('selectedFileInfo'),
    fileName: document.getElementById('fileName'),
    removeFileBtn: document.getElementById('removeFileBtn'),
    convertBtn: document.getElementById('convertBtn'),
    progressSection: document.getElementById('progressSection'),
    progressFill: document.getElementById('progressFill'),
    progressText: document.getElementById('progressText'),
    resultsSection: document.getElementById('resultsSection'),
    resultsInfo: document.getElementById('resultsInfo'),
    downloadBtn: document.getElementById('downloadBtn'),
    errorSection: document.getElementById('errorSection'),
    errorMessage: document.getElementById('errorMessage'),
    retryBtn: document.getElementById('retryBtn'),
    statusIndicator: document.getElementById('statusIndicator'),
    statusDot: document.querySelector('.status-dot'),
    statusText: document.querySelector('.status-text')
};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    initializePythonCommunication();
    updateUI();
});

// Initialize event listeners
function initializeEventListeners() {
    // File selection
    elements.selectFileBtn.addEventListener('click', selectFile);
    elements.fileUploadArea.addEventListener('click', selectFile);
    elements.removeFileBtn.addEventListener('click', removeFile);
    
    // Drag and drop
    elements.fileUploadArea.addEventListener('dragover', handleDragOver);
    elements.fileUploadArea.addEventListener('dragleave', handleDragLeave);
    elements.fileUploadArea.addEventListener('drop', handleDrop);
    
    // Format selection
    document.querySelectorAll('input[name="format"]').forEach(radio => {
        radio.addEventListener('change', handleFormatChange);
    });
    
    // Convert button
    elements.convertBtn.addEventListener('click', startConversion);
    
    // Download button
    elements.downloadBtn.addEventListener('click', downloadResult);
    
    // Retry button
    elements.retryBtn.addEventListener('click', retryConversion);
}

// Initialize Python backend communication
function initializePythonCommunication() {
    // Listen for Python process logs
    window.electronAPI.onPythonLog((event, data) => {
        console.log('Python log:', data);
        
        if (data.includes('Running on')) {
            updatePythonStatus('connected', 'Ready');
            appState.pythonReady = true;
            updateUI();
        }
    });
    
    // Listen for Python process errors
    window.electronAPI.onPythonError((event, data) => {
        console.error('Python error:', data);
        updatePythonStatus('error', 'Error');
        appState.pythonReady = false;
        updateUI();
    });
    
    // Listen for Python process closed
    window.electronAPI.onPythonClosed((event, code) => {
        console.log('Python process closed with code:', code);
        updatePythonStatus('error', 'Disconnected');
        appState.pythonReady = false;
        updateUI();
    });
    
    // Check backend status periodically
    checkBackendStatus();
    setInterval(checkBackendStatus, 5000);
}

// Check if backend is ready
async function checkBackendStatus() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);
        
        const response = await fetch('http://localhost:5000/health', {
            method: 'GET',
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
            const data = await response.json();
            updatePythonStatus('connected', 'Ready');
            appState.pythonReady = true;
        } else {
            updatePythonStatus('error', 'Error');
            appState.pythonReady = false;
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            updatePythonStatus('error', 'Timeout');
        } else {
            updatePythonStatus('error', 'Connecting...');
        }
        appState.pythonReady = false;
    }
    
    updateUI();
}

// Update Python status indicator
function updatePythonStatus(status, message) {
    elements.statusDot.className = `status-dot ${status}`;
    elements.statusText.textContent = message;
}

// File selection handlers
async function selectFile() {
    if (appState.isProcessing) return;
    
    try {
        const filePath = await window.electronAPI.selectFile();
        if (filePath) {
            appState.selectedFile = filePath;
            updateUI();
        }
    } catch (error) {
        showError('Failed to select file: ' + error.message);
    }
}

function removeFile() {
    if (appState.isProcessing) return;
    
    appState.selectedFile = null;
    hideResults();
    hideError();
    updateUI();
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    elements.fileUploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.fileUploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.fileUploadArea.classList.remove('dragover');
    
    if (appState.isProcessing) return;
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        const allowedExtensions = ['pdf', 'jpg', 'jpeg', 'png'];
        const fileExtension = file.name.split('.').pop().toLowerCase();
        
        if (allowedExtensions.includes(fileExtension)) {
            appState.selectedFile = file.path;
            updateUI();
        } else {
            showError('File type not supported. Please select PDF, JPG, JPEG, or PNG files.');
        }
    }
}

// Format selection handler
function handleFormatChange(e) {
    appState.selectedFormat = e.target.value;
    updateUI();
}

// Conversion process
async function startConversion() {
    if (!appState.selectedFile || !appState.pythonReady || appState.isProcessing) {
        return;
    }
    
    appState.isProcessing = true;
    hideResults();
    hideError();
    showProgress();
    updateUI();
    
    try {
        // Create FormData for file upload
        const formData = new FormData();
        
        // Update progress
        updateProgress(10, 'Reading file...');
        
        // Read file using electron API
        const fileData = await window.electronAPI.readFile(appState.selectedFile);
        const fileBlob = new Blob([fileData]);
        
        formData.append('file', fileBlob, getFileName(appState.selectedFile));
        formData.append('format', appState.selectedFormat);
        
        // Update progress
        updateProgress(30, 'Uploading file...');
        
        // Send conversion request
        const response = await fetch('http://localhost:5000/convert', {
            method: 'POST',
            body: formData
        });
        
        updateProgress(60, 'Processing file...');
        
        if (!response.ok) {
            let errorMessage = 'Conversion failed';
            try {
                const errorData = await response.json();
                errorMessage = errorData.error || errorMessage;
            } catch (e) {
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        updateProgress(80, 'Generating output...');
        
        // Get result
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.message || 'Conversion failed');
        }
        
        updateProgress(100, 'Complete!');
        
        // Store result and show success
        appState.lastResult = result;
        showResults(result);
        
    } catch (error) {
        console.error('Conversion error:', error);
        
        // Provide more specific error messages
        let errorMessage = error.message;
        if (error.message.includes('fetch')) {
            errorMessage = 'Cannot connect to backend server. Please make sure the Python backend is running.';
        } else if (error.message.includes('Failed to read file')) {
            errorMessage = 'Cannot read the selected file. Please check file permissions and try again.';
        }
        
        showError(errorMessage);
    } finally {
        appState.isProcessing = false;
        hideProgress();
        updateUI();
    }
}

// Download result
async function downloadResult() {
    if (!appState.lastResult) return;
    
    try {
        const result = appState.lastResult;
        const originalName = getFileName(appState.selectedFile, false);
        const extension = getFileExtension(appState.selectedFormat);
        const defaultName = `${originalName}.${extension}`;
        
        const filters = getFileFilters(appState.selectedFormat);
        const savePath = await window.electronAPI.saveFile(defaultName, filters);
        
        if (savePath) {
            // Write file
            const success = await window.electronAPI.writeFile(
                savePath, 
                result.file_data, 
                'base64'
            );
            
            if (success) {
                showSuccess('File saved successfully!');
            } else {
                showError('Failed to save file');
            }
        }
    } catch (error) {
        showError('Download failed: ' + error.message);
    }
}

// Retry conversion
function retryConversion() {
    hideError();
    if (appState.selectedFile && appState.pythonReady) {
        startConversion();
    }
}

// UI update functions
function updateUI() {
    // Update file selection UI
    if (appState.selectedFile) {
        elements.fileUploadArea.style.display = 'none';
        elements.selectedFileInfo.style.display = 'block';
        elements.fileName.textContent = getFileName(appState.selectedFile);
    } else {
        elements.fileUploadArea.style.display = 'block';
        elements.selectedFileInfo.style.display = 'none';
    }
    
    // Update convert button
    const canConvert = appState.selectedFile && appState.pythonReady && !appState.isProcessing;
    elements.convertBtn.disabled = !canConvert;
    
    if (appState.isProcessing) {
        elements.convertBtn.querySelector('.btn-text').style.display = 'none';
        elements.convertBtn.querySelector('.btn-loading').style.display = 'flex';
    } else {
        elements.convertBtn.querySelector('.btn-text').style.display = 'inline';
        elements.convertBtn.querySelector('.btn-loading').style.display = 'none';
    }
}

// Progress functions
function showProgress() {
    elements.progressSection.style.display = 'block';
    updateProgress(0, 'Starting conversion...');
}

function updateProgress(percentage, message) {
    elements.progressFill.style.width = percentage + '%';
    elements.progressText.textContent = message;
}

function hideProgress() {
    elements.progressSection.style.display = 'none';
}

// Results functions
function showResults(result) {
    elements.resultsSection.style.display = 'block';
    
    const tableCount = result.table_count || 0;
    const message = tableCount > 0 
        ? `Successfully extracted ${tableCount} table${tableCount > 1 ? 's' : ''} from the file.`
        : 'No tables were detected in the file.';
    
    elements.resultsInfo.textContent = message;
    elements.downloadBtn.style.display = tableCount > 0 ? 'inline-flex' : 'none';
}

function hideResults() {
    elements.resultsSection.style.display = 'none';
}

// Error functions
function showError(message) {
    elements.errorSection.style.display = 'block';
    elements.errorMessage.textContent = message;
}

function hideError() {
    elements.errorSection.style.display = 'none';
}

// Success notification
function showSuccess(message) {
    // Create temporary success notification
    const notification = document.createElement('div');
    notification.className = 'success-notification';
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #28a745;
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        z-index: 1001;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Utility functions
function getFileName(filePath, withExtension = true) {
    if (!filePath) return '';
    
    const parts = filePath.split(/[/\\]/);
    const fileName = parts[parts.length - 1];
    
    if (withExtension) {
        return fileName;
    } else {
        const dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 ? fileName.substring(0, dotIndex) : fileName;
    }
}

function getFileExtension(format) {
    const extensions = {
        'xlsx': 'xlsx',
        'csv': 'csv',
        'ods': 'ods',
        'gsheet': 'xlsx'
    };
    return extensions[format] || 'xlsx';
}

function getFileFilters(format) {
    const filters = {
        'xlsx': [{ name: 'Excel Files', extensions: ['xlsx'] }],
        'csv': [{ name: 'CSV Files', extensions: ['csv'] }],
        'ods': [{ name: 'OpenDocument Spreadsheet', extensions: ['ods'] }],
        'gsheet': [{ name: 'Excel Files', extensions: ['xlsx'] }]
    };
    return filters[format] || filters['xlsx'];
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);