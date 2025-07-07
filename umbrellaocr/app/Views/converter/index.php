<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?= $title ?></title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        .center-title {
            text-align: center;
        }
        .step-number {
            display: inline-block;
            width: 30px;
            height: 30px;
            background-color: #0078ff;
            color: white;
            text-align: center;
            line-height: 30px;
            border-radius: 50%;
            margin-right: 10px;
            font-weight: bold;
        }
        .step-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .upload-area {
            border: 2px dashed #dee2e6;
            border-radius: 8px;
            padding: 40px 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .upload-area:hover {
            border-color: #0078ff;
            background-color: #f8f9fa;
        }
        .upload-area.dragover {
            border-color: #0078ff;
            background-color: #e3f2fd;
        }
        .file-info {
            display: none;
            background-color: #e8f5e8;
            border: 1px solid #4caf50;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
        }
        .progress-container {
            display: none;
            margin-top: 20px;
        }
        .spinner-border-sm {
            width: 1rem;
            height: 1rem;
        }
        .format-option {
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .format-option:hover {
            background-color: #f8f9fa;
        }
        .format-option.selected {
            background-color: #0078ff;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <!-- Header -->
        <div class="row mb-5">
            <div class="col-12">
                <h1 class="center-title"><?= $title ?></h1>
                <p class="center-title text-muted">Ekstrak tabel dari file PDF dan gambar ke berbagai format spreadsheet</p>
            </div>
        </div>

        <!-- Main Content -->
        <div class="row">
            <!-- Step 1: File Upload -->
            <div class="col-md-4 mb-4">
                <div class="step-header">
                    <span class="step-number">1</span> Pilih File
                </div>
                
                <div class="upload-area" id="uploadArea">
                    <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                    <p class="mb-2">Klik atau seret file ke sini</p>
                    <small class="text-muted">PDF, JPG, PNG (Max: <?= $maxFileSize ?>)</small>
                    <input type="file" id="fileInput" class="d-none" accept=".pdf,.jpg,.jpeg,.png">
                </div>
                
                <div class="file-info" id="fileInfo">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <i class="fas fa-file text-success me-2"></i>
                            <span id="fileName"></span>
                        </div>
                        <button type="button" class="btn btn-sm btn-outline-danger" id="removeFile">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <small class="text-muted" id="fileSize"></small>
                </div>
                
                <div class="progress-container" id="progressContainer">
                    <div class="progress">
                        <div class="progress-bar" id="progressBar" role="progressbar" style="width: 0%"></div>
                    </div>
                    <small class="text-muted mt-1" id="progressText">Mengupload...</small>
                </div>
            </div>

            <!-- Step 2: Format Selection -->
            <div class="col-md-4 mb-4">
                <div class="step-header">
                    <span class="step-number">2</span> Format Output
                </div>
                
                <div class="list-group" id="formatOptions">
                    <div class="list-group-item format-option selected" data-format="xlsx">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <i class="fas fa-file-excel text-success me-2"></i>
                                <strong>XLSX</strong>
                            </div>
                            <small>Excel Spreadsheet</small>
                        </div>
                    </div>
                    <div class="list-group-item format-option" data-format="csv">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <i class="fas fa-file-csv text-info me-2"></i>
                                <strong>CSV</strong>
                            </div>
                            <small>Comma Separated Values</small>
                        </div>
                    </div>
                    <div class="list-group-item format-option" data-format="ods">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <i class="fas fa-file-alt text-warning me-2"></i>
                                <strong>ODS</strong>
                            </div>
                            <small>OpenDocument Spreadsheet</small>
                        </div>
                    </div>

                </div>
                
                <small class="text-muted mt-2 d-block">Pilih format file output untuk hasil ekstraksi tabel</small>
            </div>

            <!-- Step 3: Convert -->
            <div class="col-md-4 mb-4">
                <div class="step-header">
                    <span class="step-number">3</span> Konversi
                </div>
                
                <button type="button" class="btn btn-primary btn-lg w-100 mb-3" id="convertBtn" disabled>
                    <span id="convertText">Konversi</span>
                    <span class="spinner-border spinner-border-sm ms-2 d-none" id="convertSpinner"></span>
                </button>
                
                <small class="text-muted d-block">Klik untuk memulai proses konversi</small>
                
                <!-- Download Section -->
                <div class="mt-4 d-none" id="downloadSection">
                    <div class="alert alert-success" role="alert">
                        <i class="fas fa-check-circle me-2"></i>
                        <span id="successMessage">Konversi berhasil!</span>
                    </div>
                    <button type="button" class="btn btn-success w-100" id="downloadBtn">
                        <i class="fas fa-download me-2"></i>
                        <span id="downloadText">Download</span>
                    </button>
                </div>
            </div>
        </div>

        <!-- Alert Messages -->
        <div class="row">
            <div class="col-12">
                <div class="alert alert-danger d-none" id="errorAlert">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <span id="errorMessage"></span>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const fileInfo = document.getElementById('fileInfo');
            const fileName = document.getElementById('fileName');
            const fileSize = document.getElementById('fileSize');
            const removeFile = document.getElementById('removeFile');
            const progressContainer = document.getElementById('progressContainer');
            const progressBar = document.getElementById('progressBar');
            const progressText = document.getElementById('progressText');
            const formatOptions = document.querySelectorAll('.format-option');
            const convertBtn = document.getElementById('convertBtn');
            const convertText = document.getElementById('convertText');
            const convertSpinner = document.getElementById('convertSpinner');
            const downloadSection = document.getElementById('downloadSection');
            const downloadBtn = document.getElementById('downloadBtn');
            const downloadText = document.getElementById('downloadText');
            const successMessage = document.getElementById('successMessage');
            const errorAlert = document.getElementById('errorAlert');
            const errorMessage = document.getElementById('errorMessage');

            let selectedFormat = 'xlsx';
            let uploadedFile = null;
            let downloadId = null;

            // Upload area events
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', handleDragOver);
            uploadArea.addEventListener('dragleave', handleDragLeave);
            uploadArea.addEventListener('drop', handleDrop);
            fileInput.addEventListener('change', handleFileSelect);
            removeFile.addEventListener('click', resetUpload);

            // Format selection
            formatOptions.forEach(option => {
                option.addEventListener('click', function() {
                    formatOptions.forEach(opt => opt.classList.remove('selected'));
                    this.classList.add('selected');
                    selectedFormat = this.dataset.format;
                });
            });

            // Convert button
            convertBtn.addEventListener('click', processFile);
            downloadBtn.addEventListener('click', downloadFile);

            function handleDragOver(e) {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            }

            function handleDragLeave(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
            }

            function handleDrop(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            }

            function handleFileSelect(e) {
                const file = e.target.files[0];
                if (file) {
                    handleFile(file);
                }
            }

            function handleFile(file) {
                // Validate file type
                const allowedTypes = ['application/pdf', 'image/jpeg', 'image/jpg', 'image/png'];
                if (!allowedTypes.includes(file.type)) {
                    showError('Tipe file tidak didukung. Pilih file PDF, JPG, atau PNG.');
                    return;
                }

                // Validate file size (assuming max 10MB)
                if (file.size > 10 * 1024 * 1024) {
                    showError('Ukuran file terlalu besar. Maksimal 10MB.');
                    return;
                }

                uploadedFile = file;
                uploadFile(file);
            }

            function uploadFile(file) {
                const formData = new FormData();
                formData.append('file', file);

                showProgress(0);
                hideError();

                fetch('<?= base_url('converter/upload') ?>', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    hideProgress();
                    if (data.success) {
                        showFileInfo(file);
                        convertBtn.disabled = false;
                    } else {
                        showError(data.message || 'Upload gagal');
                    }
                })
                .catch(error => {
                    hideProgress();
                    showError('Error saat upload: ' + error.message);
                });
            }

            function showFileInfo(file) {
                fileName.textContent = file.name;
                fileSize.textContent = formatFileSize(file.size);
                fileInfo.style.display = 'block';
                uploadArea.style.display = 'none';
            }

            function resetUpload() {
                uploadedFile = null;
                fileInfo.style.display = 'none';
                uploadArea.style.display = 'block';
                convertBtn.disabled = true;
                hideDownloadSection();
                hideError();
                fileInput.value = '';
            }

            function showProgress(percent) {
                progressContainer.style.display = 'block';
                progressBar.style.width = percent + '%';
                progressText.textContent = percent < 100 ? 'Mengupload...' : 'Upload selesai';
            }

            function hideProgress() {
                progressContainer.style.display = 'none';
            }

            function processFile() {
                if (!uploadedFile) {
                    showError('Pilih file terlebih dahulu');
                    return;
                }

                setConvertingState(true);
                hideError();
                hideDownloadSection();

                const formData = new FormData();
                formData.append('format', selectedFormat);

                fetch('<?= base_url('converter/process') ?>', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    setConvertingState(false);
                    if (data.success) {
                        downloadId = data.download_id;
                        showDownloadSection(data.tables_count);
                    } else {
                        showError(data.message || 'Konversi gagal');
                    }
                })
                .catch(error => {
                    setConvertingState(false);
                    showError('Error saat konversi: ' + error.message);
                });
            }

            function downloadFile() {
                if (downloadId) {
                    window.location.href = '<?= base_url('converter/download/') ?>' + downloadId;
                }
            }

            function setConvertingState(converting) {
                convertBtn.disabled = converting;
                if (converting) {
                    convertText.textContent = 'Memproses...';
                    convertSpinner.classList.remove('d-none');
                } else {
                    convertText.textContent = 'Konversi';
                    convertSpinner.classList.add('d-none');
                }
            }

            function showDownloadSection(tablesCount) {
                successMessage.textContent = `Berhasil mengekstrak ${tablesCount} tabel!`;
                downloadText.textContent = `Download ${selectedFormat.toUpperCase()}`;
                downloadSection.classList.remove('d-none');
            }

            function hideDownloadSection() {
                downloadSection.classList.add('d-none');
            }

            function showError(message) {
                errorMessage.textContent = message;
                errorAlert.classList.remove('d-none');
                setTimeout(() => {
                    hideError();
                }, 5000);
            }

            function hideError() {
                errorAlert.classList.add('d-none');
            }

            function formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
        });
    </script>
</body>
</html>