<?php

namespace App\Controllers;

use CodeIgniter\Controller;
use CodeIgniter\HTTP\CLIRequest;
use CodeIgniter\HTTP\IncomingRequest;
use CodeIgniter\HTTP\RequestInterface;
use CodeIgniter\HTTP\ResponseInterface;
use Psr\Log\LoggerInterface;

class C_Converter extends BaseController
{
    protected $flaskUrl = 'http://localhost:5000'; // URL Flask service
    
    public function initController(RequestInterface $request, ResponseInterface $response, LoggerInterface $logger)
    {
        parent::initController($request, $response, $logger);
    }
    
    public function index()
    {
        $data = [
            'title' => 'PDF to Spreadsheet Converter',
            'maxFileSize' => ini_get('upload_max_filesize')
        ];
        
        return view('converter/index', $data);
    }
    
    public function upload()
    {
        $validationRule = [
            'file' => [
                'label' => 'File',
                'rules' => 'uploaded[file]|max_size[file,10240]|ext_in[file,pdf,jpg,jpeg,png]',
            ],
        ];
        
        if (!$this->validate($validationRule)) {
            return $this->response->setJSON([
                'success' => false,
                'message' => 'File validation failed',
                'errors' => $this->validator->getErrors()
            ]);
        }
        
        $file = $this->request->getFile('file');
        
        if ($file->isValid() && !$file->hasMoved()) {
            // Generate unique filename
            $fileName = uniqid() . '_' . $file->getName();
            $tempPath = sys_get_temp_dir() . '/' . $fileName;
            
            // Move file to temporary location
            if ($file->move(sys_get_temp_dir(), $fileName)) {
                session()->set('uploaded_file', $tempPath);
                session()->set('original_filename', pathinfo($file->getName(), PATHINFO_FILENAME));
                
                return $this->response->setJSON([
                    'success' => true,
                    'message' => 'File uploaded successfully',
                    'filename' => $file->getName(),
                    'filesize' => $file->getSize()
                ]);
            }
        }
        
        return $this->response->setJSON([
            'success' => false,
            'message' => 'Failed to upload file'
        ]);
    }
    
    public function process()
    {
        $filePath = session()->get('uploaded_file');
        $format = $this->request->getPost('format') ?? 'xlsx';
        
        if (!$filePath || !file_exists($filePath)) {
            return $this->response->setJSON([
                'success' => false,
                'message' => 'No file uploaded or file not found'
            ]);
        }
        
        try {
            // Call Flask service to process the file
            $result = $this->callFlaskService($filePath, $format);
            
            if ($result['success']) {
                // Store result in session for download
                $downloadId = uniqid();
                session()->set('download_' . $downloadId, $result['data']);
                session()->set('download_format_' . $downloadId, $format);
                
                // Clean up uploaded file
                if (file_exists($filePath)) {
                    unlink($filePath);
                }
                
                return $this->response->setJSON([
                    'success' => true,
                    'message' => 'Conversion completed successfully',
                    'download_id' => $downloadId,
                    'tables_count' => $result['tables_count']
                ]);
            } else {
                return $this->response->setJSON([
                    'success' => false,
                    'message' => $result['message']
                ]);
            }
        } catch (\Exception $e) {
            log_message('error', 'Flask service error: ' . $e->getMessage());
            return $this->response->setJSON([
                'success' => false,
                'message' => 'Processing failed: ' . $e->getMessage()
            ]);
        }
    }
    
    public function download($downloadId)
    {
        $data = session()->get('download_' . $downloadId);
        $format = session()->get('download_format_' . $downloadId);
        $originalFilename = session()->get('original_filename') ?? 'converted';
        
        if (!$data) {
            throw new \CodeIgniter\Exceptions\PageNotFoundException('Download not found');
        }
        
        // Clean up session data
        session()->remove('download_' . $downloadId);
        session()->remove('download_format_' . $downloadId);
        session()->remove('original_filename');
        
        // Set appropriate headers based on format
        $mimeTypes = [
            'xlsx' => 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'csv' => 'text/csv',
            'ods' => 'application/vnd.oasis.opendocument.spreadsheet'
        ];
        
        $extensions = [
            'xlsx' => 'xlsx',
            'csv' => 'csv',
            'ods' => 'ods'
        ];
        
        $mimeType = $mimeTypes[$format] ?? 'application/octet-stream';
        $extension = $extensions[$format] ?? 'xlsx';
        
        // For CSV with multiple tables, it will be a ZIP file
        if ($format === 'csv' && strpos($data, 'UEsDB') === 0) { // ZIP file signature in base64
            $mimeType = 'application/zip';
            $extension = 'zip';
        }
        
        $filename = $originalFilename . '.' . $extension;
        
        // Decode the base64 data
        $fileData = base64_decode($data);
        
        // Verify the decoded data is not empty
        if (empty($fileData)) {
            throw new \CodeIgniter\Exceptions\PageNotFoundException('File data is corrupted');
        }
        
        return $this->response
            ->setHeader('Content-Type', $mimeType)
            ->setHeader('Content-Disposition', 'attachment; filename="' . $filename . '"')
            ->setHeader('Content-Length', strlen($fileData))
            ->setHeader('Cache-Control', 'no-cache, must-revalidate')
            ->setHeader('Expires', 'Sat, 26 Jul 1997 05:00:00 GMT')
            ->setBody($fileData);
    }
    
    private function callFlaskService($filePath, $format)
    {
        // Read file content
        $fileContent = file_get_contents($filePath);
        $fileBase64 = base64_encode($fileContent);
        
        // Prepare data for Flask service
        $postData = [
            'file_content' => $fileBase64,
            'filename' => basename($filePath),
            'format' => $format
        ];
        
        // Initialize cURL
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $this->flaskUrl . '/process');
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($postData));
        curl_setopt($ch, CURLOPT_HTTPHEADER, [
            'Content-Type: application/json'
        ]);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, 120); // 2 minutes timeout
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);
        
        if ($error) {
            throw new \Exception('Flask service connection error: ' . $error);
        }
        
        if ($httpCode !== 200) {
            throw new \Exception('Flask service returned HTTP ' . $httpCode);
        }
        
        $result = json_decode($response, true);
        
        if (!$result) {
            throw new \Exception('Invalid response from Flask service');
        }
        
        return $result;
    }
}