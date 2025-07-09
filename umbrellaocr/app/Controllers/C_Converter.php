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
    protected $flaskUrl = 'http://192.19.27.105:5000';
    protected $sharedFolderPath = '/mnt/converter_share'; 
    
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
            
            if ($file->move($this->sharedFolderPath, $fileName)) {
                // Store the filename (not the full path) in the session for the next step
                session()->set('uploaded_filename', $fileName);
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
        $fileName = session()->get('uploaded_filename');
        $format = $this->request->getPost('format') ?? 'xlsx';
        
        if (!$fileName) {
            return $this->response->setJSON(['success' => false, 'message' => 'No file uploaded or session expired']);
        }

        // Verify the file exists in the share before processing
        if (!file_exists($this->sharedFolderPath . '/' . $fileName)) {
            return $this->response->setJSON(['success' => false, 'message' => 'Uploaded file not found in the shared directory.']);
        }
        
        try {
            // Call Flask service to process the file
            $result = $this->callFlaskService($fileName, $format);
            
            if ($result['success']) {
                // Store the result filename from Flask for download
                $downloadId = uniqid();
                session()->set('download_id_' . $downloadId, $result['output_filename']);
                session()->set('download_format_' . $downloadId, $format);
                // The original uploaded file is already deleted by Flask, so no cleanup needed here.
                
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
        $outputFilename = session()->get('download_id_' . $downloadId);
        $format = session()->get('download_format_' . $downloadId);
        $originalFilename = session()->get('original_filename') ?? 'converted';
        
        if (!$outputFilename) {
            throw new \CodeIgniter\Exceptions\PageNotFoundException('Download data not found or expired.');
        }

        $outputFilePath = $this->sharedFolderPath . '/' . $outputFilename;

        if (!file_exists($outputFilePath)) {
             throw new \CodeIgniter\Exceptions\PageNotFoundException('Converted file not found on the share.');
        }

        // Read the processed file from the share
        $fileData = file_get_contents($outputFilePath);
        
        // Clean up session data
        session()->remove('download_id_' . $downloadId);
        session()->remove('download_format_' . $downloadId);
        session()->remove('original_filename');
        
        // Clean up the processed file from the share after reading it
        if (file_exists($outputFilePath)) {
            unlink($outputFilePath);
        }
        
        // Set appropriate headers based on format
        $mimeTypes = [
            'xlsx' => 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'csv' => 'text/csv',
            'ods' => 'application/vnd.oasis.opendocument.spreadsheet',
            'zip' => 'application/zip' // Added for multi-table CSV
        ];
        
        $extensions = [
            'xlsx' => 'xlsx',
            'csv' => 'csv',
            'ods' => 'ods',
            'zip' => 'zip' // Added for multi-table CSV
        ];
        
        $fileExtension = pathinfo($outputFilename, PATHINFO_EXTENSION);
        $mimeType = $mimeTypes[$fileExtension] ?? 'application/octet-stream';
        $downloadFilename = $originalFilename . '.' . $fileExtension;
        
        if (empty($fileData)) {
            throw new \CodeIgniter\Exceptions\PageNotFoundException('File data is empty or corrupted');
        }
        
        return $this->response
            ->setHeader('Content-Type', $mimeType)
            ->setHeader('Content-Disposition', 'attachment; filename="' . $downloadFilename . '"')
            ->setHeader('Content-Length', strlen($fileData))
            ->setBody($fileData);
    }
    
    private function callFlaskService($fileName, $format)
    {
         $postData = [
            'filename' => $fileName,
            'format' => $format
        ];
        
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $this->flaskUrl . '/process');
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($postData));
        curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, 120);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);
        
        if ($error) {
            throw new \Exception('Flask service connection error: ' . $error);
        }
        
        if ($httpCode !== 200) {
            $errorBody = json_decode($response, true);
            $errorMessage = $errorBody['message'] ?? 'An unknown error occurred';
            throw new \Exception('Flask service returned HTTP ' . $httpCode . ': ' . $errorMessage);
        }
        
        $result = json_decode($response, true);
        
        if (!$result) {
            throw new \Exception('Invalid JSON response from Flask service');
        }
        
        return $result;
    }
}