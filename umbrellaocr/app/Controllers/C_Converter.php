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
        $fileName = $this->request->getPost('filename');
        $format = $this->request->getPost('format') ?? 'xlsx';
        
        if (!$fileName) {
            return $this->response->setJSON(['success' => false, 'message' => 'No file uploaded']);
        }
        
        try {
            $result = $this->callFlaskService($fileName, $format);
            
            if ($result['success']) {
                return $this->response->setJSON([
                    'success' => true,
                    'message' => 'Conversion completed successfully',
                    'output_filename' => $result['output_filename'], 
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
    
    public function download($outputFilename)
    {
        $originalFilename = $this->request->getGet('original') ?? 'converted';
        $outputFilePath = $this->sharedFolderPath . '/' . $outputFilename;

        if (!file_exists($outputFilePath)) {
            throw new \CodeIgniter\Exceptions\PageNotFoundException('Converted file not found on the share.');
        }

        if (!file_exists($outputFilePath)) {
             throw new \CodeIgniter\Exceptions\PageNotFoundException('Converted file not found on the share.');
        }

        // Read the processed file from the share
        $fileData = file_get_contents($outputFilePath);

        // Hapus file setelah dibaca
        if (file_exists($outputFilePath)) {
            unlink($outputFilePath);
        }

        // Tentukan nama file download
        $fileExtension = pathinfo($outputFilename, PATHINFO_EXTENSION);
        $downloadFilename = $originalFilename . '.' . $fileExtension;
        
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