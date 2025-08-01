from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import tempfile
import base64
import logging
from werkzeug.utils import secure_filename

# Import the OCR processing functions
from ocr_processor import (
    process_pdf, 
    process_image, 
    load_ocr_engine
)
from table_processor import create_tables_export

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extension(format_type):
    """Get file extension based on output format"""
    extensions = {
        'xlsx': 'xlsx',
        'csv': 'csv',
        'ods': 'ods',
        'gsheet': 'xlsx'
    }
    return extensions.get(format_type, 'xlsx')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'OCR Backend is running'
    }), 200

@app.route('/convert', methods=['POST'])
def convert_file():
    """Main conversion endpoint"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        format_type = request.form.get('format', 'xlsx').lower()
        
        # Validate file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
        # Validate format
        if format_type not in ['xlsx', 'csv', 'ods', 'gsheet']:
            return jsonify({'error': 'Output format not supported'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        logger.info(f"Processing file: {filename}, format: {format_type}")
        
        try:
            # Process file based on type
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            if file_extension == 'pdf':
                tables, sheet_names = process_pdf(file_path)
            else:
                tables = process_image(file_path)
                sheet_names = [f"Table_{i+1}" for i in range(len(tables))]
            
            logger.info(f"Extracted {len(tables)} tables")
            
            if not tables:
                return jsonify({
                    'success': True,
                    'message': 'No tables detected in the file',
                    'table_count': 0
                }), 200
            
            # Create export file
            export_data, mime_type = create_tables_export(
                tables, 
                format_type, 
                sheet_names
            )
            
            # Encode file data as base64
            file_data_b64 = base64.b64encode(export_data).decode('utf-8')
            
            # Get original filename without extension
            original_name = os.path.splitext(filename)[0]
            output_extension = get_file_extension(format_type)
            output_filename = f"{original_name}.{output_extension}"
            
            return jsonify({
                'success': True,
                'message': f'Successfully extracted {len(tables)} table(s)',
                'table_count': len(tables),
                'file_data': file_data_b64,
                'filename': output_filename,
                'mime_type': mime_type
            }), 200
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({
                'error': f'Processing failed: {str(e)}'
            }), 500
            
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up file {file_path}: {e}")
    
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/formats', methods=['GET'])
def get_supported_formats():
    """Get list of supported output formats"""
    formats = [
        {
            'code': 'xlsx',
            'name': 'Microsoft Excel',
            'description': 'Excel workbook format (.xlsx)',
            'extension': 'xlsx'
        },
        {
            'code': 'csv',
            'name': 'Comma Separated Values',
            'description': 'CSV format (.csv)',
            'extension': 'csv'
        },
        {
            'code': 'ods',
            'name': 'OpenDocument Spreadsheet',
            'description': 'LibreOffice Calc format (.ods)',
            'extension': 'ods'
        },
        {
            'code': 'gsheet',
            'name': 'Google Sheets',
            'description': 'Excel format optimized for Google Sheets',
            'extension': 'xlsx'
        }
    ]
    
    return jsonify({
        'formats': formats
    }), 200

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    try:
        # Initialize OCR engine
        logger.info("Initializing OCR engine...")
        load_ocr_engine()
        logger.info("OCR engine loaded successfully")
        
        # Start Flask server
        logger.info("Starting Flask server on http://localhost:5000")
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)