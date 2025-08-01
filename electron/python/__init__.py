# PDF to Spreadsheet Converter - Python Backend
# This file makes the python directory a package

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "OCR-based PDF to Spreadsheet Converter Backend"

# Import main functions for easy access
from .ocr_processor import load_ocr_engine, process_pdf, process_image
from .table_processor import create_tables_export
from .image_processor import (
    detect_tables, 
    process_image_for_tables, 
    enhance_image_quality,
    validate_table_structure,
    merge_similar_tables,
    optimize_table_detection
)

__all__ = [
    'load_ocr_engine',
    'process_pdf', 
    'process_image',
    'create_tables_export',
    'detect_tables',
    'process_image_for_tables',
    'enhance_image_quality',
    'validate_table_structure', 
    'merge_similar_tables',
    'optimize_table_detection'
]