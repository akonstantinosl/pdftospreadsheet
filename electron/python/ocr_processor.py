from rapidocr_onnxruntime import RapidOCR
import numpy as np
from PIL import Image
import pdf2image
import tempfile
import os
import pandas as pd
import io
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
import cv2
from collections import defaultdict
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys

# Global OCR engine
engine = None

def get_model_path(model_name):
    """Get path to OCR model files"""
    if getattr(sys, 'frozen', False):  # If running as a bundled executable
        base_path = sys._MEIPASS  # Temporary folder created by PyInstaller
    else:
        base_path = os.path.abspath('.')  # Current directory during development
    
    # Join the path to your models folder
    return os.path.join(base_path, 'models', model_name)

def load_ocr_engine():
    """Initialize and load the OCR engine"""
    global engine
    if engine is None:
        engine = RapidOCR(
            rec_batch_num=6,  # Increase batch size for better context
            det_model_name=get_model_path('ch_PP-OCRv4_det_infer.onnx'),
            cls_model_name=get_model_path('ch_ppocr_mobile_v2.0_cls_infer.onnx'),
            rec_model_name=get_model_path('ch_PP-OCRv4_rec_infer.onnx'),
            use_angle_cls=True,  # Enable angle classification
            det_limit_side_len=960,  # Increase for better small text detection
            det_db_thresh=0.3,  # Lower threshold for better text detection
            det_db_box_thresh=0.5,  # Adjust box threshold
            det_db_unclip_ratio=1.6  # Higher unclip ratio to avoid merging small text
        )
    return engine

def preprocess_image_for_ocr(img_array):
    """
    Preprocess image to improve OCR results for small text.
    
    Parameters:
    - img_array: Input image as numpy array
    
    Returns:
    - Preprocessed image
    """
    # Convert to grayscale if not already
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    # Apply adaptive thresholding to better separate small text
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply slight dilation to enhance small features
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Perform edge enhancement
    edge_enhanced = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    sharpened = cv2.addWeighted(gray, 1.5, edge_enhanced, -0.5, 0)
    
    return sharpened

def detect_lines(img_gray, min_length=50, orientation='horizontal', line_type='any'):
    """
    Detects lines in an image with enhanced support for dotted and dashed lines.
    
    Parameters:
    - img_gray: Grayscale image
    - min_length: Minimum length of line to detect
    - orientation: 'horizontal' or 'vertical'
    - line_type: 'solid', 'dotted', or 'any'
    
    Returns:
    - Binary image with detected lines
    """
    height, width = img_gray.shape
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # For dotted line detection, we'll use a different approach
    if line_type in ['dotted', 'any']:
        # Create a profile (projection) along the appropriate axis
        if orientation == 'horizontal':
            profile = np.sum(binary, axis=1)  # Sum along rows for horizontal lines
            smooth_profile = gaussian_filter1d(profile, sigma=2)  # Smooth the profile
            
            # Create a blank image for the detected lines
            lines_img = np.zeros_like(binary)
            
            # Find peaks in the profile (possible lines)
            peaks, _ = find_peaks(smooth_profile, height=width*0.1, distance=15)
            
            # Draw the detected lines
            for peak in peaks:
                if peak > 0 and peak < height:
                    lines_img[peak, :] = 255
                    
        else:  # vertical
            profile = np.sum(binary, axis=0)  # Sum along columns for vertical lines
            smooth_profile = gaussian_filter1d(profile, sigma=2)
            
            lines_img = np.zeros_like(binary)
            
            peaks, _ = find_peaks(smooth_profile, height=height*0.1, distance=15)
            
            for peak in peaks:
                if peak > 0 and peak < width:
                    lines_img[:, peak] = 255
                    
        # For traditional line detection (morphological operations)
        if line_type == 'any':
            # Also use traditional method
            if orientation == 'horizontal':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
            else:  # vertical
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))
            
            morph_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Combine both methods
            combined_lines = cv2.bitwise_or(lines_img, morph_lines)
            return combined_lines
        
        return lines_img
    
    else:  # Traditional solid line detection using morphological operations
        if orientation == 'horizontal':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
        else:  # vertical
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))
        
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return lines

def detect_tables(img_array, line_sensitivity=25, use_grid=True, dotted_line_support=True):
    """
    Detects tables in an image with enhanced support for dotted/dashed lines.
    
    Parameters:
    - img_array: Input image as numpy array
    - line_sensitivity: Sensitivity parameter for line detection
    - use_grid: Whether to use grid-based detection
    - dotted_line_support: Whether to enable specific dotted line detection
    
    Returns:
    - List of detected table regions with line information
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Check if we should use grid detection
    if use_grid:
        # Detect horizontal and vertical lines with dotted line support if enabled
        if dotted_line_support:
            horizontal_lines = detect_lines(gray, min_length=line_sensitivity, orientation='horizontal', line_type='any')
            vertical_lines = detect_lines(gray, min_length=line_sensitivity, orientation='vertical', line_type='any')
        else:
            # Apply adaptive thresholding for better line detection
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Traditional method (from original code)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_sensitivity, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_sensitivity))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine horizontal and vertical lines to get table grid
        table_grid = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Dilate the grid to connect components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_grid = cv2.dilate(table_grid, kernel, iterations=2)
        
        # Find contours of tables
        contours, _ = cv2.findContours(dilated_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that might be tables
        table_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            # Filter by minimum size (adjust based on image resolution)
            if area > (img_array.shape[0] * img_array.shape[1]) / 30:  # More lenient size filter
                # Check if it's reasonably rectangular
                aspect_ratio = w / float(h)
                if 0.2 < aspect_ratio < 5:
                    table_contours.append((x, y, w, h, horizontal_lines, vertical_lines))
        
        # If tables found with grid detection, return them
        if table_contours:
            return table_contours
    
    # Fall back to text-based table detection if grid detection fails or is disabled
    from table_processor import text_based_table_detection
    return text_based_table_detection(img_array, gray)

def process_image_for_tables(img_array):
    """
    Process an image to detect and extract tables with improved handling for small text.
    
    Parameters:
    - img_array: Input image as numpy array
    
    Returns:
    - List of extracted tables as DataFrames
    """
    # Set default parameters
    use_grid = True
    line_sensitivity = 25
    dotted_line_support = True
    small_text_enhancement = True
    
    # Get OCR engine
    engine = load_ocr_engine()
    
    # Preprocess image for better OCR if small text enhancement enabled
    if small_text_enhancement:
        preprocessed_img = preprocess_image_for_ocr(img_array)
        # Run OCR on preprocessed image for better text detection
        ocr_result = engine(preprocessed_img)
    else:
        # Run regular OCR
        ocr_result = engine(img_array)
    
    # Detect tables in the image with user settings
    table_rects = detect_tables(img_array, line_sensitivity, use_grid, dotted_line_support)
    
    if not table_rects:
        # If no tables detected but OCR found text, create a simple table
        if isinstance(ocr_result, tuple) and len(ocr_result) >= 1 and ocr_result[0]:
            # Create a simple dataframe with all text
            all_texts = [item[1] for item in ocr_result[0]]
            return [pd.DataFrame({"Text": all_texts})]
        return []
    
    # Process each detected table
    tables = []
    for i, table_rect in enumerate(table_rects):
        # Process OCR results to reconstruct the table
        from table_processor import process_table_ocr
        table_data = process_table_ocr(ocr_result, table_rect, use_inferred_grid=True)
        
        if table_data:
            df = pd.DataFrame(table_data)
            tables.append(df)
    
    return tables

def process_pdf(pdf_path):
    """
    Process a PDF file and extract tables from all pages.
    
    Parameters:
    - pdf_path: Path to the PDF file
    
    Returns:
    - List of extracted tables and sheet names
    """
    # Convert PDF to images
    pdf_images = pdf2image.convert_from_path(pdf_path, dpi=300)
    
    # Process each page
    all_tables = []
    sheet_names = []
    
    # Process each page
    for i, img in enumerate(pdf_images):
        # Convert PIL Image to numpy array
        img_array = np.array(img.convert('RGB'))
        
        # Process tables
        page_tables = process_image_for_tables(img_array)
        
        # Add tables to the overall list
        for j, df in enumerate(page_tables):
            all_tables.append(df)
            sheet_names.append(f"Page{i+1}")
    
    return all_tables, sheet_names

def process_image(image_path):
    """
    Process an image file and extract tables.
    
    Parameters:
    - image_path: Path to the image file
    
    Returns:
    - List of extracted tables
    """
    # Open the image
    image = Image.open(image_path)
    
    # Convert image to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Process tables
    tables = process_image_for_tables(img_array)
    
    return tables