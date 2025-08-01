import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pdf2image
from ocr_processor import load_ocr_engine, preprocess_image_for_ocr
from table_processor import (
    detect_cells_from_text, 
    process_table_ocr, 
    text_based_table_detection,
    detect_lines
)

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

def enhance_image_quality(img_array):
    """
    Enhance image quality for better OCR results.
    
    Parameters:
    - img_array: Input image as numpy array
    
    Returns:
    - Enhanced image
    """
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Reduce noise
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def validate_table_structure(table_data):
    """
    Validate and clean table structure.
    
    Parameters:
    - table_data: Raw table data as list of lists
    
    Returns:
    - Cleaned table data
    """
    if not table_data:
        return None
    
    # Remove completely empty rows
    cleaned_data = []
    for row in table_data:
        if any(cell.strip() for cell in row if cell):
            cleaned_data.append(row)
    
    if not cleaned_data:
        return None
    
    # Ensure all rows have the same number of columns
    max_cols = max(len(row) for row in cleaned_data)
    normalized_data = []
    
    for row in cleaned_data:
        # Pad rows with empty strings if needed
        while len(row) < max_cols:
            row.append('')
        normalized_data.append(row[:max_cols])  # Trim if too long
    
    return normalized_data

def merge_similar_tables(tables, similarity_threshold=0.8):
    """
    Merge tables that have similar structure.
    
    Parameters:
    - tables: List of pandas DataFrames
    - similarity_threshold: Threshold for considering tables similar
    
    Returns:
    - List of merged tables
    """
    if len(tables) <= 1:
        return tables
    
    merged_tables = []
    used_indices = set()
    
    for i, table1 in enumerate(tables):
        if i in used_indices:
            continue
        
        similar_tables = [table1]
        used_indices.add(i)
        
        for j, table2 in enumerate(tables[i+1:], i+1):
            if j in used_indices:
                continue
            
            # Check if tables have similar structure
            if (table1.shape[1] == table2.shape[1] and 
                len(table1.columns) == len(table2.columns)):
                
                # Simple similarity check based on column count
                similarity = 1.0 if table1.shape[1] == table2.shape[1] else 0.0
                
                if similarity >= similarity_threshold:
                    similar_tables.append(table2)
                    used_indices.add(j)
        
        # Merge similar tables
        if len(similar_tables) > 1:
            merged_table = pd.concat(similar_tables, ignore_index=True)
            merged_tables.append(merged_table)
        else:
            merged_tables.append(similar_tables[0])
    
    return merged_tables

def optimize_table_detection(img_array, params=None):
    """
    Optimize table detection parameters based on image characteristics.
    
    Parameters:
    - img_array: Input image as numpy array
    - params: Optional parameters dictionary
    
    Returns:
    - Optimized parameters
    """
    if params is None:
        params = {}
    
    # Analyze image characteristics
    height, width = img_array.shape[:2]
    image_area = height * width
    
    # Convert to grayscale for analysis
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    # Calculate image complexity
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / image_area
    
    # Adjust parameters based on image characteristics
    default_params = {
        'line_sensitivity': 25,
        'use_grid': True,
        'dotted_line_support': True,
        'small_text_enhancement': True
    }
    
    # Adjust line sensitivity based on image size and complexity
    if image_area > 2000000:  # Large image
        default_params['line_sensitivity'] = 40
    elif image_area < 500000:  # Small image
        default_params['line_sensitivity'] = 15
    
    # Adjust based on edge density
    if edge_density > 0.1:  # Complex image
        default_params['dotted_line_support'] = True
        default_params['small_text_enhancement'] = True
    else:  # Simple image
        default_params['dotted_line_support'] = False
    
    # Update with user parameters
    default_params.update(params)
    
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