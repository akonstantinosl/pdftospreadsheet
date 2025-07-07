import streamlit as st
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

# Configure application with wide layout - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide", page_title="PDF to Spreadsheet Converter")

# Initialize OCR engine with improved parameters for better small text detection
@st.cache_resource
def load_ocr_engine():
    return RapidOCR(
        rec_batch_num=6,  # Increase batch size for better context
        det_model_name='ch_PP-OCRv4_det',  # Use newer detection model
        cls_model_name='ch_ppocr_mobile_v2.0_cls',
        rec_model_name='ch_PP-OCRv4_rec',
        use_angle_cls=True,  # Enable angle classification
        det_limit_side_len=960,  # Increase for better small text detection
        det_db_thresh=0.3,  # Lower threshold for better text detection
        det_db_box_thresh=0.5,  # Adjust box threshold
        det_db_unclip_ratio=1.6  # Higher unclip ratio to avoid merging small text
    )

engine = load_ocr_engine()

# Preprocess image for better OCR results with small text
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

# Enhanced line detection for dotted/dashed lines
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

# Enhanced table detection with support for dotted lines
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

# Text-based table detection with improved handling for small text
def text_based_table_detection(img_array, gray):
    """
    Detects tables based on text alignment and clustering when line detection fails.
    
    Parameters:
    - img_array: Input image as numpy array
    - gray: Grayscale version of the image
    
    Returns:
    - List of detected table regions
    """
    # Apply preprocessing for better small text detection
    preprocessed_img = preprocess_image_for_ocr(img_array)
    
    # Run OCR on the preprocessed image
    ocr_result = engine(preprocessed_img)
    
    if not ocr_result or len(ocr_result[0]) == 0:
        return []
    
    # Extract text boxes and their positions
    text_boxes = []
    for box, text, confidence in ocr_result[0]:
        if text.strip():  # Skip empty text
            # Calculate bounding box
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            width, height = x_max - x_min, y_max - y_min
            
            text_boxes.append({
                'box': box,
                'text': text,
                'confidence': confidence,
                'x': x_min,
                'y': y_min,
                'width': width,
                'height': height,
                'x_center': (x_min + x_max) / 2,
                'y_center': (y_min + y_max) / 2
            })
    
    if not text_boxes:
        return []
    
    # Cluster text boxes into potential table regions
    # First, sort by y-coordinate to identify rows
    text_boxes_sorted_y = sorted(text_boxes, key=lambda b: b['y'])
    
    # Calculate average text height for row grouping
    avg_height = sum(box['height'] for box in text_boxes) / len(text_boxes)
    row_threshold = max(avg_height * 1.5, 15)  # Adjust threshold as needed
    
    # Group text boxes by rows
    rows = []
    current_row = [text_boxes_sorted_y[0]]
    current_y = text_boxes_sorted_y[0]['y_center']
    
    for box in text_boxes_sorted_y[1:]:
        if abs(box['y_center'] - current_y) <= row_threshold:
            # Same row
            current_row.append(box)
        else:
            # New row
            rows.append(current_row)
            current_row = [box]
            current_y = box['y_center']
    
    if current_row:
        rows.append(current_row)
    
    # We need at least 2 rows to consider it a table
    if len(rows) < 2:
        return []
    
    # Find the bounding rectangle of potential tables
    table_candidates = []
    
    # Check each group of consecutive rows as a potential table
    min_rows_for_table = 3  # Adjust as needed
    for i in range(len(rows) - min_rows_for_table + 1):
        # Consider rows[i:i+min_rows_for_table+k] as a potential table
        for k in range(len(rows) - i - min_rows_for_table + 1):
            potential_table_rows = rows[i:i+min_rows_for_table+k]
            
            # Check if these rows form a consistent table structure
            # For simplicity, we'll just use consistent number of elements per row
            row_lengths = [len(row) for row in potential_table_rows]
            
            # Basic check: consistent number of columns
            if max(row_lengths) - min(row_lengths) <= 1:  # Allow some flexibility
                # Calculate the bounding box
                all_boxes = [box for row in potential_table_rows for box in row]
                x_min = min(box['x'] for box in all_boxes)
                y_min = min(box['y'] for box in all_boxes)
                x_max = max(box['x'] + box['width'] for box in all_boxes)
                y_max = max(box['y'] + box['height'] for box in all_boxes)
                
                # Add some padding
                x_min = max(0, x_min - 10)
                y_min = max(0, y_min - 10)
                x_max = min(img_array.shape[1], x_max + 10)
                y_max = min(img_array.shape[0], y_max + 10)
                
                width = x_max - x_min
                height = y_max - y_min
                
                # Create empty arrays for consistency with the grid-based method
                empty_lines = np.zeros_like(gray)
                
                table_candidates.append((int(x_min), int(y_min), int(width), int(height), empty_lines, empty_lines))
    
    return table_candidates

# Enhanced cell detection based on text positions
def detect_cells_from_text(text_items, table_width, table_height):
    """
    Detects cell structure from text positions when explicit grid lines are missing.
    
    Parameters:
    - text_items: List of detected text items with positions
    - table_width: Width of the table region
    - table_height: Height of the table region
    
    Returns:
    - Tuple of (row_boundaries, column_boundaries)
    """
    if not text_items:
        return None, None
    
    # Calculate average text dimensions
    avg_height = sum(item['height'] for item in text_items) / len(text_items)
    avg_width = sum(item['width'] for item in text_items) / len(text_items)
    
    # Sort text by y-coordinate (for rows)
    sorted_by_y = sorted(text_items, key=lambda t: t['y_center'])
    
    # Create a histogram of y-centers to find rows
    y_centers = [t['y_center'] for t in sorted_by_y]
    y_threshold = max(avg_height * 0.8, 10)  # Adjust threshold based on text size
    
    # Function to cluster coordinates into groups
    def cluster_coordinates(coords, threshold):
        clusters = []
        current_cluster = [coords[0]]
        current_value = coords[0]
        
        for val in coords[1:]:
            if abs(val - current_value) <= threshold:
                # Add to current cluster
                current_cluster.append(val)
            else:
                # Start a new cluster
                if current_cluster:
                    clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [val]
                current_value = val
        
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    # Get row centers
    row_centers = cluster_coordinates(y_centers, y_threshold)
    
    # Convert row centers to row boundaries
    row_boundaries = [0]  # Start at the top of the table
    for i in range(len(row_centers) - 1):
        # Add boundary between rows
        boundary = int((row_centers[i] + row_centers[i+1]) / 2)
        row_boundaries.append(boundary)
    row_boundaries.append(int(table_height))  # End at the bottom of the table
    
    # Sort text by x-coordinate (for columns)
    sorted_by_x = sorted(text_items, key=lambda t: t['x_center'])
    
    # Create a histogram of x-centers to find columns
    x_centers = [t['x_center'] for t in sorted_by_x]
    x_threshold = max(avg_width * 0.8, 20)  # Adjust threshold based on text width
    
    # Get column centers
    col_centers = cluster_coordinates(x_centers, x_threshold)
    
    # Convert column centers to column boundaries
    col_boundaries = [0]  # Start at the left of the table
    for i in range(len(col_centers) - 1):
        # Add boundary between columns
        boundary = int((col_centers[i] + col_centers[i+1]) / 2)
        col_boundaries.append(boundary)
    col_boundaries.append(int(table_width))  # End at the right of the table
    
    return row_boundaries, col_boundaries

# Enhanced table reconstruction from OCR results
def process_table_ocr(ocr_results, table_rect, use_inferred_grid=True):
    """
    Process OCR results to reconstruct a table.
    
    Parameters:
    - ocr_results: OCR detection results
    - table_rect: Detected table rectangle
    - use_inferred_grid: Whether to use grid inference for missing lines
    
    Returns:
    - Reconstructed table data
    """
    x_table, y_table, w_table, h_table = table_rect[:4]
    
    if not ocr_results or len(ocr_results[0]) == 0:
        return None
    
    # Filter text inside the table area and collect positions
    table_texts = []
    for box, text, confidence in ocr_results[0]:
        # Calculate center of the text box
        x_center = sum([p[0] for p in box]) / 4
        y_center = sum([p[1] for p in box]) / 4
        
        # Check if the text is inside or overlapping with the table
        if ((x_table - 5 <= x_center <= x_table + w_table + 5) and 
            (y_table - 5 <= y_center <= y_table + h_table + 5)):
            # Get bounding box coordinates
            x_min = min([p[0] for p in box])
            y_min = min([p[1] for p in box])
            width = max([p[0] for p in box]) - x_min
            height = max([p[1] for p in box]) - y_min
            
            # Normalize coordinates relative to the table
            x_rel = x_min - x_table
            y_rel = y_min - y_table
            
            # Skip empty text
            if text.strip():
                table_texts.append({
                    'text': text.strip(),
                    'confidence': confidence,
                    'x': x_rel,
                    'y': y_rel,
                    'width': width,
                    'height': height,
                    'x_center': x_center - x_table,
                    'y_center': y_center - y_table,
                    'original_box': box
                })
    
    if not table_texts:
        return None
    
    # If we should use grid inference for missing lines
    if use_inferred_grid:
        # Detect cell structure based on text positions
        row_boundaries, col_boundaries = detect_cells_from_text(table_texts, w_table, h_table)
        
        if row_boundaries and col_boundaries and len(row_boundaries) > 1 and len(col_boundaries) > 1:
            # Initialize empty table structure
            num_rows = len(row_boundaries) - 1  # Number of cells between boundaries
            num_cols = len(col_boundaries) - 1
            table_data = [[''] * num_cols for _ in range(num_rows)]
            
            # Place text in the corresponding cells
            for text_item in table_texts:
                # Find the cell this text belongs to
                row_idx = None
                for i in range(len(row_boundaries) - 1):
                    if row_boundaries[i] <= text_item['y_center'] < row_boundaries[i+1]:
                        row_idx = i
                        break
                
                col_idx = None
                for i in range(len(col_boundaries) - 1):
                    if col_boundaries[i] <= text_item['x_center'] < col_boundaries[i+1]:
                        col_idx = i
                        break
                
                # Add text to the cell if indices are valid
                if row_idx is not None and col_idx is not None:
                    if row_idx < num_rows and col_idx < num_cols:
                        if table_data[row_idx][col_idx]:
                            table_data[row_idx][col_idx] += ' ' + text_item['text']
                        else:
                            table_data[row_idx][col_idx] = text_item['text']
            
            return table_data
    
    # Fall back to standard table reconstruction method
    return reconstruct_table(table_texts, w_table, h_table)

# Improved table reconstruction with better row/column detection
def reconstruct_table(texts, table_width, table_height):
    """
    Reconstructs table structure from text positions.
    
    Parameters:
    - texts: List of detected text items
    - table_width: Width of the table region
    - table_height: Height of the table region
    
    Returns:
    - Reconstructed table data
    """
    if not texts:
        return None
    
    # Step 1: Identify rows based on y-coordinate clustering with dynamic thresholding
    texts_sorted_by_y = sorted(texts, key=lambda t: t['y'])
    
    # Calculate average text height for better row separation
    avg_text_height = sum(t['height'] for t in texts) / len(texts)
    row_threshold = max(avg_text_height * 0.8, 12)  # Adaptive threshold based on text size
    
    # Group texts by rows
    rows = []
    current_row = [texts_sorted_by_y[0]]
    row_y = texts_sorted_by_y[0]['y_center']
    
    for t in texts_sorted_by_y[1:]:
        if abs(t['y_center'] - row_y) <= row_threshold:
            # Same row
            current_row.append(t)
        else:
            # New row
            if current_row:
                # Sort the current row by x-coordinate
                current_row = sorted(current_row, key=lambda t: t['x'])
                rows.append(current_row)
            
            # Start a new row
            current_row = [t]
            row_y = t['y_center']
    
    # Add the last row
    if current_row:
        current_row = sorted(current_row, key=lambda t: t['x'])
        rows.append(current_row)
    
    # Step 2: Identify columns with improved clustering
    # Get all x-centers and sort them
    all_x_centers = []
    for row in rows:
        for t in row:
            all_x_centers.append(t['x_center'])
    
    # Function to cluster x-centers with dynamic threshold
    def cluster_x_centers(x_centers):
        if not x_centers:
            return []
        
        # Calculate average text width for better column separation
        avg_text_width = sum(t['width'] for t in texts) / len(texts)
        col_threshold = max(avg_text_width * 1.5, 20)  # Adaptive threshold
        
        x_centers = sorted(x_centers)
        clusters = [[x_centers[0]]]
        
        for x in x_centers[1:]:
            if abs(x - clusters[-1][-1]) <= col_threshold:
                # Add to the last cluster
                clusters[-1].append(x)
            else:
                # Start a new cluster
                clusters.append([x])
        
        # Calculate average for each cluster
        return [sum(cluster) / len(cluster) for cluster in clusters]
    
    # Get column centers
    col_centers = cluster_x_centers(all_x_centers)
    num_cols = len(col_centers)
    
    # Ensure we have at least one column
    if num_cols == 0:
        num_cols = 1
        col_centers = [table_width / 2]
    
    # Step 3: Create the table data structure with improved cell assignment
    table_data = []
    
    # Assign each text to the closest column in its row
    for row_texts in rows:
        row_data = [''] * num_cols
        
        for text in row_texts:
            # Find the closest column center
            col_idx = min(range(num_cols), 
                       key=lambda i: abs(text['x_center'] - col_centers[i]))
            
            # If the cell already has content, append with a space
            if row_data[col_idx]:
                row_data[col_idx] += ' ' + text['text']
            else:
                row_data[col_idx] = text['text']
        
        table_data.append(row_data)
    
    return table_data

# Function to create output files from tables in different formats
def create_tables_export(tables, output_format='xlsx', sheet_names=None):
    """
    Creates a file from extracted tables in the specified format.
    
    Parameters:
    - tables: List of pandas DataFrames
    - output_format: Format to export ('xlsx', 'csv', 'ods', 'gsheet')
    - sheet_names: Optional list of sheet names
    
    Returns:
    - File data as bytes and appropriate mime type
    """
    output = io.BytesIO()
    
    # Helper function to auto-adjust column widths for Excel formats
    def auto_adjust_excel_column_widths(workbook, sheet_name, dataframe):
        worksheet = workbook.sheets[sheet_name]
        for idx, col in enumerate(dataframe.columns):
            max_length = 0
            for entry in dataframe[col].astype(str):
                max_length = max(max_length, len(entry))
            column_width = max(max_length, len(str(col)))
            worksheet.column_dimensions[get_column_letter(idx+1)].width = column_width + 2
    
    # Helper function to calculate column widths for any format
    def calculate_column_widths(dataframe):
        widths = []
        for col in dataframe.columns:
            max_length = 0
            for entry in dataframe[col].astype(str):
                max_length = max(max_length, len(entry))
            column_width = max(max_length, len(str(col)))
            widths.append(column_width + 2)  # Add padding
        return widths
    
    if output_format == 'xlsx':
        # Excel format (XLSX)
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for i, df in enumerate(tables):
                sheet_name = f"Table {i+1}" if not sheet_names else sheet_names[i]
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                
                # Auto-adjust column widths
                auto_adjust_excel_column_widths(writer, sheet_name, df)
        
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    elif output_format == 'csv':
        # CSV format (single table or zipped multiple tables)
        if len(tables) == 1:
            # Single table - simple CSV
            tables[0].to_csv(output, index=False, header=False)
            mime_type = "text/csv"
        else:
            # Multiple tables - create a ZIP file with multiple CSVs
            import zipfile
            with zipfile.ZipFile(output, 'w') as zf:
                for i, df in enumerate(tables):
                    file_name = f"table_{i+1}.csv" if not sheet_names else f"{sheet_names[i]}.csv"
                    # Create a temporary buffer for each CSV
                    temp_csv = io.StringIO()
                    df.to_csv(temp_csv, index=False, header=False)
                    zf.writestr(file_name, temp_csv.getvalue())
            mime_type = "application/zip"
    
    elif output_format == 'ods':
        # ODS format - Try to use odfpy for better column width support
        try:
            # Try using odfpy for better control over column widths
            from odf.opendocument import OpenDocumentSpreadsheet
            from odf.style import Style, TableColumnProperties, TableRowProperties
            from odf.table import Table, TableColumn, TableRow, TableCell
            from odf.text import P
            
            # Create a new ODS document
            doc = OpenDocumentSpreadsheet()
            
            # Create tables and set column widths
            for i, df in enumerate(tables):
                sheet_name = f"Table {i+1}" if not sheet_names else sheet_names[i]
                
                # Create table
                table = Table(name=sheet_name)
                doc.spreadsheet.addElement(table)
                
                # Calculate optimal column widths
                col_widths = calculate_column_widths(df)
                
                # Add columns with width styles
                for col_idx, width in enumerate(col_widths):
                    # Create style for this column
                    style_name = f"col{col_idx}_style"
                    col_style = Style(name=style_name, family="table-column")
                    # Convert character width to cm (approx. 0.18cm per char)
                    width_cm = width * 0.18
                    col_style.addElement(TableColumnProperties(columnwidth=f"{width_cm:.2f}cm"))
                    doc.automaticstyles.addElement(col_style)
                    
                    # Add column with style
                    table.addElement(TableColumn(stylename=style_name))
                
                # Add data rows
                for row_idx, row in enumerate(df.values):
                    tr = TableRow()
                    table.addElement(tr)
                    
                    for col_idx, cell_value in enumerate(row):
                        tc = TableCell()
                        tr.addElement(tc)
                        
                        # Handle different value types
                        if cell_value is not None:
                            p = P(text=str(cell_value))
                            tc.addElement(p)
            
            # Save to output buffer
            doc.save(output)
            
        except ImportError:
            # Fall back to pyexcel_ods3 which is more actively maintained than pyexcel_ods
            try:
                import pyexcel_ods3
                
                # Prepare data structure for ODS export
                ods_data = {}
                for i, df in enumerate(tables):
                    sheet_name = f"Table {i+1}" if not sheet_names else sheet_names[i]
                    # Convert DataFrame to list of lists
                    sheet_data = df.values.tolist()
                    ods_data[sheet_name] = sheet_data
                
                # Write to output buffer
                pyexcel_ods3.save_data(output, ods_data)
                
            except ImportError:
                # Try original pyexcel_ods
                try:
                    import pyexcel_ods
                    
                    # Prepare data structure for ODS export
                    ods_data = {}
                    for i, df in enumerate(tables):
                        sheet_name = f"Table {i+1}" if not sheet_names else sheet_names[i]
                        # Convert DataFrame to list of lists
                        sheet_data = df.values.tolist()
                        ods_data[sheet_name] = sheet_data
                    
                    # Write to output buffer
                    pyexcel_ods.save_data(output, ods_data)
                    
                except ImportError:
                    # Fallback to saving as Excel
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for i, df in enumerate(tables):
                            sheet_name = f"Table {i+1}" if not sheet_names else sheet_names[i]
                            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                            
                            # Auto-adjust column widths
                            auto_adjust_excel_column_widths(writer, sheet_name, df)
        
        mime_type = "application/vnd.oasis.opendocument.spreadsheet"
    
    elif output_format == 'gsheet':
        # For Google Sheets, we'll create an XLSX that's formatted well for upload
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for i, df in enumerate(tables):
                sheet_name = f"Table {i+1}" if not sheet_names else sheet_names[i]
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                
                # Auto-adjust column widths
                auto_adjust_excel_column_widths(writer, sheet_name, df)
        
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    else:
        # Default to Excel for unsupported formats
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for i, df in enumerate(tables):
                sheet_name = f"Table {i+1}" if not sheet_names else sheet_names[i]
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                
                # Auto-adjust column widths
                auto_adjust_excel_column_widths(writer, sheet_name, df)
        
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    # Return the value and MIME type
    return output.getvalue(), mime_type

# Process a single image to detect and extract tables with improved small text detection
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
        table_data = process_table_ocr(ocr_result, table_rect, use_inferred_grid=True)
        
        if table_data:
            df = pd.DataFrame(table_data)
            tables.append(df)
    
    return tables

# Function to handle PDF processing
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
            sheet_names.append(f"Page{i+1}_Table{j+1}")
    
    return all_tables, sheet_names

# Function to handle image processing
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

# Main UI
def main():
    # Add custom CSS for centering and step styling
    st.markdown("""
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
    .stButton > button {
        width: 100%;
    }
    /* Center the spinner text */
    .stSpinner > div > div > div:last-child {
        display: flex;
        justify-content: center;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Center title
    st.markdown('<h1 class="center-title">PDF to Spreadsheet Converter</h1>', unsafe_allow_html=True)
    st.markdown('<p class="center-title">Ekstrak tabel dari file PDF dan gambar ke berbagai format spreadsheet</p>', unsafe_allow_html=True)
    
    # Create a row for steps 1, 2, and 3 with percentage widths
    col1, col2, col3 = st.columns([33, 33, 34])
    
    # Step 1: File Upload
    with col1:
        st.markdown('<div class="step-header"><span class="step-number">1</span> Pilih File</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("File Upload", type=["jpg", "jpeg", "png", "pdf"], label_visibility="collapsed")
        st.caption("Pilih file PDF atau gambar untuk dikonversi")
    
    # Step 2: Format Selection
    with col2:
        st.markdown('<div class="step-header"><span class="step-number">2</span> Format Output</div>', unsafe_allow_html=True)
        
        # Use a dropdown instead of buttons
        selected_format = st.selectbox(
            "Format Selection",
            options=["XLSX", "CSV", "ODS", "GSHEET"],
            index=0,
            format_func=lambda x: x,
            label_visibility="collapsed"
        )
        
        # Convert to lowercase for processing
        selected_format = selected_format.lower()
        
        st.caption("Pilih format file output untuk hasil ekstraksi tabel")
    
    # Step 3: Convert - now without extra spacing
    with col3:
        st.markdown('<div class="step-header"><span class="step-number">3</span> Konversi</div>', unsafe_allow_html=True)
        convert_button = st.button("Konversi", use_container_width=True, type="primary")
        st.caption("Klik untuk memulai proses konversi")
    
    # Storage for extracted tables and filename
    if "extracted_tables" not in st.session_state:
        st.session_state.extracted_tables = None
        st.session_state.sheet_names = None
        st.session_state.selected_format = None
        st.session_state.original_filename = None
    
    # Process file when convert button is clicked
    if uploaded_file is not None and convert_button:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Store original filename without extension
        st.session_state.original_filename = os.path.splitext(uploaded_file.name)[0]
        
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        try:
            # Center the spinner text with custom message
            with st.spinner("Mengekstrak tabel..."):
                if file_extension == 'pdf':
                    # Process PDF
                    tables, sheet_names = process_pdf(file_path)
                else:
                    # Process image
                    tables = process_image(file_path)
                    sheet_names = [f"Table_{i+1}" for i in range(len(tables))]
                
                # Store tables in session state
                st.session_state.extracted_tables = tables
                st.session_state.sheet_names = sheet_names
                st.session_state.selected_format = selected_format
                
                # Show success message
                if tables:
                    st.success(f"Berhasil mengekstrak {len(tables)} tabel!")
                else:
                    st.warning("Tidak ada tabel yang terdeteksi dalam file")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        finally:
            # Clean up the temporary file
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    # Show download section if tables have been extracted
    if st.session_state.extracted_tables and len(st.session_state.extracted_tables) > 0:
        # Create export file
        export_data, mime_type = create_tables_export(
            st.session_state.extracted_tables, 
            st.session_state.selected_format,
            st.session_state.sheet_names
        )
        
        # Determine file extension based on format
        file_extensions = {
            'xlsx': 'xlsx',
            'csv': 'csv' if len(st.session_state.extracted_tables) == 1 else 'zip',
            'ods': 'ods',
            'gsheet': 'xlsx'  # Fallback for GSHEET
        }
        file_ext = file_extensions.get(st.session_state.selected_format, 'xlsx')
        
        # Use original filename for download
        download_filename = f"{st.session_state.original_filename}.{file_ext}"
        
        # Center the download button - using percentage columns for consistency
        left_col, center_col, right_col = st.columns([33, 34, 33])
        with center_col:
            st.download_button(
                label=f"Download {st.session_state.selected_format.upper()}",
                data=export_data,
                file_name=download_filename,
                mime=mime_type,
                use_container_width=True
            )

if __name__ == "__main__":
    main()