import cv2
import numpy as np
import pandas as pd
import io
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from openpyxl.utils import get_column_letter

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

def text_based_table_detection(img_array, gray):
    """
    Detects tables based on text alignment and clustering when line detection fails.
    
    Parameters:
    - img_array: Input image as numpy array
    - gray: Grayscale version of the image
    
    Returns:
    - List of detected table regions
    """
    # Import here to avoid circular imports
    from ocr_processor import load_ocr_engine, preprocess_image_for_ocr
    
    # Get OCR engine
    engine = load_ocr_engine()
    
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
    def auto_adjust_excel_column_widths(writer, sheet_name, dataframe):
        try:
            # Access the workbook and worksheet
            workbook = writer.book
            worksheet = workbook[sheet_name]
            
            for idx, col in enumerate(dataframe.columns):
                max_length = 0
                column_letter = get_column_letter(idx + 1)
                
                # Calculate max length for the column
                for entry in dataframe[col].astype(str):
                    max_length = max(max_length, len(entry))
                
                # Also check column header length
                column_width = max(max_length, len(str(col)))
                
                # Set column width (add padding)
                adjusted_width = min(column_width + 2, 50)  # Cap at 50 characters
                worksheet.column_dimensions[column_letter].width = adjusted_width
        except Exception as e:
            # If column width adjustment fails, continue without it
            print(f"Warning: Could not adjust column widths for {sheet_name}: {e}")
            pass
    
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
        # ODS format
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