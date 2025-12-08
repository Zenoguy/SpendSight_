# ocr_utils.py
import easyocr
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import cv2

reader = easyocr.Reader(['en'], gpu=True)

# Table Transformer model for table detection
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection").to(device)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",".webp"}

def is_allowed_image(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

def detect_tables_with_transformer(img_array):
    """Detect table regions using Table Transformer"""
    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_pil = Image.fromarray(img_array)
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([img_pil.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
    
    tables = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        tables.append(box)
    
    return tables

def extract_table_from_region(img_array, bbox):
    """Extract table data from detected region using OCR"""
    x1, y1, x2, y2 = bbox
    table_img = img_array[y1:y2, x1:x2]
    
    result = reader.readtext(table_img, paragraph=False)
    if not result:
        return []
    
    # Collect all cells with their positions
    cells = []
    for bbox_ocr, text, conf in result:
        y_center = (bbox_ocr[0][1] + bbox_ocr[2][1]) / 2
        x_left = bbox_ocr[0][0]
        x_right = bbox_ocr[2][0]
        x_center = (x_left + x_right) / 2
        cells.append({
            'text': text,
            'y': y_center,
            'x_left': x_left,
            'x_center': x_center,
            'x_right': x_right
        })
    
    if not cells:
        return []
    
    # Group into rows by Y coordinate
    cells.sort(key=lambda c: c['y'])
    rows = []
    current_row = [cells[0]]
    row_threshold = 20
    
    for cell in cells[1:]:
        if abs(cell['y'] - current_row[0]['y']) < row_threshold:
            current_row.append(cell)
        else:
            rows.append(sorted(current_row, key=lambda c: c['x_center']))
            current_row = [cell]
    rows.append(sorted(current_row, key=lambda c: c['x_center']))
    
    # Find column boundaries by analyzing X positions across all rows
    all_x_centers = []
    for row in rows:
        for cell in row:
            all_x_centers.append(cell['x_center'])
    
    all_x_centers.sort()
    
    # Cluster X positions to find column centers
    column_centers = []
    col_threshold = 30
    
    for x in all_x_centers:
        if not column_centers or abs(x - column_centers[-1]) > col_threshold:
            column_centers.append(x)
        else:
            # Update to average position
            column_centers[-1] = (column_centers[-1] + x) / 2
    
    num_cols = len(column_centers)
    
    # Assign cells to columns with strict boundaries
    table_data = []
    
    # Calculate column boundaries (midpoints between column centers)
    column_boundaries = []
    for i in range(len(column_centers) - 1):
        boundary = (column_centers[i] + column_centers[i + 1]) / 2
        column_boundaries.append(boundary)
    
    for row in rows:
        row_data = [''] * num_cols
        used_cols = set()
        
        for cell in row:
            x = cell['x_center']
            
            # Determine which column this cell belongs to based on boundaries
            col_idx = 0
            for i, boundary in enumerate(column_boundaries):
                if x > boundary:
                    col_idx = i + 1
            
            # Only assign if column not already used (prevent overwriting)
            if col_idx not in used_cols:
                row_data[col_idx] = cell['text']
                used_cols.add(col_idx)
        
        table_data.append(row_data)
    
    return table_data

def run_ocr_on_image_bytes(image_bytes: bytes, suffix: str) -> tuple[str, list]:
    img = Image.open(BytesIO(image_bytes))
    img_array = np.array(img)
    
    extracted_tables = []
    result_text = []
    
    # Detect tables using Table Transformer
    try:
        table_bboxes = detect_tables_with_transformer(img_array)
        
        if table_bboxes:
            for idx, bbox in enumerate(table_bboxes):
                rows = extract_table_from_region(img_array, bbox)
                
                if rows:
                    result_text.append(f"\n<TABLE id='{idx + 1}'>")
                    for row in rows:
                        result_text.append("| " + " | ".join(row) + " |")
                    result_text.append("</TABLE>\n")
                    extracted_tables.append(rows)
            
            if extracted_tables:
                return "\n".join(result_text), extracted_tables
    except Exception as e:
        print(f"Table Transformer failed: {e}")
    
    # Fallback: Regular OCR
    result = reader.readtext(img_array)
    if result:
        lines = [text for (bbox, text, conf) in result]
        text = "\n".join(lines)
        return text if text else "No text detected", []
    
    return "No text detected", []

def txt_to_pdf_bytes(text: str, username: str, tables: list = None) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Split text by table markers
    parts = text.split('<TABLE')
    
    for part in parts:
        if "id='" in part:
            # Extract table content
            table_end = part.find('</TABLE>')
            table_content = part[:table_end]
            remaining_text = part[table_end + 8:]
            
            # Parse markdown table
            lines = [l.strip() for l in table_content.split('\n') if l.strip() and '|' in l]
            if len(lines) >= 2:
                # Convert markdown to list of lists
                table_data = []
                for line in lines:
                    if not line.startswith('|--'):
                        cells = [cell.strip() for cell in line.split('|')[1:-1]]
                        table_data.append(cells)
                
                # Create PDF table
                if table_data:
                    t = Table(table_data)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 12))
            
            # Add remaining text
            if remaining_text.strip():
                story.append(Paragraph(remaining_text.strip(), styles['Normal']))
        else:
            # Regular text
            if part.strip():
                story.append(Paragraph(part.strip().replace('\n', '<br/>'), styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.read()
