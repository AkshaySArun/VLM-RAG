import os
import fitz  # PyMuPDF
import hashlib
import numpy as np
import io
from PIL import Image
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

from src.ingestion.image_processor import ImageProcessor

# Load environment variables
load_dotenv()

class PDFParser:
    def __init__(self, output_dir: str = None):
        """
        Initializes the PDFParser.
        :param output_dir: Directory where processed assets (like images) will be stored.
        """
        base_output = output_dir or os.getenv("PROCESSED_DATA_PATH", "./data/processed")
        self.output_dir = Path(base_output)
        self.image_dir = self.output_dir / "images"
        
        # Initialize OCR for image enrichment
        self.image_processor = ImageProcessor()
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def extract_content(self, pdf_path: str, paper_id: str = None, filename: str = None) -> List[Dict[str, Any]]:
        """
        Parses a PDF file and extracts text, structured tables, and images.
        Uses unstructured for layout/tables and PyMuPDF for images.
        """
        raw_filename = os.path.basename(pdf_path)
        paper_id = paper_id or raw_filename
        display_name = filename or raw_filename
        chunks = []
        
        print(f"[*] Processing document: {display_name} (ID: {paper_id})")
        
        try:
            # strategy="fast" extracts text directly from the PDF stream.
            elements = partition_pdf(
                filename=pdf_path,
                strategy="fast",
                infer_table_structure=True,
                extract_images_in_pdf=False,
            )

            for i, element in enumerate(elements):
                element_type = element.category.lower()
                page_number = element.metadata.page_number if element.metadata.page_number else 1
                content_type = "table" if element_type == "table" else "text"
                
                chunks.append({
                    "doc_id": paper_id,
                    "paper_id": paper_id,
                    "page": page_number,
                    "type": content_type,
                    "content": str(element),
                    "metadata": {
                        "paper_id": paper_id,
                        "filename": display_name,
                        "source": display_name,
                        "page_number": page_number,
                        "content_type": content_type,
                        "element_id": f"{paper_id}_el_{i}"
                    }
                })
        except Exception as e:
            print(f"[!] Unstructured parsing failed for {display_name} (Text/Tables fallback to PyMuPDF text extraction): {e}")
            # PyMuPDF fallback for text if unstructured fails
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        chunks.append({
                            "doc_id": paper_id,
                            "paper_id": paper_id,
                            "page": page_num + 1,
                            "type": "text",
                            "content": text,
                            "metadata": {
                                "paper_id": paper_id,
                                "filename": display_name,
                                "source": display_name,
                                "page_number": page_num + 1,
                                "content_type": "text",
                                "element_id": f"{paper_id}_mupdf_{page_num+1}"
                            }
                        })
                doc.close()
            except Exception as e2:
                print(f"[!] PyMuPDF text extraction also failed for {display_name}: {e2}")

        try:
            # 2. Extract raw images using PyMuPDF
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"].lower()
                    
                    # 1. Skip very small files (logos, icons, spacing elements)
                    if len(image_bytes) < 10240: 
                        continue
                        
                    # 2. Only accept standard formats
                    if image_ext not in ["png", "jpg", "jpeg"]:
                        continue
                    
                    # 3. Filter out pure black or near-empty images
                    try:
                        img_pil = Image.open(io.BytesIO(image_bytes))
                        grayscale = img_pil.convert("L")
                        mean_brightness = np.mean(np.array(grayscale))
                        if mean_brightness < 2 or mean_brightness > 253:
                            continue
                    except Exception:
                        continue
                        
                    # Generate unique hash for the image
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    safe_paper_id = "".join(c if c.isalnum() else "_" for c in paper_id)
                    img_filename = f"{safe_paper_id}_p{page_num+1}_img{img_index}_{img_hash[:8]}.{image_ext}"
                    img_path = self.image_dir / img_filename
                    
                    if not img_path.exists():
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                    
                    print(f"  - Extracted valid image: {img_filename} (Size: {len(image_bytes)//1024}KB, Brightness: {mean_brightness:.1f})", flush=True)
                    
                    # ENRICHMENT: Run OCR on the image to make it text-searchable
                    ocr_text = self.image_processor.ocr_only(img_path)
                    if ocr_text:
                        print(f"    [OCR] Extracted: {ocr_text[:50]}...", flush=True)

                    chunks.append({
                        "doc_id": paper_id,
                        "paper_id": paper_id,
                        "page": page_num + 1,
                        "type": "image",
                        "content": str(img_path),
                        "ocr_text": ocr_text,
                        "metadata": {
                            "paper_id": paper_id,
                            "filename": display_name,
                            "source": display_name,
                            "page_number": page_num + 1,
                            "content_type": "image",
                            "image_path": str(img_path),
                            "ocr_text": ocr_text or ""
                        }
                    })
            doc.close()
        except Exception as e:
            print(f"[!] Image extraction failed for {display_name}: {e}")

        if chunks:
            print(f"[+] Successfully extracted {len(chunks)} elements from {display_name}")
            return chunks
        else:
            return []

if __name__ == "__main__":
    # Quick sanity check logic
    parser = PDFParser()
    print("PDFParser initialized and ready.")
