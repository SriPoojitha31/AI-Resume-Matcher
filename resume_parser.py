import pdfplumber
import fitz  # PyMuPDF
from docx import Document
import io

# Unified function to extract text from PDF or DOCX

def extract_text_from_file(file):
    """
    Extract text from a PDF or DOCX file-like object.
    """
    if hasattr(file, 'name') and file.name.lower().endswith('.pdf'):
        # Try pdfplumber first
        try:
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text.strip()
        except Exception:
            pass
        # Fallback to PyMuPDF
        file.seek(0)
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        return text.strip()
    elif hasattr(file, 'name') and (file.name.lower().endswith('.docx') or file.name.lower().endswith('.doc')):
        file.seek(0)
        doc = Document(io.BytesIO(file.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    else:
        return ""