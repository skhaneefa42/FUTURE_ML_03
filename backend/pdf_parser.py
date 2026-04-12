# ============================================================
# backend/pdf_parser.py
# ============================================================
# PURPOSE: Extract raw text from PDF or plain text resume files.
#
# CONCEPT: Before we can analyze a resume, we need to READ it.
# PDFs are binary files — we use pdfplumber to extract text from them.
# Plain .txt files are simply read directly.
# ============================================================

import pdfplumber   # Library to open and read PDF files
import os           # For file path operations


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    HOW IT WORKS:
    - pdfplumber opens the PDF like opening a book
    - It reads each "page" and pulls out the text characters
    - We join all pages into one big string

    Args:
        file_path: Path to the PDF file on disk

    Returns:
        A single string with all text from the PDF
    """
    text = ""  # We'll build our text here

    try:
        # Open the PDF file using pdfplumber
        with pdfplumber.open(file_path) as pdf:

            # Loop through every page in the PDF
            for page_number, page in enumerate(pdf.pages):

                # Extract text from this page (returns None if page is empty)
                page_text = page.extract_text()

                # Only add if we actually got text (not None or empty)
                if page_text:
                    text += page_text + "\n"  # Add newline between pages

        print(f"✅ Extracted text from {len(pdf.pages)} pages in PDF")

    except Exception as e:
        # If something goes wrong, tell us what happened
        print(f"❌ Error reading PDF: {e}")
        text = ""

    return text.strip()  # Remove leading/trailing whitespace


def extract_text_from_txt(file_path: str) -> str:
    """
    Read a plain text (.txt) resume file.

    Args:
        file_path: Path to the .txt file

    Returns:
        Contents of the file as a string
    """
    try:
        # Open the file in read mode with UTF-8 encoding
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"✅ Read text file: {os.path.basename(file_path)}")
        return text.strip()

    except Exception as e:
        print(f"❌ Error reading text file: {e}")
        return ""


def parse_resume(file_path: str) -> str:
    """
    MAIN FUNCTION: Automatically detect file type and extract text.

    This is the function you'll call from other parts of the project.
    It figures out if the file is a PDF or TXT and calls the right function.

    Args:
        file_path: Path to the uploaded resume file

    Returns:
        Extracted text as a string
    """
    # Get the file extension (lowercase for comparison)
    # Example: "resume.PDF" → ".pdf"
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        # It's a PDF → use pdfplumber
        return extract_text_from_pdf(file_path)

    elif file_extension == ".txt":
        # It's a text file → just read it
        return extract_text_from_txt(file_path)

    else:
        # We don't support other formats like .docx yet
        print(f"⚠️  Unsupported file type: {file_extension}")
        return ""


# ---- QUICK TEST ----
# Run this file directly to test it:
# python backend/pdf_parser.py
if __name__ == "__main__":
    # Test with a sample file (change path as needed)
    sample_path = "sample_data/sample_resume.txt"
    result = parse_resume(sample_path)
    print("\n--- Extracted Text Preview ---")
    print(result[:500])  # Show first 500 characters
