from PyPDF2 import PdfReader
from typing import List
import os


class PDFProcessor:
    def __init__(self):
        self.chunk_size = 1000  # Number of characters per chunk
        self.chunk_overlap = 200  # Number of characters to overlap between chunks

    def read_pdf(self, file_path: str) -> str:
        """Read a PDF file and return its text content."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            if end > text_length:
                end = text_length

            chunk = text[start:end]
            chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def process_pdf(self, file_path: str) -> List[str]:
        """Process a PDF file and return a list of text chunks."""
        text = self.read_pdf(file_path)
        return self.split_text(text)
