from PyPDF2 import PdfReader
from typing import List
import os


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be less than chunk_size to ensure progress."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _read_pdf_pypdf2(self, file_path: str) -> str:
        """Read a PDF file using PyPDF2 and return its text content."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        reader = PdfReader(file_path)
        # More efficient string concatenation
        page_texts = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:  # Ensure text is not None or empty
                    page_texts.append(text)
            except Exception as e:
                print(
                    f"Warning: Could not extract text from page {page_num + 1} of {file_path}: {e}"
                )
                # Optionally, add a placeholder or skip
        return "\n".join(
            page_texts
        )  # Join with newlines to preserve some page structure

    # --- Alternative PDF Readers (Examples) ---
    # def _read_pdf_pymupdf(self, file_path: str) -> str:
    #     """Read a PDF file using PyMuPDF (fitz) and return its text content."""
    #     if not os.path.exists(file_path):
    #         raise FileNotFoundError(f"PDF file not found: {file_path}")
    #     doc = fitz.open(file_path)
    #     page_texts = []
    #     for page_num in range(len(doc)):
    #         page = doc.load_page(page_num)
    #         page_texts.append(page.get_text("text")) # "text" for plain text
    #     doc.close()
    #     return "\n".join(page_texts)

    # def _read_pdf_pdfminer(self, file_path: str) -> str:
    #     """Read a PDF file using pdfminer.six and return its text content."""
    #     if not os.path.exists(file_path):
    #         raise FileNotFoundError(f"PDF file not found: {file_path}")
    #     return pdfminer_extract_text(file_path)

    def read_pdf(self, file_path: str, method: str = "pypdf2") -> str:
        """
        Read a PDF file and return its text content.
        Supported methods: "pypdf2", "pymupdf", "pdfminer".
        """
        if method == "pypdf2":
            return self._read_pdf_pypdf2(file_path)
        # elif method == "pymupdf":
        #     return self._read_pdf_pymupdf(file_path)
        # elif method == "pdfminer":
        #     return self._read_pdf_pdfminer(file_path)
        else:
            raise ValueError(f"Unsupported PDF reading method: {method}")

    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks. Corrected logic."""
        if not text:
            return []

        chunks = []
        current_pos = 0
        text_len = len(text)

        while current_pos < text_len:
            end_pos = current_pos + self.chunk_size
            chunk = text[
                current_pos:end_pos
            ]  # Slice, automatically handles if end_pos > text_len
            chunks.append(chunk)

            # Move to the next starting position
            # Ensure we make progress. If chunk_size <= chunk_overlap, this would be <= 0.
            # The __init__ check `chunk_overlap >= chunk_size` prevents this.
            next_start_pos = current_pos + self.chunk_size - self.chunk_overlap

            if next_start_pos >= text_len:  # We've covered the entire text
                break

            # If the next start position is not advancing past the current one
            # (shouldn't happen due to __init__ checks, but good for absolute safety)
            if next_start_pos <= current_pos:
                # This case indicates an issue with chunk_size/overlap logic
                # or very short text where overlap makes no sense.
                # For simplicity, if the next chunk would start before or at the same place,
                # and we haven't reached the end, we just take the rest of the text
                # or break if the last chunk already covers it.
                # However, with the __init__ guard, this shouldn't be hit.
                # If somehow it is, we might just want to break to avoid infinite loop.
                if chunk.endswith(
                    text[next_start_pos:]
                ):  # The current chunk already contains the rest
                    break
                # Fallback: just advance by one if stuck, though not ideal
                # current_pos += 1
                # Better: rely on __init__ checks and ensure next_start_pos advances
                # If this branch is hit, it's a logical error in setup.
                print(
                    f"Warning: split_text stuck or not advancing. current_pos={current_pos}, next_start_pos={next_start_pos}"
                )
                current_pos = end_pos  # Non-overlapping (fallback if logic error)

            else:
                current_pos = next_start_pos

        return chunks

    def process_pdf(self, file_path: str, read_method: str = "pypdf2") -> List[str]:
        """Process a PDF file and return a list of text chunks."""
        print(f"Processing PDF: {file_path} using {read_method}")
        text = self.read_pdf(file_path, method=read_method)
        if not text:
            print(f"Warning: No text extracted from {file_path}")
            return []
        print(f"Extracted text length: {len(text)} characters")

        chunks = self.split_text(text)
        print(f"Split into {len(chunks)} chunks.")
        return chunks
