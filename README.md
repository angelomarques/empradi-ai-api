# PDF Processing and Vector Search API

This application processes PDF documents, converts them into embeddings using Google's embedding model, and stores them in a vector database for semantic search capabilities.

## Setup

1. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

1. Start the application:

```bash
python app.py
```

2. The API provides three endpoints:

   a. Upload and process a PDF file:

   ```bash
   curl -X POST -F "file=@your_document.pdf" http://localhost:5000/upload
   ```

   b. Upload and process a PDF from URL:

   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"url":"https://example.com/document.pdf"}' \
        http://localhost:5000/upload-url
   ```

   c. Search for similar content:

   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"query":"your search query"}' \
        http://localhost:5000/search
   ```

## Features

- PDF text extraction and chunking
- Google's embedding model integration
- Vector storage using ChromaDB
- Semantic search capabilities
- RESTful API endpoints
- Support for both file uploads and URL-based PDF processing

## Directory Structure

```
.
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables
├── src/
│   ├── pdf_processor/    # PDF processing module
│   ├── embeddings/       # Embedding generation module
│   └── vector_store/     # Vector database module
├── uploads/             # Temporary PDF storage
└── data/
    └── chroma/          # Vector database storage
```

## When installed a new module

```bash
pip freeze > requirements.txt
```