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

3. Create a `.env` file in the root directory and add your API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
MONGODB_URI=your_mongodb_connection_string_here
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

   d. **Prompt Management** (New):

   ```bash
   # Create a new prompt
   curl -X POST -H "Content-Type: application/json" \
        -d '{"name":"My Prompt","content":"You are an AI assistant..."}' \
        http://localhost:5000/prompts

   # Get all prompts
   curl -X GET http://localhost:5000/prompts

   # Get active prompt
   curl -X GET http://localhost:5000/prompts/active

   # Activate a specific prompt
   curl -X POST http://localhost:5000/prompts/{prompt_id}/activate
   ```

## Features

- PDF text extraction and chunking
- Google's embedding model integration
- Vector storage using ChromaDB
- Semantic search capabilities
- RESTful API endpoints
- Support for both file uploads and URL-based PDF processing
- **Dynamic prompt management** - Store and manage AI prompts in the database
- **MongoDB integration** - Store articles and prompts with full CRUD operations

## Directory Structure

```
.
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables
├── init_prompt.py        # Initialize default prompt in database
├── test_prompt.http      # Test prompt management endpoints
├── src/
│   ├── pdf_processor/    # PDF processing module
│   ├── embeddings/       # Embedding generation module
│   ├── vector_store/     # Vector database module
│   └── models/           # Database models
│       ├── article.py    # Article model
│       └── prompt.py     # Prompt model (NEW)
├── uploads/             # Temporary PDF storage
└── data/
    └── chroma/          # Vector database storage
```

## Prompt Management

The application now supports dynamic prompt management through the database. This allows you to:

- Store multiple prompts with different versions
- Activate/deactivate prompts without restarting the application
- Update prompts in real-time
- Maintain prompt history and versioning

### Initial Setup

To initialize the default prompt in the database:

```bash
python init_prompt.py
```

### API Endpoints

| Method | Endpoint                 | Description                     |
| ------ | ------------------------ | ------------------------------- |
| POST   | `/prompts`               | Create a new prompt             |
| GET    | `/prompts`               | Get all prompts                 |
| GET    | `/prompts/{id}`          | Get a specific prompt           |
| PUT    | `/prompts/{id}`          | Update a prompt                 |
| DELETE | `/prompts/{id}`          | Delete a prompt                 |
| POST   | `/prompts/{id}/activate` | Activate a prompt               |
| GET    | `/prompts/active`        | Get the currently active prompt |

### Prompt Model Fields

- `name`: Unique name for the prompt
- `content`: The actual prompt content
- `description`: Optional description
- `is_active`: Boolean flag for active status
- `version`: Version string for tracking
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

## When installed a new module

```bash
pip freeze > requirements.txt
```
