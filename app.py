# app.py
from flask import Flask, request, jsonify
from src.pdf_processor.processor import PDFProcessor
from src.embeddings.generator import EmbeddingGenerator
from src.vector_store.store import VectorStore
import os
from werkzeug.utils import secure_filename
import requests
from urllib.parse import urlparse
import tempfile
from flask_cors import CORS  # Import CORS
from dotenv import load_dotenv
from google import genai

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# --- Configure CORS ---
# Option 1: Basic - Allow all origins for all routes (good for development)
CORS(app)

# Option 2: More Specific - Allow only your React app's origin
# This is better for security, especially as you move towards production.
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
# The above line means:
# - For any route starting with "/api/" (e.g., /api/data, /api/users)
# - Allow requests specifically from "http://localhost:5173"

# If you want to allow credentials (cookies, authorization headers)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

# Create necessary directories
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("data/chroma", exist_ok=True)

# Initialize components
pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
embedding_generator = EmbeddingGenerator()
vector_store = VectorStore()


@app.route("/upload", methods=["POST"])
def upload_pdf():
    print("Uploading PDF")
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # Process PDF
        # TODO: find processor alternative
        text_chunks = pdf_processor.process_pdf(filepath)

        # Generate embeddings
        embeddings = embedding_generator.generate_embeddings(text_chunks)

        # Store in vector database
        metadata = [{"source": filename} for _ in text_chunks]
        vector_store.store_embeddings(text_chunks, embeddings, metadata)

        return (
            jsonify(
                {"message": "PDF processed successfully", "chunks": len(text_chunks)}
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up uploaded file
        print("Cleaning up uploaded file")
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Generate embedding for the query
        query_embedding = embedding_generator.generate_embeddings([data["query"]])[0]

        # Search for similar documents
        results = vector_store.search_similar(query_embedding)

        # Prepare context for Gemini
        context = "\n\n".join([result["text"] for result in results])

        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        # Generate response using Gemini
        client = genai.Client(api_key=api_key)

        prompt = f"""Based on the following context, please answer the question. 
        If the context doesn't contain enough information to answer the question, say so.
        
        Context:
        {context}
        
        Question: {data['query']}
        
        Answer:"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        gemini_response = response.text

        return jsonify({"results": results, "answer": gemini_response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload-url", methods=["POST"])
def upload_pdf_from_url():
    print("Uploading PDF from URL")
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data["url"]
    title = data.get(
        "title", "Untitled"
    )  # Get title from request body, default to "Untitled"

    # Validate URL
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return jsonify({"error": "Invalid URL provided"}), 400
    except Exception:
        return jsonify({"error": "Invalid URL provided"}), 400

    # Create a temporary file to store the PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        try:
            # Download the PDF
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Check if the content type is PDF
            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" not in content_type:
                return jsonify({"error": "URL does not point to a PDF file"}), 400

            # Save the PDF to the temporary file
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()

            # Process PDF
            # TODO: find processor alternative
            text_chunks = pdf_processor.process_pdf(temp_file.name)

            # Generate embeddings
            embeddings = embedding_generator.generate_embeddings(text_chunks)

            # Store in vector database
            filename = os.path.basename(parsed_url.path) or "downloaded.pdf"
            metadata = [
                {"source": filename, "url": url, "title": title} for _ in text_chunks
            ]
            vector_store.store_embeddings(text_chunks, embeddings, metadata)

            return (
                jsonify(
                    {
                        "message": "PDF processed successfully",
                        "chunks": len(text_chunks),
                    }
                ),
                200,
            )

        except requests.RequestException as e:
            return jsonify({"error": f"Failed to download PDF: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            print("Cleaning up temporary file")
            # Clean up the temporary file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


if __name__ == "__main__":
    # This is for local development only.
    # Gunicorn will run the app in production.
    app.run(debug=True, host="0.0.0.0", port=5000)
