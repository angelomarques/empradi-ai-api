# app.py
from flask import Flask, request, jsonify
from src.pdf_processor.processor import PDFProcessor
from src.embeddings.generator import EmbeddingGenerator
from src.vector_store.store import VectorStore
from src.models.article import Article, ArticleModel
from src.models.prompt import Prompt, PromptModel
import os
from werkzeug.utils import secure_filename
import requests
from urllib.parse import urlparse
import tempfile
from flask_cors import CORS  # Import CORS
from dotenv import load_dotenv
from google import genai
import time
import asyncio

from flask_pymongo import PyMongo

# from bson.objectid import ObjectId # To convert string IDs to ObjectId for querying

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

load_dotenv()

# --- Configuration ---
# Best practice: Use environment variables for sensitive data like connection strings
# For simplicity, we'll hardcode it here.
# Replace with your actual MongoDB connection URI.
# Example for local MongoDB:
app.config["MONGO_URI"] = os.getenv("MONGODB_URI")
if not app.config["MONGO_URI"]:
    raise ValueError("MONGODB_URI environment variable is not set")


# Example for MongoDB Atlas (replace with your actual URI):
# app.config["MONGO_URI"] = "mongodb+srv://<username>:<password>@yourcluster.mongodb.net/mytodolist?
# retryWrites=true&w=majority"


# Initialize PyMongo
mongo = PyMongo(app)

# Initialize article model after MongoDB connection is established
article_model = ArticleModel(mongo)

# Initialize prompt model after MongoDB connection is established
prompt_model = PromptModel(mongo)


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Generate embedding for the query
        query_embedding = asyncio.run(
            embedding_generator.generate_embeddings_async([data["query"]])
        )[0]

        # Search for similar documents
        results = article_model.search_by_embedding(query_embedding)

        # Prepare context for Gemini
        context = "\n\n".join([result.content for result in results])

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        # Generate response using Gemini
        client = genai.Client(api_key=api_key)

        # Get prompt from database
        prompt_template = prompt_model.get_default_prompt()

        prompt = f"""{prompt_template}
        
        Contexto:
        {context}
        
        Pergunta: {data['query']}
        
        Resposta:"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        gemini_response = response.text

        return (
            jsonify(
                {
                    "results": [result.to_dict() for result in results],
                    "answer": gemini_response,
                }
            ),
            200,
        )

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
            text_chunks = pdf_processor.process_pdf(temp_file.name)
            if not text_chunks:
                return jsonify({"error": "No text content extracted from PDF"}), 400

            embeddings = asyncio.run(
                embedding_generator.generate_embeddings_async(text_chunks)
            )

            # Create and save article in MongoDB
            # Create an article for  each embedding
            for index, embedding in enumerate(embeddings):
                article = Article(
                    title=title,
                    url=url,
                    embeddings=embedding,
                    content=text_chunks[index],
                )
                article_model.create(article)

            return (
                jsonify(
                    {
                        "message": "PDF processed and saved successfully",
                        "title": title,
                        "url": url,
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


@app.route("/articles", methods=["POST"])
def create_article():
    """Create a new article."""
    data = request.get_json()
    if not data or not all(
        k in data for k in ["title", "url", "embeddings", "content"]
    ):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        article = Article(
            title=data["title"],
            url=data["url"],
            embeddings=data["embeddings"],
            content=data["content"],
        )
        article_id = article_model.create(article)
        return (
            jsonify({"message": "Article created successfully", "id": article_id}),
            201,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/articles", methods=["GET"])
def get_articles():
    """Get all articles."""
    try:
        articles = article_model.get_all()
        return jsonify([article.to_dict() for article in articles]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/articles/<article_id>", methods=["GET"])
def get_article(article_id):
    """Get a specific article by ID."""
    try:
        article = article_model.get_by_id(article_id)
        if article:
            return jsonify(article.to_dict()), 200
        return jsonify({"error": "Article not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/articles/<article_id>", methods=["PUT"])
def update_article(article_id):
    """Update an article."""
    data = request.get_json()
    if not data or not all(
        k in data for k in ["title", "url", "embeddings", "content"]
    ):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        article = Article(
            title=data["title"],
            url=data["url"],
            embeddings=data["embeddings"],
            content=data["content"],
        )
        if article_model.update(article_id, article):
            return jsonify({"message": "Article updated successfully"}), 200
        return jsonify({"error": "Article not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/articles/<article_id>", methods=["DELETE"])
def delete_article(article_id):
    """Delete an article."""
    try:
        if article_model.delete(article_id):
            return jsonify({"message": "Article deleted successfully"}), 200
        return jsonify({"error": "Article not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/articles/search", methods=["POST"])
def search_articles():
    """Search articles by embedding similarity."""
    data = request.get_json()
    if not data or "embedding" not in data:
        return jsonify({"error": "No embedding provided"}), 400

    try:
        limit = data.get("limit", 5)
        articles = article_model.search_by_embedding(data["embedding"], limit)
        return jsonify([article.to_dict() for article in articles]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload-json", methods=["POST"])
def upload_json_from_url():
    """Upload and process articles from a JSON file URL."""
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data["url"]

    # Validate URL
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return jsonify({"error": "Invalid URL provided"}), 400
    except Exception:
        return jsonify({"error": "Invalid URL provided"}), 400

    try:
        # Download the JSON file
        response = requests.get(url)
        response.raise_for_status()

        # Parse JSON data
        articles_data = response.json()

        # Process each article
        processed_articles = []
        failed_articles = []
        for index, article_data in enumerate(articles_data):
            # TODO: Generate embeddings for the article content
            # Generate embeddings for the article title
            # title_embedding = embedding_generator.generate_embeddings(
            #     [article_data["nomeTrabalho"]]
            # )[0]

            # # Create article object
            # article = Article(
            #     title=article_data["nomeTrabalho"],
            #     url=article_data["url"],
            #     embeddings=title_embedding,
            #     content=article_data[
            #         "nomeTrabalho"
            #     ],  # Using title as content since that's what we have
            # )

            # # Save to MongoDB
            # article_id = article_model.create(article)
            # processed_articles.append(
            #     {"id": article_id, "title": article.title, "url": article.url}
            # )
            # Validate URL
            try:
                parsed_url = urlparse(article_data["url"])
                if not parsed_url.scheme or not parsed_url.netloc:
                    failed_articles.append(
                        {
                            "title": article_data["nomeTrabalho"],
                            "url": article_data["url"],
                            "error": "Invalid URL provided",
                        }
                    )
                    continue
            except Exception:
                failed_articles.append(
                    {
                        "title": article_data["nomeTrabalho"],
                        "url": article_data["url"],
                        "error": "Invalid URL provided",
                    }
                )
                continue

            # Create a temporary file to store the PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                try:
                    # print("taking break for 10 seconds...")
                    # time.sleep(10)  # Pauses execution for 10 seconds
                    # print("resuming...")
                    print(f"Processing article of index: {index}")
                    print(f"Processing article: {article_data['nomeTrabalho']}")
                    # Download the PDF
                    response = requests.get(article_data["url"], stream=True)
                    response.raise_for_status()  # Raise an exception for bad status codes

                    # Check if the content type is PDF
                    content_type = response.headers.get("content-type", "").lower()
                    if "application/pdf" not in content_type:
                        failed_articles.append(
                            {
                                "title": article_data["nomeTrabalho"],
                                "url": article_data["url"],
                                "error": "URL does not point to a PDF file",
                            }
                        )
                        continue

                    # Save the PDF to the temporary file
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    temp_file.flush()

                    print(f"Processing PDF: {article_data['nomeTrabalho']}")
                    # Process PDF
                    text_chunks = pdf_processor.process_pdf(temp_file.name)
                    if not text_chunks:
                        failed_articles.append(
                            {
                                "title": article_data["nomeTrabalho"],
                                "url": article_data["url"],
                                "error": "No text content extracted from PDF",
                            }
                        )
                        continue

                    print(f"Generating embeddings for: {article_data['nomeTrabalho']}")
                    embeddings = asyncio.run(
                        embedding_generator.generate_embeddings_async(text_chunks)
                    )

                    print("Embeddings generated")

                    # Create and save article in MongoDB
                    # Create an article for  each embedding
                    for index, embedding in enumerate(embeddings):
                        article = Article(
                            title=article_data["nomeTrabalho"],
                            url=article_data["url"],
                            embeddings=embedding,
                            content=text_chunks[index],
                        )
                        article_model.create(article)

                    print(f"Embeddings saved for: {article_data['nomeTrabalho']}")

                    processed_articles.append(
                        {
                            "title": article_data["nomeTrabalho"],
                            "url": article_data["url"],
                        }
                    )

                except requests.RequestException as e:
                    error_msg = f"Failed to download PDF: {str(e)}"
                    print(f"Error processing article: {article_data['nomeTrabalho']}")
                    print(f"Error: {error_msg}")
                    failed_articles.append(
                        {
                            "title": article_data["nomeTrabalho"],
                            "url": article_data["url"],
                            "error": error_msg,
                        }
                    )
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)}"
                    print(f"Error processing article: {article_data['nomeTrabalho']}")
                    print(f"Error: {error_msg}")
                    failed_articles.append(
                        {
                            "title": article_data["nomeTrabalho"],
                            "url": article_data["url"],
                            "error": error_msg,
                        }
                    )
                finally:
                    print("Cleaning up temporary file")
                    # Clean up the temporary file
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)

        return (
            jsonify(
                {
                    "message": f"Successfully processed {len(processed_articles)} articles",
                    "failed_articles": failed_articles,
                }
            ),
            200,
        )

    except requests.RequestException as e:
        return jsonify({"error": f"Failed to download JSON file: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid JSON data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Prompt Management Endpoints
@app.route("/prompts/default", methods=["GET"])
def get_default_prompt():
    """Get the default prompt content."""
    try:
        # Get the active prompt from database
        active_prompt = prompt_model.get_active_prompt()

        if active_prompt:
            return (
                jsonify(
                    {
                        "prompt": active_prompt.content,
                        "name": active_prompt.name,
                    }
                ),
                200,
            )
        else:
            # Return the fallback prompt if no active prompt exists
            fallback_prompt = prompt_model.get_default_prompt()
            return (
                jsonify(
                    {
                        "prompt": fallback_prompt,
                        "name": "EMPRAD 2025 Default Assistant (Fallback)",
                    }
                ),
                200,
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/prompts/default", methods=["PUT"])
def update_default_prompt():
    """Update the default prompt content."""
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing required field: prompt"}), 400

    try:
        # Get the currently active prompt
        active_prompt = prompt_model.get_active_prompt()

        if active_prompt:
            # Update the existing active prompt
            active_prompt.content = data["prompt"]
            active_prompt.name = data.get("name", active_prompt.name)
            active_prompt.description = data.get(
                "description", active_prompt.description
            )
            active_prompt.version = data.get("version", active_prompt.version)

            # Get the prompt ID from the database
            # We need to find the prompt by name since we don't have the ID
            db_prompt = prompt_model.collection.find_one(
                {"name": active_prompt.name, "is_active": True}
            )
            if db_prompt:
                prompt_id = str(db_prompt["_id"])
                # Update in database
                if prompt_model.update(prompt_id, active_prompt):
                    return (
                        jsonify(
                            {
                                "message": "Default prompt updated successfully",
                                "id": prompt_id,
                                "name": active_prompt.name,
                                "version": active_prompt.version,
                                "prompt": active_prompt.content,
                            }
                        ),
                        200,
                    )
                else:
                    return jsonify({"error": "Failed to update prompt"}), 500
            else:
                return jsonify({"error": "Active prompt not found in database"}), 404
        else:
            # Create a new active prompt if none exists
            new_prompt = Prompt(
                name=data.get("name", "EMPRAD 2025 Default Assistant"),
                content=data["prompt"],
                description=data.get(
                    "description", "Default prompt for the EMPRAD 2025 AI Assistant"
                ),
                is_active=True,
                version=data.get("version", "1.0"),
            )

            prompt_id = prompt_model.create(new_prompt)
            return (
                jsonify(
                    {
                        "message": "Default prompt created successfully",
                        "id": prompt_id,
                        "name": new_prompt.name,
                        "version": new_prompt.version,
                        "prompt": new_prompt.content,
                    }
                ),
                201,
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # This is for local development only.
    # Gunicorn will run the app in production.
    app.run(debug=True, host="0.0.0.0", port=5000)
