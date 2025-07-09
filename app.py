# app.py
from flask import Flask, request, jsonify
from src.pdf_processor.processor import PDFProcessor
from src.embeddings.generator import EmbeddingGenerator
from src.vector_store.store import VectorStore
from src.models.article import Article, ArticleModel
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

        prompt = f"""### 🧠 Prompt para o Assistente de IA – EMPRAD 2025 (Formato OpenAI)

Você é o **Assistente EMPRAD**, uma inteligência artificial treinada para responder exclusivamente com base nos **artigos publicados no EMPRAD 2025 (Encontro de Empreendedorismo e Gestão para o Desenvolvimento)**.

Seu objetivo é **ajudar participantes do evento a encontrarem artigos relevantes** sobre temas de empreendedorismo, administração e negócios, trazendo respostas fundamentadas, confiáveis e referenciadas.

---

### 🎯 Regras de Comportamento

1. **Base de conhecimento restrita**:
    
    Você só pode responder com base no conteúdo dos artigos publicados no EMPRAD.
    
    - **Não use conhecimento externo** ou invente informações.
    - Se a base não contém resposta para a pergunta, diga:
        
        > "Não encontrei artigos no EMPRAD 2025 que abordem diretamente essa questão."
        > 
2. **Temas obrigatórios**:
    
    Aceite apenas perguntas relacionadas a:
    
    - Empreendedorismo
    - Startups
    - Inovação
    - Gestão e administração
    - Estratégia organizacional
    - Negócios de impacto
    - Sustentabilidade empresarial
    - Políticas públicas voltadas ao desenvolvimento
    - Finanças e investimentos em novos negócios
    - Educação empreendedora
    
    Se o tema estiver fora desse escopo, responda:
    
    > "Este assistente é voltado apenas a temas abordados no EMPRAD. Reformule sua pergunta com foco em empreendedorismo, negócios ou administração."
    > 
3. **Linguagem inadequada**:
    
    Não aceite perguntas com palavrões, ofensas ou termos depreciativos.
    
    Se detectar esse tipo de linguagem, responda:
    
    > "Sua pergunta contém termos inadequados. Reformule-a de forma respeitosa e dentro do escopo temático do evento."
    > 

---

### 🖼️ Formato da Resposta (Obrigatório)

Toda resposta deve seguir a estrutura abaixo:

1. **Resposta objetiva e resumida**:
    - Comece com uma breve explicação com base nos artigos encontrados.
    - Destaque as principais descobertas, abordagens metodológicas e implicações práticas observadas nos estudos.
2. **Lista de artigos encontrados**:
    
    Para cada artigo relevante, mostre as seguintes informações:
    
    - **Título completo**
    - **Autores e ano**
    - **Número da página**
    - **Trecho relevante** (curto e direto, com aspas)
    - **Palavras-chave** (de 2 a 4)
    - Botões ou links:
        - `Ver artigo` (link interno)
        - `Download`
        - `Copiar citação`
        - `Resumo`

---

### ✅ Exemplo de resposta ideal

> Com base na sua consulta sobre "O que é uma startup?", analisei os anais do EMPRAD e encontrei 3 artigos relevantes publicados entre 2023 e 2025.
> 
> 
> **Síntese das descobertas**: Os estudos apontam para a importância da contextualização das práticas de gestão e empreendedorismo à realidade brasileira, considerando as especificidades culturais, econômicas e sociais dos países em desenvolvimento.
> 
> **Abordagens metodológicas**: Análises comparativas entre diferentes regiões e setores, evidenciando padrões distintos entre startups urbanas e rurais.
> 
> **Implicações práticas**: A literatura recomenda que startups adotem práticas colaborativas e desenvolvam visão sistêmica para enfrentar ambientes de negócios dinâmicos e incertos.
>
        
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
                    return jsonify({"error": "Invalid URL provided"}), 400
            except Exception:
                return jsonify({"error": "Invalid URL provided"}), 400

            # Create a temporary file to store the PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                try:
                    print("taking break for 10 seconds...")
                    time.sleep(10)  # Pauses execution for 10 seconds
                    print("resuming...")
                    print(f"Processing article of index: {index}")
                    print(f"Processing article: {article_data['nomeTrabalho']}")
                    # Download the PDF
                    response = requests.get(article_data["url"], stream=True)
                    response.raise_for_status()  # Raise an exception for bad status codes

                    # Check if the content type is PDF
                    content_type = response.headers.get("content-type", "").lower()
                    if "application/pdf" not in content_type:
                        return (
                            jsonify({"error": "URL does not point to a PDF file"}),
                            400,
                        )

                    # Save the PDF to the temporary file
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    temp_file.flush()

                    print(f"Processing PDF: {article_data['nomeTrabalho']}")
                    # Process PDF
                    text_chunks = pdf_processor.process_pdf(temp_file.name)
                    if not text_chunks:
                        print("No text content extracted from PDF")

                    print(f"Generating embeddings for: {article_data['nomeTrabalho']}")
                    embeddings = asyncio.run(
                        embedding_generator.generate_embeddings_async(text_chunks)
                    )

                    # Create and save article in MongoDB
                    # Create an article for  each embedding
                    for index, embedding in enumerate(embeddings):
                        article = Article(
                            title=article_data["nomeTrabalho"],
                            url=article_data["url"],
                            embeddings=embedding,
                            content=text_chunks[index],
                        )
                        print(f"Saving article: {article_data['nomeTrabalho']}")
                        article_model.create(article)

                    processed_articles.append(
                        {
                            "title": article_data["nomeTrabalho"],
                            "url": article_data["url"],
                        }
                    )

                except requests.RequestException as e:
                    print(f"Error processing article: {article_data['nomeTrabalho']}")
                    print(f"Error: {str(e)}")
                except Exception as e:
                    print(f"Error processing article: {article_data['nomeTrabalho']}")
                    print(f"Error: {str(e)}")
                finally:
                    print("Cleaning up temporary file")
                    # Clean up the temporary file
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)

        return (
            jsonify(
                {
                    "message": f"Successfully processed {len(processed_articles)} articles",
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


if __name__ == "__main__":
    # This is for local development only.
    # Gunicorn will run the app in production.
    app.run(debug=True, host="0.0.0.0", port=5000)
