from google import genai
from typing import List
import os
from dotenv import load_dotenv


class EmbeddingGenerator:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        client = genai.Client(api_key=api_key)
        self.model = client

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text chunks."""
        embeddings = []
        for text in texts:
            result = self.model.models.embed_content(
                model="text-embedding-004",
                contents=text,
            )

            for embedding in result.embeddings:
                embeddings.append(embedding.values)

        print("Embeddings generated")
        return embeddings
