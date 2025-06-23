from google import genai
from typing import List
import os
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI


class EmbeddingGenerator:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        client = genai.Client(api_key=api_key)
        self.model = client

        # Initialize OpenAI client for async embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.openai_client = AsyncOpenAI(api_key=openai_api_key)

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

    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text chunks using OpenAI and asyncio.TaskGroup."""

        async def generate_single_embedding(text: str) -> List[float]:
            """Generate embedding for a single text chunk."""
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return response.data[0].embedding

        # Use asyncio.TaskGroup to generate all embeddings concurrently
        embeddings = []
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(generate_single_embedding(text)) for text in texts]

        # Collect results from all tasks
        embeddings = [task.result() for task in tasks]

        print("Async embeddings generated")
        return embeddings
