import chromadb
from chromadb.config import Settings
from typing import List, Dict
import os


class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents"):
        self.client = chromadb.Client(
            Settings(persist_directory="data/chroma", is_persistent=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def store_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict] = None,
    ):
        """Store text chunks and their embeddings in the vector store."""
        if metadata is None:
            metadata = [{"source": "pdf"} for _ in texts]

        ids = [str(i) for i in range(len(texts))]

        self.collection.add(
            embeddings=embeddings, documents=texts, metadatas=metadata, ids=ids
        )

    def search_similar(
        self, query_embedding: List[float], n_results: int = 5
    ) -> List[Dict]:
        """Search for similar documents using a query embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )

        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
