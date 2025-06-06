from datetime import datetime
from typing import List
from bson import ObjectId
from flask_pymongo import PyMongo


class Article:
    def __init__(self, title: str, url: str, embeddings: List[float], content: str):
        self.title = title
        self.url = url
        self.embeddings = embeddings
        self.content = content
        self.created_at = datetime.today()
        self.updated_at = datetime.today()

    def to_dict(self):
        return {
            "title": self.title,
            "url": self.url,
            "embeddings": self.embeddings,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: dict):
        article = Article(
            title=data["title"],
            url=data["url"],
            embeddings=data["embeddings"],
            content=data["content"],
        )
        article.created_at = data.get("created_at", datetime.today())
        article.updated_at = data.get("updated_at", datetime.today())
        return article


class ArticleModel:
    def __init__(self, mongo: PyMongo):
        if mongo is None:
            raise ValueError("MongoDB connection is not properly initialized")
        self.mongo = mongo
        self.collection = mongo.db.emprad_articles

    def create(self, article: Article) -> str:
        """Create a new article in the database."""
        if self.collection is None:
            raise ValueError("Articles collection is not properly initialized")
        result = self.collection.insert_one(article.to_dict())
        return str(result.inserted_id)

    def get_by_id(self, article_id: str) -> Article:
        """Get an article by its ID."""
        if self.collection is None:
            raise ValueError("Articles collection is not properly initialized")
        result = self.collection.find_one({"_id": ObjectId(article_id)})
        if result:
            return Article.from_dict(result)
        return None

    def get_all(self) -> List[Article]:
        """Get all articles."""
        if self.collection is None:
            raise ValueError("Articles collection is not properly initialized")
        results = self.collection.find()
        return [Article.from_dict(result) for result in results]

    def update(self, article_id: str, article: Article) -> bool:
        """Update an article."""
        if self.collection is None:
            raise ValueError("Articles collection is not properly initialized")
        article_dict = article.to_dict()
        article_dict["updated_at"] = datetime.today()
        result = self.collection.update_one(
            {"_id": ObjectId(article_id)}, {"$set": article_dict}
        )
        return result.modified_count > 0

    def delete(self, article_id: str) -> bool:
        """Delete an article."""
        if self.collection is None:
            raise ValueError("Articles collection is not properly initialized")
        result = self.collection.delete_one({"_id": ObjectId(article_id)})
        return result.deleted_count > 0

    def search_by_embedding(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Article]:
        """Search articles by embedding similarity."""
        if self.collection is None:
            raise ValueError("Articles collection is not properly initialized")
        # Note: This is a simple implementation. For production, you might want to use
        # MongoDB's vector search capabilities or a dedicated vector database
        pipeline = [
            {
                "$addFields": {
                    "similarity": {
                        "$reduce": {
                            "input": {"$range": [0, {"$size": "$embeddings"}]},
                            "initialValue": 0,
                            "in": {
                                "$add": [
                                    "$$value",
                                    {
                                        "$multiply": [
                                            {"$arrayElemAt": ["$embeddings", "$$this"]},
                                            {
                                                "$arrayElemAt": [
                                                    query_embedding,
                                                    "$$this",
                                                ]
                                            },
                                        ]
                                    },
                                ]
                            },
                        }
                    }
                }
            },
            {"$sort": {"similarity": -1}},
            {"$limit": limit},
        ]

        results = self.collection.aggregate(pipeline)
        return [Article.from_dict(result) for result in results]
