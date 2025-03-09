"""
Vector database functionality with Pinecone (VoyageAI Version)
"""

import os
from pinecone import Pinecone, ServerlessSpec
from voyageai import Client as VoyageClient
from typing import List, Dict, Any
from pinecone.exceptions import NotFoundException


class VectorDB:
    def __init__(self, index_name: str = "deep-research", dimension: int = 1024):
        """
        Initialize Pinecone vector database with Voyage embeddings
        """
        # Initialize Voyage embeddings
        self.voyage = VoyageClient(api_key=os.getenv("VOYAGE_API_KEY"))
        self.embed_model = "voyage-2"  # Latest Voyage model
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Delete existing index if dimensions mismatch
        if index_name in self.pc.list_indexes().names():
            index = self.pc.Index(index_name)
            if index.describe_index_stats().dimension != dimension:
                self.pc.delete_index(index_name)
        
        # Create new index with correct dimensions
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
        self.index = self.pc.Index(index_name)
        self.namespace = "main"
        self.document_map = {}

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents with Voyage embeddings
        """
        if not documents:
            return []
            
        # Generate embeddings
        texts = [doc["content"] for doc in documents]
        embeddings = self.voyage.embed(
            texts=texts,
            model=self.embed_model,
            input_type="document"
        ).embeddings
        
        # Prepare vectors with namespace
        vectors = []
        doc_ids = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{i}_{hash(doc['url'])}"
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": {
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "content": doc["content"][:4000]  # Store first 4k chars
                }
            })
            doc_ids.append(doc_id)
            self.document_map[doc_id] = doc
            
        # Upsert with namespace
        self.index.upsert(
            vectors=vectors,
            namespace=self.namespace
        )
        return doc_ids

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search with Voyage embeddings
        """
        # Embed query
        query_embedding = self.voyage.embed(
            texts=[query],
            model=self.embed_model,
            input_type="query"
        ).embeddings[0]
        
        # Query with namespace
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=n_results,
                include_metadata=True,
                namespace=self.namespace
            )
        except NotFoundException:
            return []
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
                "content": match.metadata.get("content", "")
            }
            for match in results.matches
        ]

    def clear(self):
        """Safe clear with namespace handling"""
        try:
            self.index.delete(
                delete_all=True,
                namespace=self.namespace
            )
        except NotFoundException:
            pass  # Namespace already empty
        self.document_map = {}