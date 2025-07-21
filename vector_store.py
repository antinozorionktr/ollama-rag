import os
import pickle
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from config import Config

class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=Config.VECTOR_STORE_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection("documents")
        except:
            self.collection = self.client.create_collection("documents")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Create unique IDs
        ids = [f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', i)}" 
               for i, doc in enumerate(documents)]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def similarity_search(self, query: str, k: int = Config.TOP_K_RESULTS) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        documents_with_scores = []
        if results['documents'] and results['documents'][0]:
            for i, (doc_text, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                doc = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                # ChromaDB returns distances, convert to similarity score
                similarity_score = 1 - distance
                documents_with_scores.append((doc, similarity_score))
        
        return documents_with_scores
    
    def get_all_sources(self) -> List[str]:
        """Get all unique sources in the vector store."""
        try:
            all_data = self.collection.get()
            sources = set()
            for metadata in all_data['metadatas']:
                sources.add(metadata.get('source', 'unknown'))
            return list(sources)
        except:
            return []
    
    def delete_by_source(self, source: str) -> None:
        """Delete all documents from a specific source."""
        try:
            # Get all documents with the specific source
            all_data = self.collection.get()
            ids_to_delete = []
            
            for i, metadata in enumerate(all_data['metadatas']):
                if metadata.get('source') == source:
                    ids_to_delete.append(all_data['ids'][i])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
        except Exception as e:
            print(f"Error deleting documents: {e}")
    
    def clear_all(self) -> None:
        """Clear all documents from the vector store."""
        try:
            self.client.delete_collection("documents")
            self.collection = self.client.create_collection("documents")
        except Exception as e:
            print(f"Error clearing vector store: {e}")