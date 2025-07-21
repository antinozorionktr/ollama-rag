from typing import List, Dict, Any, Optional
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_client import LLMClient
from config import Config

class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()
    
    def add_document_from_file(self, file_path: str) -> Dict[str, Any]:
        """Add document from file to the knowledge base."""
        try:
            # Process the file
            documents = self.document_processor.process_file(file_path)
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            return {
                "success": True,
                "message": f"Successfully processed {len(documents)} chunks from file",
                "chunks_count": len(documents),
                "source": file_path
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing file: {str(e)}",
                "chunks_count": 0,
                "source": file_path
            }
    
    def add_document_from_url(self, url: str) -> Dict[str, Any]:
        """Add document from URL to the knowledge base."""
        try:
            # Process the URL
            documents = self.document_processor.process_url(url)
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            return {
                "success": True,
                "message": f"Successfully processed {len(documents)} chunks from URL",
                "chunks_count": len(documents),
                "source": url
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing URL: {str(e)}",
                "chunks_count": 0,
                "source": url
            }
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Query the RAG system."""
        try:
            # Search for relevant documents
            relevant_docs = self.vector_store.similarity_search(
                question, 
                k=Config.TOP_K_RESULTS
            )
            
            if not relevant_docs:
                return {
                    "success": True,
                    "answer": "I don't have any relevant information to answer your question. Please add some documents first.",
                    "sources": [],
                    "context_used": ""
                }
            
            # Prepare context from relevant documents
            context = self._prepare_context(relevant_docs)
            
            # Generate answer using LLM
            answer = self.llm_client.generate_response(question, context)
            
            # Prepare sources information
            sources = []
            if include_sources:
                sources = self._extract_sources(relevant_docs)
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "context_used": context[:500] + "..." if len(context) > 500 else context
            }
        except Exception as e:
            return {
                "success": False,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "context_used": ""
            }
    
    def query_stream(self, question: str):
        """Query the RAG system with streaming response."""
        try:
            # Search for relevant documents
            relevant_docs = self.vector_store.similarity_search(
                question, 
                k=Config.TOP_K_RESULTS
            )
            
            if not relevant_docs:
                yield "I don't have any relevant information to answer your question. Please add some documents first."
                return
            
            # Prepare context from relevant documents
            context = self._prepare_context(relevant_docs)
            
            # Generate streaming answer using LLM
            for chunk in self.llm_client.generate_response_stream(question, context):
                yield chunk
                
        except Exception as e:
            yield f"Error processing query: {str(e)}"
    
    def _prepare_context(self, relevant_docs: List[tuple]) -> str:
        """Prepare context from relevant documents."""
        context_parts = []
        for doc, score in relevant_docs:
            context_parts.append(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}\n")
        
        return "\n---\n".join(context_parts)
    
    def _extract_sources(self, relevant_docs: List[tuple]) -> List[Dict[str, Any]]:
        """Extract source information from relevant documents."""
        sources = []
        seen_sources = set()
        
        for doc, score in relevant_docs:
            source = doc.metadata.get('source', 'Unknown')
            if source not in seen_sources:
                sources.append({
                    "source": source,
                    "relevance_score": round(score, 3),
                    "chunk_info": f"Chunk {doc.metadata.get('chunk_id', 0) + 1} of {doc.metadata.get('total_chunks', 1)}"
                })
                seen_sources.add(source)
        
        return sources
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the current knowledge base."""
        sources = self.vector_store.get_all_sources()
        return {
            "total_sources": len(sources),
            "sources": sources
        }
    
    def delete_source(self, source: str) -> Dict[str, Any]:
        """Delete a specific source from the knowledge base."""
        try:
            self.vector_store.delete_by_source(source)
            return {
                "success": True,
                "message": f"Successfully deleted source: {source}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error deleting source: {str(e)}"
            }
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clear the entire knowledge base."""
        try:
            self.vector_store.clear_all()
            return {
                "success": True,
                "message": "Successfully cleared knowledge base"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error clearing knowledge base: {str(e)}"
            }
    
    def check_system_status(self) -> Dict[str, Any]:
        """Check the status of the RAG system."""
        model_available = self.llm_client.check_model_availability()
        
        return {
            "ollama_model": Config.OLLAMA_MODEL,
            "model_available": model_available,
            "embedding_model": Config.EMBEDDING_MODEL,
            "vector_store_path": Config.VECTOR_STORE_PATH,
            "knowledge_base": self.get_knowledge_base_info()
        }