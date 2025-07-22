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
            # Search for relevant documents with higher retrieval count
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
            
            # For better accuracy, also search with related keywords
            additional_searches = []
            question_lower = question.lower()
            
            # Add specific searches for common queries
            if any(word in question_lower for word in ['project', 'projects', 'work']):
                additional_docs = self.vector_store.similarity_search("projects experience work", k=5)
                additional_searches.extend(additional_docs)
            
            if any(word in question_lower for word in ['skill', 'skills', 'technology', 'technologies']):
                additional_docs = self.vector_store.similarity_search("skills technologies programming", k=5)
                additional_searches.extend(additional_docs)
            
            if any(word in question_lower for word in ['experience', 'years', 'career']):
                additional_docs = self.vector_store.similarity_search("experience years career", k=5)
                additional_searches.extend(additional_docs)
            
            # Combine and deduplicate results
            all_docs = relevant_docs + additional_searches
            seen_content = set()
            unique_docs = []
            
            for doc, score in all_docs:
                content_key = doc.page_content[:100]  # Use first 100 chars as key
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_docs.append((doc, score))
            
            # Take top results after deduplication
            final_docs = unique_docs[:min(15, len(unique_docs))]
            
            # Prepare context from relevant documents
            context = self._prepare_context(final_docs)
            
            # Generate answer using LLM
            answer = self.llm_client.generate_response(question, context)
            
            # Prepare sources information
            sources = []
            if include_sources:
                sources = self._extract_sources(final_docs[:Config.TOP_K_RESULTS])
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "context_used": context[:1000] + "..." if len(context) > 1000 else context
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
        total_length = 0
        max_context_length = getattr(Config, 'MAX_CONTEXT_LENGTH', 3000)
        
        for doc, score in relevant_docs:
            doc_text = f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}\n"
            
            # Check if adding this would exceed max length
            if total_length + len(doc_text) > max_context_length:
                # Try to add a truncated version
                remaining_space = max_context_length - total_length - 100  # Leave some buffer
                if remaining_space > 200:  # Only add if meaningful amount of space left
                    truncated_text = doc_text[:remaining_space] + "...\n"
                    context_parts.append(truncated_text)
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
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