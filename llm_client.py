import ollama
from typing import List, Dict, Any
from config import Config

class LLMClient:
    def __init__(self):
        self.client = ollama.Client(host=Config.OLLAMA_BASE_URL)
        self.model = Config.OLLAMA_MODEL
    
    def check_model_availability(self) -> bool:
        """Check if the specified model is available."""
        try:
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            return self.model in available_models
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Pull the model if it's not available."""
        try:
            self.client.pull(self.model)
            return True
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama."""
        try:
            # Create the full prompt with context
            full_prompt = self._create_rag_prompt(prompt, context)
            
            response = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                stream=False
            )
            
            return response['response']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_response_stream(self, prompt: str, context: str = ""):
        """Generate streaming response using Ollama."""
        try:
            full_prompt = self._create_rag_prompt(prompt, context)
            
            stream = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                stream=True
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a RAG prompt with context and question."""
        if context:
            prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. If the context doesn't contain 
enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer: """
        else:
            prompt = f"""You are a helpful assistant. Please answer the following question:

Question: {question}

Answer: """
        
        return prompt
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []