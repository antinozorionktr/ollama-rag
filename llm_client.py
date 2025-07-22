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
            models_response = self.client.list()
            print(f"Raw response type: {type(models_response)}")
            
            available_models = []
            
            # Handle ListResponse object from newer ollama versions
            if hasattr(models_response, 'models'):
                model_list = models_response.models
                print(f"Found {len(model_list)} models")
                
                for model in model_list:
                    # Handle Model objects
                    if hasattr(model, 'model'):
                        model_name = model.model
                        available_models.append(model_name)
                        
                        # Also add the base name without version tag
                        if ':' in model_name:
                            base_name = model_name.split(':')[0]
                            available_models.append(base_name)
                        
                        print(f"   Found model: {model_name}")
            
            # Fallback for older response format
            elif isinstance(models_response, dict) and 'models' in models_response:
                model_list = models_response['models']
                for model in model_list:
                    if isinstance(model, dict):
                        model_name = model.get('name') or model.get('model') or model.get('id')
                        if model_name:
                            available_models.append(model_name)
                            if ':' in model_name:
                                available_models.append(model_name.split(':')[0])
            
            print(f"Available models: {available_models}")
            print(f"Looking for: {self.model}")
            
            # Check if our model is available (exact match or base name match)
            return self.model in available_models
            
        except Exception as e:
            print(f"Error checking model availability: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
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
Use ALL the information from the context to provide a comprehensive and complete answer. 
If you find multiple pieces of relevant information, include all of them in your response.
Be thorough and detailed in your answer.

Context:
{context}

Question: {question}

Instructions: 
- Provide a complete answer using all relevant information from the context
- If listing items (like projects, skills, experience), include ALL items mentioned
- Be specific and include details, numbers, and descriptions where available
- If the context doesn't contain enough information, clearly state what's missing

Answer: """
        else:
            prompt = f"""You are a helpful assistant. Please answer the following question:

Question: {question}

Answer: """
        
        return prompt
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            models_response = self.client.list()
            available_models = []
            
            # Handle ListResponse object from newer ollama versions
            if hasattr(models_response, 'models'):
                model_list = models_response.models
                
                for model in model_list:
                    # Handle Model objects
                    if hasattr(model, 'model'):
                        model_name = model.model
                        available_models.append(model_name)
                        
                        # Also add the base name without version tag
                        if ':' in model_name:
                            base_name = model_name.split(':')[0]
                            if base_name not in available_models:
                                available_models.append(base_name)
            
            # Fallback for older response format
            elif isinstance(models_response, dict) and 'models' in models_response:
                model_list = models_response['models']
                for model in model_list:
                    if isinstance(model, dict):
                        model_name = model.get('name') or model.get('model') or model.get('id')
                        if model_name:
                            available_models.append(model_name)
                            if ':' in model_name:
                                base_name = model_name.split(':')[0]
                                if base_name not in available_models:
                                    available_models.append(base_name)
            
            return available_models
            
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []