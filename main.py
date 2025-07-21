import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag_service import RAGService
from config import Config

# Create upload directory if it doesn't exist
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="RAG ChatBot API",
    description="A RAG-based chatbot API that answers questions based on uploaded documents and URLs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = RAGService()

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True

class URLRequest(BaseModel):
    url: str

class SourceDeleteRequest(BaseModel):
    source: str

class QueryResponse(BaseModel):
    success: bool
    answer: str
    sources: List[dict] = []
    context_used: str = ""

class UploadResponse(BaseModel):
    success: bool
    message: str
    chunks_count: int
    source: str

class SystemStatus(BaseModel):
    ollama_model: str
    model_available: bool
    embedding_model: str
    vector_store_path: str
    knowledge_base: dict

# API endpoints
@app.get("/")
async def root():
    return {"message": "RAG ChatBot API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get the current system status."""
    return rag_service.check_system_status()

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document file."""
    
    # Check file size
    if file.size > Config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File size exceeds maximum allowed size of {Config.MAX_FILE_SIZE} bytes"
        )
    
    # Check file extension
    allowed_extensions = ['.pdf', '.docx', '.txt']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file
        result = rag_service.add_document_from_file(file_path)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    
    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/add-url", response_model=UploadResponse)
async def add_url(request: URLRequest):
    """Add and process content from a URL."""
    try:
        result = rag_service.add_document_from_url(request.url)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = rag_service.query(request.question, request.include_sources)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["answer"])
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query-stream")
async def query_stream(request: QueryRequest):
    """Query the RAG system with streaming response."""
    try:
        def generate():
            for chunk in rag_service.query_stream(request.question):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing streaming query: {str(e)}")

@app.get("/knowledge-base")
async def get_knowledge_base_info():
    """Get information about the current knowledge base."""
    return rag_service.get_knowledge_base_info()

@app.delete("/source")
async def delete_source(request: SourceDeleteRequest):
    """Delete a specific source from the knowledge base."""
    try:
        result = rag_service.delete_source(request.source)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting source: {str(e)}")

@app.delete("/knowledge-base")
async def clear_knowledge_base():
    """Clear the entire knowledge base."""
    try:
        result = rag_service.clear_knowledge_base()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing knowledge base: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )