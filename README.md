# RAG ChatBot

A Retrieval-Augmented Generation (RAG) based chatbot that answers questions based on uploaded documents (PDFs, DOCX, TXT) and web URLs. Built with Python, Ollama, FastAPI, and Streamlit.

## Features

- 📄 **Document Upload**: Support for PDF, DOCX, and TXT files
- 🌐 **URL Processing**: Extract and process content from web URLs
- 🤖 **Local LLM**: Uses Ollama for local language model inference
- 🔍 **Vector Search**: Efficient similarity search using ChromaDB and sentence transformers
- 💬 **Interactive Chat**: User-friendly Streamlit interface
- 🚀 **FastAPI Backend**: RESTful API for all operations
- 📚 **Source Citations**: Shows which documents were used to generate answers
- 🗑️ **Knowledge Management**: Add, view, and delete sources from knowledge base

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │────│    FastAPI      │────│     Ollama      │
│   Frontend      │    │    Backend      │    │   (LLM Model)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              │
                       ┌─────────────────┐
                       │   ChromaDB      │
                       │ (Vector Store)  │
                       └─────────────────┘
```

## Prerequisites

Before running the project, ensure you have the following installed:

### 1. Python 3.8+
```bash
python --version
```

### 2. Ollama
Install Ollama from [https://ollama.ai](https://ollama.ai)

**On macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**On Windows:**
Download and install from the official website.

### 3. Start Ollama and Pull a Model
```bash
# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull llama2
# or for a smaller model:
ollama pull tinyllama
# or for a more capable model:
ollama pull mistral
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd rag-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file with your configurations
# The default settings should work for most setups
```

### 5. Create Required Directories
```bash
mkdir uploads
mkdir vector_store
```

## Running the Project

### Method 1: Run Both Services Separately

#### Terminal 1 - Start FastAPI Backend
```bash
python main.py
```
The API will be available at `http://localhost:8000`

#### Terminal 2 - Start Streamlit Frontend
```bash
streamlit run streamlit_app.py
```
The web interface will be available at `http://localhost:8501`

### Method 2: Using a Process Manager (Recommended)

Create a `run.py` file:
```python
import subprocess
import sys
import time
import threading

def run_fastapi():
    subprocess.run([sys.executable, "main.py"])

def run_streamlit():
    time.sleep(3)  # Wait for FastAPI to start
    subprocess.run(["streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(target=run_fastapi)
    api_thread.daemon = True
    api_thread.start()
    
    # Start Streamlit in main thread
    run_streamlit()
```

Then run:
```bash
python run.py
```

## Usage

### 1. Upload Documents
- Open the Streamlit interface at `http://localhost:8501`
- Use the sidebar to upload PDF, DOCX, or TXT files
- Or add content from URLs

### 2. Ask Questions
- Type your questions in the chat interface
- The system will search through your uploaded documents
- Get answers with source citations

### 3. Manage Knowledge Base
- View all uploaded sources in the sidebar
- Delete specific sources or clear the entire knowledge base
- Monitor system status and model availability

## API Endpoints

The FastAPI backend provides the following endpoints:

- `GET /` - Health check
- `GET /status` - System status
- `POST /upload` - Upload document file
- `POST /add-url` - Add content from URL
- `POST /query` - Query the RAG system
- `POST /query-stream` - Streaming query response
- `GET /knowledge-base` - Get knowledge base info
- `DELETE /source` - Delete specific source
- `DELETE /knowledge-base` - Clear all sources

API documentation is available at `http://localhost:8000/docs` when the server is running.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama2` | Ollama model name |
| `VECTOR_STORE_PATH` | `./vector_store` | ChromaDB storage path |
| `CHUNK_SIZE` | `1000` | Text chunk size for processing |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `API_HOST` | `0.0.0.0` | FastAPI host |
| `API_PORT` | `8000` | FastAPI port |
| `MAX_FILE_SIZE` | `10485760` | Max upload file size (10MB) |
| `UPLOAD_DIR` | `./uploads` | Temporary upload directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TOP_K_RESULTS` | `5` | Number of similar documents to retrieve |

### Supported File Types
- **PDF**: `.pdf`
- **Word Document**: `.docx`
- **Text File**: `.txt`

### Supported Ollama Models
- `llama2` (default)
- `mistral`
- `codellama`
- `tinyllama` (lightweight)
- Any other model available in Ollama

## Troubleshooting

### Common Issues

#### 1. Ollama Model Not Available
```bash
# Pull the model
ollama pull llama2

# Check available models
ollama list
```

#### 2. API Connection Error
- Ensure FastAPI is running on `http://localhost:8000`
- Check if ports are not blocked by firewall
- Verify Ollama is running on `http://localhost:11434`

#### 3. Memory Issues
- Use a smaller model like `tinyllama`
- Reduce `CHUNK_SIZE` in environment variables
- Reduce `TOP_K_RESULTS` for fewer retrieved documents

#### 4. File Upload Issues
- Check file size (default max: 10MB)
- Ensure file format is supported
- Verify `uploads` directory exists and is writable

#### 5. Vector Store Issues
```bash
# Clear and recreate vector store
rm -rf vector_store
mkdir vector_store
```

### Performance Optimization

1. **Use GPU with Ollama** (if available):
   ```bash
   # Ollama automatically uses GPU if available
   ollama serve
   ```

2. **Adjust chunk size** for your documents:
   - Larger chunks: Better context, slower processing
   - Smaller chunks: Faster processing, less context

3. **Choose appropriate model**:
   - `tinyllama`: Fast, less accurate
   - `llama2`: Balanced
   - `mistral`: More accurate, slower

## Project Structure

```
rag-chatbot/
├── main.py                 # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── config.py              # Configuration settings
├── document_processor.py   # Document processing utilities
├── vector_store.py        # Vector store operations
├── llm_client.py          # Ollama client
├── rag_service.py         # Main RAG service
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── uploads/              # Temporary file uploads
└── vector_store/         # ChromaDB storage
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai) for local LLM inference
- [LangChain](https://langchain.com) for document processing
- [ChromaDB](https://www.trychroma.com) for vector storage
- [Sentence Transformers](https://www.sbert.net) for embeddings
- [FastAPI](https://fastapi.tiangolo.com) for the backend API
- [Streamlit](https://streamlit.io) for the frontend interface