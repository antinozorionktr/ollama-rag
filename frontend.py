import streamlit as st
import requests
import time
from typing import List, Dict
import json

# Configuration
API_BASE_URL = "http://localhost:8000"

# Set page config
st.set_page_config(
    page_title="RAG ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #1e3a5f;
        color: #ffffff;
        border-left: 4px solid #1f77b4;
    }
    .bot-message {
        background-color: #2e2e2e;
        color: #ffffff;
        border-left: 4px solid #4caf50;
    }
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = []

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_status():
    """Get system status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def upload_file(file):
    """Upload file to API."""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=60)
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}

def add_url(url):
    """Add URL to knowledge base."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/add-url", 
            json={"url": url}, 
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}

def query_rag(question, include_sources=True):
    """Query the RAG system."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "include_sources": include_sources},
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"success": False, "answer": f"Error: {str(e)}", "sources": []}

def get_knowledge_base_info():
    """Get knowledge base information."""
    try:
        response = requests.get(f"{API_BASE_URL}/knowledge-base", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"total_sources": 0, "sources": []}
    except:
        return {"total_sources": 0, "sources": []}

def delete_source(source):
    """Delete a source from knowledge base."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/source",
            json={"source": source},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}

def clear_knowledge_base():
    """Clear the entire knowledge base."""
    try:
        response = requests.delete(f"{API_BASE_URL}/knowledge-base", timeout=30)
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}

# Main app
def main():
    st.markdown('<h1 class="main-header">🤖 RAG ChatBot</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the FastAPI server first.")
        st.code("python main.py")
        return
    
    # Sidebar for configuration and knowledge base management
    with st.sidebar:
        st.header("📚 Knowledge Base Management")
        
        # System status
        with st.expander("🔧 System Status"):
            status = get_system_status()
            if status:
                st.success("✅ System is running")
                st.write(f"**Model:** {status['ollama_model']}")
                st.write(f"**Model Available:** {'✅' if status['model_available'] else '❌'}")
                st.write(f"**Embedding Model:** {status['embedding_model']}")
                
                if not status['model_available']:
                    st.warning("⚠️ Ollama model is not available. Please ensure Ollama is running and the model is installed.")
                    st.code(f"ollama pull {status['ollama_model']}")
            else:
                st.error("❌ Cannot get system status")
        
        # File upload
        st.subheader("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_file is not None:
            if st.button("Upload Document"):
                with st.spinner("Processing document..."):
                    result = upload_file(uploaded_file)
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.info(f"Added {result['chunks_count']} chunks to knowledge base")
                        st.session_state.knowledge_base = get_knowledge_base_info()
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
        
        # URL input
        st.subheader("🌐 Add URL")
        url_input = st.text_input(
            "Enter URL",
            placeholder="https://example.com/article",
            help="Add content from a webpage"
        )
        
        if url_input and st.button("Add URL"):
            with st.spinner("Processing URL..."):
                result = add_url(url_input)
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    st.info(f"Added {result['chunks_count']} chunks to knowledge base")
                    st.session_state.knowledge_base = get_knowledge_base_info()
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
        
        # Knowledge base info
        st.subheader("📊 Current Knowledge Base")
        kb_info = get_knowledge_base_info()
        st.session_state.knowledge_base = kb_info
        
        if kb_info["total_sources"] > 0:
            st.write(f"**Total Sources:** {kb_info['total_sources']}")
            
            # Display sources with delete buttons
            for source in kb_info["sources"]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Truncate long source names
                    display_source = source if len(source) <= 40 else source[:37] + "..."
                    st.write(f"📎 {display_source}")
                with col2:
                    if st.button("🗑️", key=f"delete_{source}", help="Delete this source"):
                        with st.spinner("Deleting..."):
                            result = delete_source(source)
                            if result["success"]:
                                st.success("✅ Source deleted")
                                st.rerun()
                            else:
                                st.error(f"❌ {result['message']}")
            
            # Clear all button
            if st.button("🗑️ Clear All", type="secondary"):
                if st.session_state.get("confirm_clear", False):
                    with st.spinner("Clearing knowledge base..."):
                        result = clear_knowledge_base()
                        if result["success"]:
                            st.success("✅ Knowledge base cleared")
                            st.session_state.knowledge_base = {"total_sources": 0, "sources": []}
                            st.session_state.confirm_clear = False
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Click again to confirm clearing all data")
        else:
            st.info("No documents in knowledge base. Upload files or add URLs to get started.")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message bot-message"><strong>Bot:</strong> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(
                                f'<div class="source-box">'
                                f'<strong>Source {i}:</strong> {source["source"]}<br>'
                                f'<small>Relevance: {source["relevance_score"]} | {source["chunk_info"]}</small>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
        
        # Chat input
        question = st.chat_input("Ask a question about your documents...")
        
        if question:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Show user message immediately
            st.markdown(
                f'<div class="chat-message user-message"><strong>You:</strong> {question}</div>',
                unsafe_allow_html=True
            )
            
            # Get bot response
            with st.spinner("Thinking..."):
                response = query_rag(question)
                
                if response["success"]:
                    # Add bot message
                    bot_message = {
                        "role": "bot", 
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    }
                    st.session_state.messages.append(bot_message)
                    
                    # Display bot response
                    st.markdown(
                        f'<div class="chat-message bot-message"><strong>Bot:</strong> {response["answer"]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Display sources
                    if response.get("sources"):
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(
                                    f'<div class="source-box">'
                                    f'<strong>Source {i}:</strong> {source["source"]}<br>'
                                    f'<small>Relevance: {source["relevance_score"]} | {source["chunk_info"]}</small>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                else:
                    error_message = {"role": "bot", "content": f"Error: {response['answer']}"}
                    st.session_state.messages.append(error_message)
                    st.error(f"❌ {response['answer']}")
            
            st.rerun()
    
    with col2:
        st.header("🎛️ Controls")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Settings
        with st.expander("⚙️ Settings"):
            include_sources = st.checkbox("Include Sources", value=True)
        
        # Instructions
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
            2. **Add URLs**: Enter webpage URLs to extract content
            3. **Ask Questions**: Type questions in the chat input
            4. **View Sources**: Click on "Sources" to see which documents were used
            5. **Manage Knowledge Base**: Delete specific sources or clear all data
            """)
        
        # Tips
        with st.expander("💡 Tips"):
            st.markdown("""
            - Ask specific questions about your documents
            - Try different phrasings if you don't get good results
            - Upload multiple related documents for better context
            - Check sources to verify the information
            - Clear chat history to start fresh conversations
            """)

if __name__ == "__main__":
    main()