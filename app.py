import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from RAG_llama3 import DocumentQA

# Page configuration
st.set_page_config(
    page_title="Document QA System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "qa_system" not in st.session_state:
    st.session_state.qa_system = DocumentQA()
if "vectorstore_initialized" not in st.session_state:
    st.session_state.vectorstore_initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Available models
OLLAMA_MODELS = [
    "llama3.2:latest", 
    "llama2", 
    "mistral", 
    "codellama", 
    "phi3"
]

OPENROUTER_MODELS = [
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "google/gemini-pro-1.5",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "mistralai/mistral-7b-instruct",
    "huggingfaceh4/zephyr-7b-beta"
]

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents", 
        type=["csv", "pdf"],
        accept_multiple_files=True
    )
    
    # Collection name
    collection_name = st.text_input("Collection Name", value="documents")
    
    # Model selection
    model_type = st.radio(
        "Select model type",
        ["Ollama", "OpenRouter"]
    )
    
    if model_type == "Ollama":
        model_name = st.selectbox("Select model", OLLAMA_MODELS)
        openrouter_api_key = None
    else:
        model_name = st.selectbox("Select model", OPENROUTER_MODELS)
        openrouter_api_key = st.text_input(
            "OpenRouter API Key", 
            type="password",
            help="Get your API key from https://openrouter.ai"
        )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    
    # Initialize button
    if st.button("Initialize System"):
        if not uploaded_files:
            st.error("Please upload at least one document")
        else:
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files temporarily
                    temp_files = []
                    for uploaded_file in uploaded_files:
                        temp_dir = tempfile.mkdtemp()
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_files.append(file_path)
                    
                    # Load and process documents
                    documents = st.session_state.qa_system.load_documents(temp_files)
                    
                    # Initialize vector store
                    vectorstore = st.session_state.qa_system.initialize_vectorstore(
                        documents, 
                        collection_name=collection_name,
                        force_recreate=True
                    )
                    
                    # Initialize LLM
                    llm = st.session_state.qa_system.load_llm(
                        model_type, 
                        model_name, 
                        openrouter_api_key,
                        temperature
                    )
                    
                    # Create chain
                    st.session_state.chain = st.session_state.qa_system.create_chain(vectorstore, llm)
                    st.session_state.vectorstore_initialized = True
                    st.session_state.current_collection = collection_name
                    
                    # Clean up temp files
                    for file_path in temp_files:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                    
                    st.success("System initialized successfully!")
                    
                except Exception as e:
                    st.error(f"Error initializing system: {str(e)}")
    
    if st.button("Clear Conversation"):
        st.session_state.qa_system.clear_history()
        st.session_state.messages = []
        st.rerun()

    # Collection management section
    st.divider()
    st.subheader("Collection Management")
    
    # List available collections
    try:
        collections = st.session_state.qa_system.list_collections()
        if collections:
            st.write("Available collections:")
            for col in collections:
                st.write(f"- {col}")
        else:
            st.write("No collections available")
    except:
        st.write("No collections available")
    
    # Delete collection option
    col_to_delete = st.selectbox(
        "Select collection to delete",
        options=collections if 'collections' in locals() and collections else ["No collections available"]
    )
    
    if st.button("Delete Selected Collection") and col_to_delete != "No collections available":
        try:
            st.session_state.qa_system.delete_collection(col_to_delete)
            st.success(f"Collection '{col_to_delete}' deleted successfully!")
            if st.session_state.vectorstore_initialized and st.session_state.current_collection == col_to_delete:
                st.session_state.vectorstore_initialized = False
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")

# Main chat interface
st.title("📚 Document QA System with ChromaDB")

# Display information about the system
with st.expander("About this system", expanded=False):
    st.markdown("""
    This Document QA System allows you to:
    - Upload CSV and PDF documents
    - Ask questions about your documents
    - Choose between local models (Ollama) or cloud models (OpenRouter)
    - View the sources used to generate answers
    
    **How to use:**
    1. Upload one or more documents (CSV/PDF)
    2. Configure your model preferences
    3. Click 'Initialize System'
    4. Start asking questions about your documents
    """)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View sources"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"Source {i+1}:")
                    st.info(source.page_content)
                    st.caption(f"Document: {source.metadata.get('source', 'Unknown')}")

# Display status
if not st.session_state.vectorstore_initialized:
    st.info("Please upload documents and initialize the system using the sidebar.")
else:
    st.success("System ready! You can now ask questions about your documents.")

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    if not st.session_state.vectorstore_initialized:
        st.error("Please initialize the system first by uploading documents and clicking 'Initialize System'")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, sources = st.session_state.qa_system.conversational_chat(
                        st.session_state.chain, prompt
                    )
                    
                    # Display assistant response
                    st.markdown(response)
                    
                    # Display sources if available
                    if sources:
                        with st.expander("View sources"):
                            for i, source in enumerate(sources):
                                st.write(f"Source {i+1}:")
                                st.info(source.page_content)
                                st.caption(f"Document: {source.metadata.get('source', 'Unknown')} - Page: {source.metadata.get('page', 'N/A')}")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")