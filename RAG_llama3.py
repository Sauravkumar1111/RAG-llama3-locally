import os
import tempfile
from typing import List, Tuple
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

class DocumentQA:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.history = []
        self.persist_directory = "chroma_db"
        
        # Initialize Chroma client with new API
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
    def load_documents(self, file_paths: List[str]):
        """Load documents from various file types"""
        documents = []
        for file_path in file_paths:
            if file_path.endswith('.csv'):
                loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
                documents.extend(loader.load())
            elif file_path.endswith('.pdf'):
                try:
                    # Try structured loader first
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                except:
                    # Fall back to unstructured loader for scanned PDFs
                    loader = UnstructuredPDFLoader(file_path)
                    documents.extend(loader.load())
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    
    def initialize_vectorstore(self, documents: List[Document], collection_name: str = "documents", force_recreate: bool = False):
        """Initialize or load Chroma vector store using new API"""
        # Delete collection if force recreate is requested
        if force_recreate:
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass  # Collection might not exist
        
        # Create LangChain Chroma vectorstore
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            client=self.chroma_client,
            persist_directory=self.persist_directory
        )
        
        # Add documents if we're recreating or if collection is empty
        if force_recreate or vectorstore._collection.count() == 0:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
        
        return vectorstore
    
    def load_llm(self, model_type: str, model_name: str, openrouter_api_key: str = None, temperature: float = 0):
        """Load LLM based on user selection"""
        if model_type == "Ollama":
            return ChatOllama(model=model_name, temperature=temperature)
        elif model_type == "OpenRouter":
            if not openrouter_api_key:
                raise ValueError("OpenRouter API key is required")
            
            base_url = "https://openrouter.ai/api/v1"
            return ChatOpenAI(
                model=model_name,
                openai_api_key=openrouter_api_key,
                openai_api_base=base_url,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def create_chain(self, vectorstore, llm):
        """Create conversational retrieval chain"""
        return ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type="stuff"
        )
    
    def conversational_chat(self, chain, user_input: str):
        """Generate response using the RAG system"""
        result = chain.invoke({"question": user_input, "chat_history": self.history})
        self.history.append((user_input, result["answer"]))
        return result["answer"], result.get("source_documents", [])
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
    
    def list_collections(self):
        """List all available collections"""
        return [col.name for col in self.chroma_client.list_collections()]
    
    def delete_collection(self, collection_name: str):
        """Delete a specific collection"""
        self.chroma_client.delete_collection(collection_name)

def main():
    # Initialize the QA system
    qa_system = DocumentQA()
    
    # Test file paths (replace with your actual file paths)
    test_files = [r"C:\Users\Saurav Kumar\Downloads\Shekhar__FDA__CV (2).pdf"]  # Update with your actual files
    
    try:
        # Load and process documents
        print("Loading documents...")
        documents = qa_system.load_documents(test_files)
        print(f"Loaded {len(documents)} document chunks")
        
        # Initialize vector store
        print("Initializing vector store...")
        vectorstore = qa_system.initialize_vectorstore(
            documents, 
            collection_name="test_collection",
            force_recreate=True
        )
        print("Vector store initialized successfully")
        
        # Initialize LLM (using Ollama by default)
        print("Loading LLM...")
        llm = qa_system.load_llm("Ollama", "llama3.2:latest")
        print("LLM loaded successfully")
        
        # Create conversation chain
        print("Creating conversation chain...")
        chain = qa_system.create_chain(vectorstore, llm)
        print("Chain created successfully")
        
        # Test conversation
        print("\n" + "="*50)
        print("Document QA System Test")
        print("="*50)
        print("Type 'exit' to quit the conversation")
        print("Type 'clear' to clear conversation history")
        print("="*50)
        
        while True:
            user_input = input("\nYour question: ")
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                qa_system.clear_history()
                print("Conversation history cleared")
                continue
            
            # Generate response
            print("Thinking...")
            try:
                response, sources = qa_system.conversational_chat(chain, user_input)
                print(f"Bot: {response}")
                
                if sources:
                    print("\nSources used:")
                    for i, source in enumerate(sources, 1):
                        print(f"{i}. {source.page_content[:100]}...")
                        print(f"   Document: {source.metadata.get('source', 'Unknown')}")
                        if 'page' in source.metadata:
                            print(f"   Page: {source.metadata['page']}")
            except Exception as e:
                print(f"Error generating response: {e}")
    
    except Exception as e:
        print(f"Error initializing system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()