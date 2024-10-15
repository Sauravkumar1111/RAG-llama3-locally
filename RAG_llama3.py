import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = Ollama(
        model="llama3:instruct",
        temperature=0
    )
    return llm
    
def conversational_chat(chain, user_input, history):
    # Using invoke instead of __call__ to interact with the chain
    result = chain.invoke({"question": user_input, "chat_history": history})
    history.append((user_input, result["answer"]))
    return result["answer"]

def main():
    file_path = 'Controls Framework - GFCF.csv'
    
    try:
        # Loading CSV data directly from the given file path
        loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()
        
        # Initializing the embeddings
        embeddings = HuggingFaceEmbeddings()

        # Create a FAISS vector store or load the existing one
        if not os.path.exists(DB_FAISS_PATH):
            db = FAISS.from_documents(data, embeddings)
            db.save_local(DB_FAISS_PATH)
        else:
            db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        
        llm = load_llm()
        
        # Create the conversational retrieval chain using FAISS retriever
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
        
        history = []
        print(f"Hello! You can now ask questions about {file_path}. Type 'exit' to quit.")
        
        while True:
            user_input = input("Your question: ")
            if user_input.lower() == "exit":
                break
            
            # Generate response using the RAG system
            response = conversational_chat(chain, user_input, history)
            print("Bot:", response)
    
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        print(traceback.format_exc()) 

if __name__ == "__main__":
    main()
