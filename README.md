# RAG-llama3-locally

A powerful document question-answering system that allows you to chat with your documents using local or cloud-based AI models.

---

## 🚀 How to Run the Application

### 1. Install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Install and set up Ollama (for local models):

```bash
# Install Ollama from https://ollama.ai
# Then pull the models you want to use:
ollama pull llama3:instruct
```

### 3. Run the Streamlit application:

```bash
streamlit run app.py
```

---

## ✨ Features

- 📄 **Multi-format Support**: Handles both CSV and PDF documents
- 🤖 **Flexible Model Selection**:
  - Local models via **Ollama**
  - Cloud models via **OpenRouter** (including free options)
- 💬 **User-Friendly Interface**: Streamlit-based GUI for easy interaction
- 📌 **Source Citation**: Displays which parts of documents were used to answer queries
- 🧹 **Conversation Management**: Clear chat history as needed
- 🎛️ **Temperature Control**: Adjust creativity of model responses

---

## 📒 Usage Notes

- For **OpenRouter models**, sign up at [https://openrouter.ai](https://openrouter.ai) to get an API key.
- Some OpenRouter models may have usage limits or costs.
- The system creates temporary files during processing but cleans them afterward.
- Processing **large documents** may take a few moments.

---

## 📦 Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Set up **Ollama** for local model inference:

```bash
ollama pull llama3:instruct
```

---

## ▶️ Usage

1. Run the app:

```bash
streamlit run app.py
```

2. Open your browser (usually at `http://localhost:8501`)
3. Upload one or more CSV or PDF documents
4. Select:
   - **Local (Ollama)** or **Cloud (OpenRouter)**
   - Desired model
   - Enter your OpenRouter API key (if applicable)
   - Adjust temperature (optional)
5. Click **Initialize System**
6. Start chatting with your documents!

---

## 🤖 Model Options

### Local Models (Ollama)

- `llama3:instruct`
- `llama2`
- `mistral`
- `codellama`
- `phi3`

### Cloud Models (OpenRouter)

- `meta-llama/llama-3-70b-instruct`
- `meta-llama/llama-3-8b-instruct`
- `google/gemini-pro-1.5`
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- *...and many more*

---

## 🔐 OpenRouter Setup

1. Sign up at [https://openrouter.ai](https://openrouter.ai)
2. Copy your API key from the dashboard
3. Paste the key into the app when prompted

> ⚠️ Some models may have limitations or paid usage

---

## 📂 File Requirements

### CSV Files

- UTF-8 encoded
- First row must contain headers

### PDF Files

- Must be text-based (not image scans)
- Digitally-created PDFs yield the best results

---

## 🧠 Technical Details

- Uses **FAISS** for vector similarity search
- Employs **HuggingFace Sentence Transformers** for embeddings
- Built using **LangChain** for orchestration
- UI powered by **Streamlit**

---

## 🛠️ Troubleshooting

### Common Issues

| Issue                        | Solution                                          |
|-----------------------------|---------------------------------------------------|
| `Ollama not found`          | Ensure Ollama is installed and running            |
| `Model not available`       | Use `ollama pull <model-name>`                   |
| `API key errors`            | Check your OpenRouter API key                    |
| `File processing failed`    | Ensure files are valid CSV or text-based PDFs     |

### Performance Notes

- Large documents may take longer to process
- First query might be slower due to model warm-up
- Cloud models often respond faster than local ones

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a PR.
<img width="1915" height="926" alt="Screenshot 2025-08-24 010727" src="https://github.com/user-attachments/assets/d3304731-04ea-4481-b569-f85266bba7a5" />

---

## 📚 Support & Documentation

- Ollama: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
- OpenRouter: [https://openrouter.ai/docs](https://openrouter.ai/docs)
- LangChain: [https://python.langchain.com/docs](https://python.langchain.com/docs)
