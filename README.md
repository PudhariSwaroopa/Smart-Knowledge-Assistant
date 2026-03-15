# 🧠 Smart Knowledge Assistant (PDF RAG)

A powerful **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit**, **LangChain**, and **Pinecone**.  
This assistant allows users to upload PDF documents and ask complex questions, providing accurate answers based on retrieved context using **Ollama (Llama 3)**.

---

# 🚀 Features

- **PDF Ingestion:** Automatically loads and processes PDF files from the `data/` directory.
- **Vector Search:** Uses **Pinecone** for high-speed vector retrieval.
- **Local LLM:** Powered by **Ollama (Llama 3)** for privacy and local processing.
- **Smart UI:** Interactive chat interface built with **Streamlit**.

---

# 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Framework | LangChain |
| Frontend | Streamlit |
| Vector Database | Pinecone |
| LLM | Ollama (Llama 3) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |

---

# 📋 Prerequisites

Before running the project, ensure you have:

1. **Python 3.12 (recommended)**  
   https://www.python.org/

2. **Ollama installed**  
   https://ollama.com/

3. **Pinecone API Key**  
   https://www.pinecone.io/

---

# ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Smart-Knowledge-Assistant.git
cd Smart-Knowledge-Assistant
```

---

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows (Git Bash)**

```bash
source venv/Scripts/activate
```

**Windows (CMD)**

```bash
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_api_key_here
```

---

### 5️⃣ Prepare Your Data

Place your PDF documents inside the `data/` folder.

Example:

```
data/
 └── CompleteAIML.pdf
```

---

### 6️⃣ Pull the LLM Model

```bash
ollama pull llama3
```

---

# 🏃 Running the Application

Start the Streamlit app:

```bash
python -m streamlit run app.py
```

Open the browser:

```
http://localhost:8501
```

---

# 📂 Project Structure

```
Smart-Knowledge-Assistant
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── CompleteAIML.pdf
├── smart-knowledge-assistant.ipynb
└── .env
```

---

# 🧠 Architecture

```
User Question
      ↓
Streamlit Web Interface
      ↓
PDF Loader (LangChain)
      ↓
Text Splitter
      ↓
Embeddings (MiniLM)
      ↓
Pinecone Vector Database
      ↓
Retriever
      ↓
Ollama Llama3
      ↓
Final Answer
```

---

# 📄 License

This project is licensed under the **MIT License**.

---

# ⭐ Author

**Pudhari Swaroopa**  
NLP Project – Smart Knowledge Assistant