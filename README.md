# Smart Knowledge Assistant (NLP RAG Project)

An **NLP-based Smart Knowledge Assistant** that answers questions from structured datasets using **Retrieval-Augmented Generation (RAG)**.
The system loads **CSV and JSON datasets**, converts them into embeddings, stores them in **Pinecone vector database**, and retrieves relevant information to generate answers using **LLM (Ollama/Llama3)**.

---

## Project Overview

This project demonstrates how to build a **knowledge-based question answering system** using modern NLP tools.

Pipeline:

```
Dataset (CSV + JSON)
        ↓
Document Loader
        ↓
Text Cleaning & Filtering
        ↓
Text Chunking
        ↓
Embeddings (Sentence Transformers)
        ↓
Pinecone Vector Database
        ↓
Retriever
        ↓
LLM (Llama3 via Ollama)
        ↓
Answer Generation
```

---

# Folder Structure

```
Smart-Knowledge-Assistant
│
├── data/
│   ├── archive(1)/
│   ├── archive(2)/
│   └── archive(3)/
│
├── smart-knowledge-assistant.ipynb
├── requirements.txt
├── .env
└── README.md
```

---

# Prerequisites

Install the following software:

* Python **3.10 or 3.11** (recommended)
* Git
* VS Code or Jupyter Notebook
* Ollama

Download Ollama:

https://ollama.com/download

Pull the Llama3 model:

```
ollama pull llama3
```

---

# Step 1: Clone the Repository

```
git clone <your-repository-url>
cd Smart-Knowledge-Assistant
```

---

# Step 2: Create Virtual Environment

```
python -m venv smart-knowledge
```

Activate environment:

Windows:

```
smart-knowledge\Scripts\activate
```

Mac/Linux:

```
source smart-knowledge/bin/activate
```

---

# Step 3: Install Dependencies

Install required libraries:

```
pip install -r requirements.txt
```

Example dependencies:

```
langchain
langchain-community
langchain-text-splitters
pinecone-client
sentence-transformers
pypdf
python-dotenv
pandas
```

---

# Step 4: Setup Environment Variables

Create a `.env` file in the project root.

Example:

```
PINECONE_API_KEY=your_pinecone_api_key
```

---

# Step 5: Add Dataset

Place datasets inside the **data folder**.

Example:

```
data/
   archive(1)/
       file1.csv
       file2.json
   archive(2)/
       file3.csv
   archive(3)/
       file4.json
```

---

# Step 6: Load Datasets

The system loads CSV and JSON files recursively.

Example loaders:

### CSV Loader

```
from langchain_community.document_loaders import DirectoryLoader, CSVLoader

loader = DirectoryLoader(
    "data",
    glob="**/*.csv",
    loader_cls=CSVLoader
)

documents = loader.load()
```

### JSON Loader

```
from langchain_community.document_loaders import JSONLoader

loader = DirectoryLoader(
    "data",
    glob="**/*.json",
    loader_cls=JSONLoader,
    loader_kwargs={"jq_schema": ".", "text_content": False}
)
```

---

# Step 7: Filter Documents

Reduce metadata and keep only required fields.

```
from langchain_core.documents import Document

def filter_to_minimal_docs(docs):
    minimal_docs = []

    for doc in docs:
        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs
```

---

# Step 8: Split Documents into Chunks

```
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

text_chunks = text_splitter.split_documents(minimal_docs)
```

---

# Step 9: Generate Embeddings

```
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Embedding dimension = **384**

---

# Step 10: Setup Pinecone Vector Database

```
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "smart-knowledge-assistant"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
```

---

# Step 11: Store Embeddings in Pinecone

```
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name
)
```

---

# Step 12: Create Retriever

```
retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k":8}
)
```

---

# Step 13: Load LLM

```
from langchain_community.llms import Ollama

chatmodel = Ollama(model="llama3")
```

---

# Step 14: Create RAG Chain

```
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
"You are an AI assistant for question answering tasks. "
"Use the context to answer the question."
)

prompt = ChatPromptTemplate.from_messages(
[
("system", system_prompt),
("human", "{input}")
]
)

question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

---

# Step 15: Ask Questions

```
response = rag_chain.invoke(
{"input": "What is artificial intelligence?"}
)

print(response["answer"])
```

---

# Example Query

Input:

```
What is machine learning?
```

Output:

```
Machine learning is a branch of artificial intelligence that enables systems to learn from data and improve performance without being explicitly programmed.
```

---

# Technologies Used

* Python
* LangChain
* Pinecone
* Sentence Transformers
* Ollama
* Llama3
* Pandas
* Jupyter Notebook

---

# Features

* Supports **CSV and JSON datasets**
* Uses **vector embeddings for semantic search**
* Implements **Retrieval-Augmented Generation (RAG)**
* Works locally with **Ollama Llama3**
* Scalable with Pinecone vector database

---

# Future Improvements

* Add Streamlit UI
* Add document summarization
* Support PDF datasets
* Improve chunking for tabular datasets

---

# Author

Pudhari Swaroopa
NLP Academic Project – Smart Knowledge Assistant
