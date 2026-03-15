import streamlit as st
import os
import nltk
from dotenv import load_dotenv

# LangChain & Vector Store
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# Handle the legacy chain imports
try:
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- INITIAL SETUP ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "smart-knowledge-assistant"

# --- 1. DATA PROCESSING (CACHED) ---
@st.cache_resource
def initialize_system():
    # Load Embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Load and Split PDF (Only runs if index is new or for local retrieval)
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    extracted_data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(extracted_data)

    # Connect to Vector Store
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    return docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# --- 2. RAG LOGIC ---
def get_response(user_input, retriever):
    chatmodel = Ollama(model="llama3")
    
    system_prompt = (
        "You are a Smart Knowledge assistant for question answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": user_input})
    return response["answer"]

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="AI/ML Knowledge Assistant", layout="centered")
st.title("🧠 Smart Knowledge Assistant")
st.markdown("---")

if not PINECONE_API_KEY:
    st.error("Please set your `PINECONE_API_KEY` in the `.env` file.")
    st.stop()

# Initialize system
with st.status("Initializing Knowledge Base...", expanded=False) as status:
    st.write("Loading embeddings...")
    st.write("Checking Pinecone index...")
    retriever = initialize_system()
    status.update(label="System Ready!", state="complete", expanded=False)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask a question about AI/ML..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_response(user_input, retriever)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
