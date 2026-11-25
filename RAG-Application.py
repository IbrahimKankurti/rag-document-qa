import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.vectorstores import InMemoryVectorStore   # replaces FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# ---------- Load API key ----------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Check your .env file.")
    st.stop()

# ---------- LLM ----------
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=groq_api_key,
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context provided. Please provide the most accurate answers
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# ---------- Embeddings + vector store ----------
INDEX_FILE = "vector_store.json"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64},
    )


def create_vector_embedding():
    if "vectors" in st.session_state:
        return
    embeddings = get_embeddings()

    if os.path.exists(INDEX_FILE):
        # Fast path: load the saved file
        st.session_state.vectors = InMemoryVectorStore.load(INDEX_FILE, embeddings)
    else:
        # Slow path: runs only once
        loader = PyPDFDirectoryLoader("research_papers")  # Data ingestion
        docs = loader.load()                              # Document loading
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = splitter.split_documents(docs[:50])
        store = InMemoryVectorStore(embeddings)
        store.add_documents(final_documents)
        store.dump(INDEX_FILE)  # save to disk so you never wait again
        st.session_state.vectors = store


# ---------- Streamlit UI ----------
st.title("Ask Questions to RAG Document")

user_prompt = st.text_input("Enter your query related to research paper")

if st.button("Document Search"):
    with st.spinner("Preparing documents..."):
        create_vector_embedding()
    st.write("Vector DB is ready")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Search' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        print(f"Response time: {time.process_time() - start}")

        st.write(response["answer"])