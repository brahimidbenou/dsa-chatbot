from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

def get_embedding() -> OllamaEmbeddings:
    embeddings = OllamaEmbeddings()
    return embeddings

def get_llm(model_name = "llama2") -> Ollama:
    llm = Ollama(model=model_name)
    return llm

def get_loader(file: str) -> UnstructuredFileLoader:
    loader = UnstructuredFileLoader(file)
    return loader

def get_splitter(size=900, overlap=100) -> CharacterTextSplitter:
    text_splitter = CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return text_splitter

def init_db(name, docs, embeddings):
    db = Chroma.from_documents(docs[:10], embeddings, persist_directory=name)
    return db

def load_db(name, embeddings) -> Chroma:
    db = Chroma(persist_directory=name, embedding_function=embeddings)
    return db

def get_retriever(db, search="similarity", k=3):
    retriever = db.as_retriever(search_type=search, search_kwargs={"k": k})
    return retriever

def get_rag(llm, prompt, retriever):
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

def get_prompt():
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    return prompt

def custom_ui():
    css = f"""
    <style>
        [data-testid="stChatMessage"]:nth-child(odd) {{
            flex-direction: row-reverse;
            background-color: unset;
        }}
    </style>
    """
    st.html(css)

