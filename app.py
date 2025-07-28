import streamlit as st
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI, HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from utils import translate_text, extract_text_from_pdf, extract_text_from_docx

# Load environment variables
load_dotenv()

# --- UI Config ---
st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📄",
    layout="centered"
)

# Dark/light mode toggle
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    [data-testid="stHeader"] { background-color: transparent; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = None

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
   

    st.session_state.dark_mode = st.toggle("Dark Mode", value=False)
    
    # File Upload
    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX", 
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    # Action Buttons
    if st.button("✨ Generate Summary"):
        if uploaded_files:
            text = ""
            for file in uploaded_files:
                if file.type == "application/pdf":
                    text += extract_text_from_pdf(file)
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text += extract_text_from_docx(file)
            
            llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
            prompt = PromptTemplate(
                input_variables=["text"],
                template="Summarize this in 3 bullet points:\n{text}"
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            st.session_state.summary = chain.run(text)
        else:
            st.warning("Upload files first!")

# --- Main Chat UI ---
st.title("📄 DocuChat AI")
st.caption("Chat with your documents in 50+ languages")

# Display summary if available
if st.session_state.summary:
    with st.expander("📌 Document Summary"):
        st.write(st.session_state.summary)

# Chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Thinking..."):
        # Process query
        text = ""
        for file in uploaded_files:
            if file.type == "application/pdf":
                text += extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text += extract_text_from_docx(file)
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        # Generate response
        llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )
        response = qa_chain({"question": prompt})
        
        # Translation option
        if "translate" in prompt.lower():
            response["answer"] = translate_text(response["answer"], target_lang="es")  # Spanish as default
        
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
    
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
