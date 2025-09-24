import os
from dotenv import load_dotenv
import streamlit as st
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"]=sys.modules.pop("pysqlite3")
except ImportError:
    pass

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="UN SDG Q&A Assistant", page_icon="🌍")

#loading environment variables (API keys)
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#building retriever
@st.cache_resource(show_spinner="Building or loading retriever...")
def build_retriever():
    persist_dir = "chroma_store"

    #loading documents
    loader = TextLoader("sdg_goals.txt")
    docs = loader.load()

    #embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #if Chroma store exists, load it(prolly cache). Else, build and save.
    if os.path.exists(persist_dir):
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)

    return db.as_retriever()

retriever = build_retriever()


prompt = PromptTemplate(
    template=(
        "You are a UNDP research assistant. "
        "Use the following context to answer clearly and concisely.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    input_variables=["context", "question"]
)

#initializing QA chain (priority to OpenAI, fallback to HuggingFace)
def init_qa():
    qa = None
    try:
        from langchain.chat_models import init_chat_model
        if openai_key:
            llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            #testing OpenAI credits
            _ = llm.invoke("ping")
            st.sidebar.success("Using OpenAI GPT-4o-mini")
        else:
            raise RuntimeError("OPENAI_API_KEY not set.")
    except Exception:
        from langchain_community.llms import HuggingFaceHub
        llm = HuggingFaceHub(
            # repo_id="google/flan-t5-large",
            repo_id="bigscience/bloomz-560m",
            model_kwargs={"temperature": 0.1, "max_length": 256},
            huggingfacehub_api_token=hf_token,
            task="text2text-generation"
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        st.sidebar.warning("Using Hugging Face Flan-T5 (fallback)")
    return qa

qa = init_qa()

#Streamlit UI

st.title("🌍 UN SDG Q&A Assistant")
st.write("Ask me questions about the **Sustainable Development Goals (SDGs)**. "
         "I’ll retrieve context from UN documents and provide clear answers.")

user_query = st.text_input("Enter your question:")
if user_query:
    with st.spinner("Thinking..."):
        response = qa.invoke({"query": user_query})
    st.write("### Answer")
    st.success(response["result"])

#sidebar info

st.sidebar.markdown("### About")
st.sidebar.info(
    "This app is powered by **LangChain**, **Chroma**, and **LLMs (OpenAI/HuggingFace)**.\n\n"
    "It was built as a demo project for **UNDP-style SDG Q&A**."
)

st.sidebar.markdown("#### *Note:")
st.sidebar.info(
    "To ask another question, just type in the question and press **Enter**.\n\n"
)
