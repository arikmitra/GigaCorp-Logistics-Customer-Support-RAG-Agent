## Writefile is used only for testing in kaggle

#%%writefile app.py
"""
GigaCorp Customer Support RAG Agent
------------------------------------
A Streamlit chat app that answers customer questions using a local
FAQ knowledge base (RAG over FAISS), cites the exact source lines it
used, and remembers prior turns of the conversation.

Run locally:
    export GOOGLE_API_KEY=your-free-gemini-api-key
    streamlit run app.py

Uses only free-tier resources: Gemini's free API tier for the LLM, a
local sentence-transformers model for embeddings (runs on CPU or, if
available e.g. in a Kaggle T4 notebook, GPU automatically), FAISS for
the vector store, and Streamlit Community Cloud for hosting.
"""

import os
import re
from pathlib import Path
from kaggle_secrets import UserSecretsClient

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_PATH = Path("GigaCorp-Logistics-RAG-Support-Agent/data/gigacorp_faq_final.txt")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, local, no API key needed
LLM_MODEL = "gemini-3.5-flash"  # free-tier Gemini model
TOP_K = 4


def _embedding_device():
    """Use GPU automatically when available (e.g. Kaggle T4), else CPU."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

st.set_page_config(page_title="GigaCorp Support", page_icon="🛠️", layout="centered")


# --------------------------------------------------------------------------
# Knowledge base loading + chunking (chunk = one numbered FAQ line, so we
# can cite the exact source line, e.g. "gigacorp_faq.txt, line 2.3")
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing knowledge base...")
def build_vectorstore():
    text = DATA_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    docs = []
    current_section = "General"
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        section_match = re.match(r"^SECTION \d+: (.+)$", line)
        if section_match:
            current_section = section_match.group(1)
            continue
        # numbered FAQ facts look like "1.2 Standard shipping ..."
        fact_match = re.match(r"^(\d+\.\d+)\s+(.*)$", line)
        if fact_match:
            fact_id, fact_text = fact_match.groups()
            docs.append(
                Document(
                    page_content=fact_text,
                    metadata={
                        "source": "gigacorp_faq.txt",
                        "line_id": fact_id,
                        "section": current_section,
                    },
                )
            )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": _embedding_device()},
    )
    vs = FAISS.from_documents(docs, embeddings)
    return vs


def get_retriever():
    vs = build_vectorstore()
    return vs.as_retriever(search_kwargs={"k": TOP_K})

os.environ["GOOGLE_API_KEY"] = UserSecretsClient().get_secret("GOOGLE_API_KEY")
# --------------------------------------------------------------------------
# LLM chains
# --------------------------------------------------------------------------
def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "Missing GOOGLE_API_KEY. Get a free key at "
            "https://aistudio.google.com/apikey and set it as an "
            "environment variable or Streamlit secret before running the app."
        )
        st.stop()
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, google_api_key=api_key)


CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation so far and a new user question, rewrite the "
            "new question as a standalone question that captures all needed "
            "context from the conversation. If it is already standalone, return "
            "it unchanged. Only output the rewritten question, nothing else.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, friendly GigaCorp customer support agent. "
            "Answer the user's question using ONLY the knowledge base excerpts "
            "provided below. Each excerpt is tagged with a citation id like "
            "[gigacorp_faq.txt:1.2] - when you use information from an excerpt, "
            "cite it inline using that exact tag right after the relevant "
            "sentence. If the knowledge base does not contain the answer, say "
            "so honestly and suggest the user contact support directly - do not "
            "make anything up.\n\n"
            "Knowledge base excerpts:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


class SupportAnswer(BaseModel):
    answer: str = Field(
        description="The natural, conversational reply shown to the user. "
        "Never include bracket citations, tags, or line numbers here."
    )
    used_line_ids: list[str] = Field(
        default_factory=list,
        description="Line IDs (e.g. '1.1', '4.3') from the excerpts that were "
        "actually used to answer. Empty if this was general chit-chat or the "
        "knowledge base didn't cover the question.",
    )

def format_context(docs):
    return "\n".join(
        f"[gigacorp_faq.txt:{d.metadata['line_id']}] ({d.metadata['section']}) "
        f"{d.page_content}"
        for d in docs
    )


def answer_question(question, chat_history):
    llm = get_llm()
    retriever = get_retriever()

    # 1. Condense the question using conversational memory so follow-ups
    #    like "how much does it cost to ship there?" resolve correctly.
    if chat_history:
        condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()
        standalone_question = condense_chain.invoke(
            {"chat_history": chat_history, "question": question}
        )
    else:
        standalone_question = question

    # 2. Retrieve relevant FAQ chunks for the standalone question.
    docs = retriever.invoke(standalone_question)

    # 3. Generate a grounded, cited answer using the ORIGINAL question
    #    (kept natural) plus full chat history plus retrieved context.
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = answer_chain.invoke(
        {
            "chat_history": chat_history,
            "question": question,
            "context": format_context(docs),
        }
    )
    return answer, docs, standalone_question


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.title("🛠️ GigaCorp Customer Support")
st.caption(
    "Ask me about shipping, returns, business hours, service tiers, "
    "warranty, or billing. I cite the FAQ line I used for every answer."
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "sources": [...]}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for d in msg["sources"]:
                    st.markdown(
                        f"**[{d.metadata['line_id']}] {d.metadata['section']}** "
                        f"— {d.page_content}"
                    )

if prompt := st.chat_input("e.g. Do you ship to India?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build LangChain-style chat history from prior turns for memory
    history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Checking the knowledge base..."):
            answer, sources, standalone_q = answer_question(prompt, history)
        st.markdown(answer)
        if standalone_q != prompt:
            st.caption(f"🧠 Understood with context as: *{standalone_q}*")
        with st.expander("Sources used"):
            for d in sources:
                st.markdown(
                    f"**[{d.metadata['line_id']}] {d.metadata['section']}** "
                    f"— {d.page_content}"
                )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant retrieves answers from a local FAQ knowledge base "
        "(`data/gigacorp_faq.txt`) using FAISS + sentence-transformer "
        "embeddings, and generates responses with Google's free-tier "
        "Gemini model. Conversation memory lets it resolve follow-up "
        "questions like *'how much does it cost?'* after you've already "
        "mentioned a destination."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
        
        
# import os
# os._exit(0)  # restarts the kernel; rerun cells after


# ngrok is only used because agent was tested in kaggle
#!pip install -q pyngrok
from pyngrok import ngrok
ngrok_token = UserSecretsClient().get_secret("NGROK_TOKEN")
ngrok.set_auth_token(ngrok_token) # free at ngrok.com
public_url = ngrok.connect(8501)
print(public_url)
get_ipython().system_raw("streamlit run app.py &")        
