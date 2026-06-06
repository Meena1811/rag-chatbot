import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
import os

load_dotenv()

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Company RAG Chatbot", layout="wide")
st.title("🤖 Company RAG Chatbot with RBAC & Guardrails")

# ---- USER DATABASE ----
USERS = {
    "alice": {"password": "hr123", "role": "hr"},
    "bob": {"password": "fin123", "role": "finance"},
    "carol": {"password": "ceo123", "role": "c_level"},
}

ROLE_ACCESS = {
    "hr": ["hr", "general"],
    "finance": ["finance", "general"],
    "c_level": ["hr", "finance", "general"],
}

DOCUMENT_ROLES = {
    "data/hr_policy.txt": "hr",
    "data/finance_report.txt": "finance",
    "data/general_faq.txt": "general",
}

# ---- INITIALIZE SESSION STATE ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.chat_history = []
    st.session_state.token_count = 0

# ---- LOAD VECTORSTORE ----
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    all_chunks = []
    for filepath, role in DOCUMENT_ROLES.items():
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["role"] = role
        all_chunks.extend(chunks)
    
    vectorstore = Chroma.from_documents(
        all_chunks,
        embeddings,
        persist_directory="./chroma_db_streamlit"
    )
    return vectorstore

vectorstore = load_vectorstore()

# ---- GUARDRAILS ----
def detect_prompt_injection(question):
    injection_keywords = [
        "ignore previous", "forget the system", "bypass", "override",
        "execute code", "system prompt", "instructions",
    ]
    return any(keyword in question.lower() for keyword in injection_keywords)

def is_question_in_scope(question):
    in_scope_keywords = [
        "company", "employee", "leave", "salary", "policy", "finance",
        "profit", "revenue", "office", "hr", "q1", "expenses", "support", "it"
    ]
    return any(keyword in question.lower() for keyword in in_scope_keywords)

def mask_pii(text):
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_MASKED]', text)
    return text

# ---- QUERY WITH GUARDRAILS ----
def ask_question_with_guardrails(user_role, question):
    if detect_prompt_injection(question):
        return "🚨 SECURITY: Your question contains suspicious patterns. Please rephrase."
    
    if not is_question_in_scope(question):
        return "❌ OUT OF SCOPE: I can only answer about company policies, finance, HR, and general company info."
    
    allowed_roles = ROLE_ACCESS[user_role]
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"role": {"$in": allowed_roles}}
        }
    )
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If the answer is not in the context, say "I don't have access to that information."

Context: {context}
Question: {question}
""")
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    response = mask_pii(response)
    st.session_state.token_count += len(question.split()) + len(response.split())
    return response

# ---- UI LOGIC ----
if not st.session_state.logged_in:
    st.subheader("🔐 Login")
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username", placeholder="alice, bob, or carol")
    with col2:
        password = st.text_input("Password", type="password")
    
    if st.button("Login", use_container_width=True):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_role = USERS[username]["role"]
            st.success(f"✅ Welcome {username}! Role: {st.session_state.user_role.upper()}")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    
    st.divider()
    st.markdown("""
    **Test Accounts:**
    - alice / hr123 (HR role)
    - bob / fin123 (Finance role)
    - carol / ceo123 (C-Level role)
    """)

else:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**👤 Logged in as:** {st.session_state.username} ({st.session_state.user_role.upper()})")
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
    
    question = st.chat_input("Ask a question about company data...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        
        with st.spinner("Checking guardrails & retrieving..."):
            answer = ask_question_with_guardrails(st.session_state.user_role, question)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
    
    with st.sidebar:
        st.subheader("📊 Monitoring")
        st.metric("Tokens Used", st.session_state.token_count)
        st.metric("Messages", len([m for m in st.session_state.chat_history if m["role"] == "user"]))
        
        st.divider()
        st.subheader("ℹ️ System Info")
        st.info(f"""
        **Role-Based Access:** Active
        **Guardrails:** Enabled
        - Prompt Injection Detection
        - Out-of-Scope Blocking
        - PII Masking
        
        **Accessible Roles:**
        {', '.join(ROLE_ACCESS[st.session_state.user_role])}
        """)