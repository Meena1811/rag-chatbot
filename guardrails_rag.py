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

# ---- USER DATABASE ----
USERS = {
    "alice": {"password": "hr123", "role": "hr"},
    "bob": {"password": "fin123", "role": "finance"},
    "carol": {"password": "ceo123", "role": "c_level"},
}

# ---- DOCUMENT TO ROLE MAPPING ----
DOCUMENT_ROLES = {
    "data/hr_policy.txt": "hr",
    "data/finance_report.txt": "finance",
    "data/general_faq.txt": "general",
}

# ---- ROLE ACCESS RULES ----
ROLE_ACCESS = {
    "hr": ["hr", "general"],
    "finance": ["finance", "general"],
    "c_level": ["hr", "finance", "general"],
}

# ---- SETUP EMBEDDINGS & VECTORSTORE ----
print("Loading documents...")
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
    persist_directory="./chroma_db_guardrails"
)
print("Loaded with guardrails ready!")

# ---- GUARDRAIL 1: PROMPT INJECTION DETECTION ----
def detect_prompt_injection(question):
    """Detect common prompt injection attacks"""
    injection_keywords = [
        "ignore previous",
        "forget the system",
        "bypass",
        "override",
        "execute code",
        "system prompt",
        "instructions",
    ]
    question_lower = question.lower()
    for keyword in injection_keywords:
        if keyword in question_lower:
            return True
    return False

# ---- GUARDRAIL 2: OUT-OF-SCOPE DETECTION ----
def is_question_in_scope(question):
    """Check if question is about company data"""
    in_scope_keywords = [
        "company", "employee", "leave", "salary", "policy", 
        "finance", "profit", "revenue", "office", "hr", 
        "q1", "expenses", "support", "it"
    ]
    question_lower = question.lower()
    
    has_keyword = any(keyword in question_lower for keyword in in_scope_keywords)
    return has_keyword

# ---- GUARDRAIL 3: PII MASKING ----
def mask_pii(text):
    """Mask emails and phone numbers in response"""
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_MASKED]', text)
    return text

# ---- QUERY WITH GUARDRAILS ----
def ask_question_with_guardrails(user_role, question):
    """Process question with all guardrails"""
    
    if detect_prompt_injection(question):
        return "SECURITY: Your question contains suspicious patterns. Please rephrase."
    
    if not is_question_in_scope(question):
        return "OUT OF SCOPE: I can only answer about company policies, finance, HR, and general company info."
    
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
    
    return response

# ---- LOGIN ----
def login():
    print("\n========== COMPANY CHATBOT WITH GUARDRAILS ==========")
    username = input("Enter username (alice/bob/carol): ")
    password = input("Enter password: ")
    
    if username in USERS and USERS[username]["password"] == password:
        user_role = USERS[username]["role"]
        print(f"\nWelcome {username}! Role: {user_role.upper()}")
        print("Type 'quit' to exit\n")
        return user_role
    else:
        print("Invalid credentials!")
        return None

# ---- MAIN CHAT LOOP ----
user_role = login()
if user_role:
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == "quit":
            print("Goodbye!")
            break
        if not question:
            continue
        
        print("Checking guardrails...")
        answer = ask_question_with_guardrails(user_role, question)
        print(f"Answer: {answer}")