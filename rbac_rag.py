from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# ---- USER DATABASE (simulated login) ----
USERS = {
    "alice": {"password": "hr123",      "role": "hr"},
    "bob":   {"password": "fin123",     "role": "finance"},
    "carol": {"password": "ceo123",     "role": "c_level"},
}

# ---- DOCUMENT TO ROLE MAPPING ----
DOCUMENT_ROLES = {
    "data/hr_policy.txt":      "hr",
    "data/finance_report.txt": "finance",
    "data/general_faq.txt":    "general",
}

# ---- ROLE ACCESS RULES ----
ROLE_ACCESS = {
    "hr":      ["hr", "general"],
    "finance": ["finance", "general"],
    "c_level": ["hr", "finance", "general"],
}

# ---- LOAD & TAG DOCUMENTS ----
print("Loading and tagging documents...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

all_chunks = []
for filepath, role in DOCUMENT_ROLES.items():
    loader = TextLoader(filepath, encoding="utf-8")
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["role"] = role  # tag each chunk with its role
    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")

# ---- STORE IN CHROMADB ----
vectorstore = Chroma.from_documents(
    all_chunks,
    embeddings,
    persist_directory="./chroma_db_rbac"
)
print("Stored in ChromaDB with role tags!")

# ---- LLM SETUP ----
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If the answer is not in the context, say "I don't have access to that information."

Context: {context}
Question: {question}
""")

# ---- QUERY FUNCTION WITH RBAC ----
def ask_question(user_role, question):
    allowed_roles = ROLE_ACCESS[user_role]

    # Filter retriever by allowed roles
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"role": {"$in": allowed_roles}}
        }
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)

# ---- SIMULATE LOGIN & TEST ----
def login():
    print("\n========== COMPANY CHATBOT ==========")
    username = input("Enter username: ")
    password = input("Enter password: ")

    if username in USERS and USERS[username]["password"] == password:
        user_role = USERS[username]["role"]
        print(f"\nWelcome {username}! You are logged in as: {user_role.upper()}")
        print("Type 'quit' to exit\n")
        return user_role
    else:
        print("Invalid credentials!")
        return None

# ---- MAIN CHAT LOOP ----
user_role = login()
if user_role:
    while True:
        question = input("\nYour question: ")
        if question.lower() == "quit":
            break
        print("\nSearching...")
        answer = ask_question(user_role, question)
        print(f"Answer: {answer}")