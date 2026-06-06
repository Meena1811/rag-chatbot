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

# Step 1: Load documents
print("Loading documents...")
documents = []
for filename in ["data/hr_policy.txt", "data/finance_report.txt", "data/general_faq.txt"]:
    loader = TextLoader(filename, encoding="utf-8")
    documents.extend(loader.load())
print(f"Loaded {len(documents)} documents")

# Step 2: Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3: Embeddings + ChromaDB
print("Creating embeddings (first time takes 2 mins)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Embeddings stored!")

# Step 4: LLM + chain
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Context: {context}
Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 5: Test
print("\n--- Testing RAG Pipeline ---")
questions = [
    "How many paid leaves do employees get?",
    "What was the net profit in Q1 2026?",
    "Where is the company office located?"
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {chain.invoke(q)}")