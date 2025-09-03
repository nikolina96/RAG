


"""
Document QA Pipeline with LangChain, HuggingFace Embeddings, FAISS, and Ollama (LLaMA)
Based on https://medium.com/@danushidk507/rag-with-llama-using-ollama-a-deep-dive-into-retrieval-augmented-generation-c58b9a1cfcd3

This script:
1. Loads a PDF document.
2. Splits the text into chunks for processing.
3. Creates embeddings and stores them in a FAISS vector store.
4. Initializes an Ollama LLaMA model for question answering.
5. Sets up a RetrievalQA chain with a custom prompt.
6. Runs an interactive loop for user queries.
"""

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ----------------------------
# Step 1: Load PDF document
# ----------------------------
loader = PyPDFLoader("documents/guidelines.pdf")
documents = loader.load()


# ----------------------------
# Step 2: Split document into chunks
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Max tokens per chunk
    chunk_overlap=100,   # Overlap to preserve context
    separators=["\n\n", "\n", ".", " ", ""]
)
docs = text_splitter.split_documents(documents=documents)


# ----------------------------
# Step 3: Create embeddings
# ----------------------------
embedding_model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}  # Use "cuda" if GPU is available

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs
)


# ----------------------------
# Step 4: Create FAISS vector store
# ----------------------------
# Build vector database from chunks
vectorstore = FAISS.from_documents(docs, embeddings)

# Save vector database locally
vectorstore.save_local("faiss_index_")

# Reload vector database
persisted_vectorstore = FAISS.load_local(
    "faiss_index_", embeddings, allow_dangerous_deserialization=True
)

# Create a retriever for fetching relevant chunks
retriever = persisted_vectorstore.as_retriever()


# ----------------------------
# Step 5: Initialize LLaMA model
# ----------------------------
llm = Ollama(model="llama3.1")


# ----------------------------
# Step 6: Define custom QA prompt
# ----------------------------
qa_prompt = PromptTemplate(
    template=(
        "You are a helpful assistant. Use the following context to answer:\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer concisely and only from the context above."
    ),
    input_variables=["context", "question"],
)


# ----------------------------
# Step 7: Build RetrievalQA chain
# ----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",          # Simple chain that feeds all context
    retriever=retriever,
    chain_type_kwargs={"prompt": qa_prompt}
)


# ----------------------------
# Step 8: Interactive Q&A loop
# ----------------------------
print("Interactive Document QA. Type 'Exit' to quit.\n")
while True:
    query = input("Type your query: \n")
    if query.lower() == "exit":
        print("Goodbye!")
        break

    result = qa.run(query)
    print("\nAnswer:\n", result, "\n")
