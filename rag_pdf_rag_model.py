#!/usr/bin/env python3
"""
RAG pipeline for multiple PDF documents.
Dependencies: pip install langchain faiss-cpu transformers sentence-transformers unstructured
"""
import argparse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA

def load_and_split_pdfs(pdf_paths):
    """Load multiple PDFs and split into text chunks."""
    documents = []
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        raw_docs = loader.load()
        docs = splitter.split_documents(raw_docs)
        documents.extend(docs)
    return documents

def build_vectorstore(documents):
    """Create a FAISS vectorstore from documents."""
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embed_model)
    return vectorstore

def create_qa_chain(vectorstore):
    """Initialize the RAG QA chain."""
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0, "max_length": 256}
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

def ask_rag(question, pdfs):
    """Run the RAG pipeline on the merged PDFs."""
    docs = load_and_split_pdfs(pdfs)
    vectorstore = build_vectorstore(docs)
    qa_chain = create_qa_chain(vectorstore)
    result = qa_chain(question)
    answer = result["result"]
    sources = result["source_documents"]
    print("Answer:\n", answer)
    print("\nSources:")
    for doc in sources:
        src = doc.metadata.get("source", "unknown")
        print(f"- {src} (length: {len(doc.page_content)} chars)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a RAG system over merged PDFs")
    parser.add_argument("question", type=str, help="Question to ask the RAG system")
    parser.add_argument(
        "--pdfs", 
        nargs="+", 
        default=[
            "/mnt/data/Banana Expert System.pdf",
            "/mnt/data/Banana_Bharat Agri.pdf",
            "/mnt/data/BananaCultivation.pdf",
            "/mnt/data/TNAU_Banana Expert System.pdf"
        ],
        help="List of PDF file paths to merge"
    )
    args = parser.parse_args()
    ask_rag(args.question, args.pdfs)
