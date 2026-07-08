import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
vectorstore = Chroma(
    collection_name = "pdf_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
for pdf_path in glob.glob("pdfs/*.pdf"):
    print(f"Ingesting {pdf_path}")
    docs = PyMuPDFLoader(pdf_path).load()
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
print("Done")
