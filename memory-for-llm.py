from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step 1: Load raw PDFs
DATA_PATH = "notes/"
def load_pdf_files(data):
    print(f"Looking for PDFs in: {os.path.abspath(data)}")
    if not os.path.exists(data):
        print(f"Error: Directory {data} does not exist")
        return []
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    try:
        documents = loader.load()
        print(f"Found {len(documents)} PDF pages")
        return documents
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []

documents = load_pdf_files(data=DATA_PATH)
print("Length of PDF pages: ", len(documents))
if not documents:
    print("No documents loaded. Exiting.")
    exit()

# Step 2: Create Chunks
def create_chunks(extracted_data):
    print("Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Created {len(text_chunks)} chunks")
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("Length of Text Chunks: ", len(text_chunks))
if not text_chunks:
    print("No chunks created. Exiting.")
    exit()

# Step 3: Create Vector Embeddings
def get_embedding_model():
    print("Initializing embedding model...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Embedding model loaded")
        return embedding_model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

embedding_model = get_embedding_model()
if not embedding_model:
    print("Embedding model failed to load. Exiting.")
    exit()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
print(f"Saving FAISS index to: {DB_FAISS_PATH}")
try:
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("FAISS index saved successfully")
except Exception as e:
    print(f"Error saving FAISS index: {e}")