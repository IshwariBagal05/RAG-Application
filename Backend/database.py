
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load .env from the same folder as database.py
env_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Check Backend/.env"
    )

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

# Initialize Chroma database
vector_store = Chroma(
    collection_name="myCollection",
    embedding_function=embeddings,
    persist_directory=str(
        Path(__file__).parent / "chroma_db"
    )
)


def addData(data: list):
    vector_store.add_texts(texts=data)
    print("Data stored successfully!")


def getData():
    return vector_store.get()


def askQuestion(askedQuery: str):
    chunks = vector_store.similarity_search(
        query=askedQuery,
        k=2
    )
    return chunks
