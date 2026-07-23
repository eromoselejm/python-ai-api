from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import os
import faiss
import pickle
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv



load_dotenv()

my_api_key = os.getenv("GEMINI_API_KEY")

print("API KEY FOUND:", my_api_key is not None)
print("API KEY LENGTH:", len(my_api_key) if my_api_key else 0)

client = genai.Client(api_key=my_api_key)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
app = FastAPI()



@app.get("/")
async def home():
    return {
        "message": "FastAPI is running!"
    }

@app.post("/upload")
async def upload(uploaded_file: UploadFile = File(...)):

    content = PdfReader(uploaded_file.file)
    metadata = []

    text = ""
    for f in content.pages:
        text += f.extract_text()
    
    #Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    page_number = 1

    for chunk in chunks:
        metadata.append({
        "filename": uploaded_file.filename,
        "text": chunk,
    })

    #Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    #Create output folder
    os.makedirs("vector_db", exist_ok=True)

    #Save FAISS index
    faiss.write_index(index, "vector_db/faiss.index")

    with open("vector_db/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    return "Done"


#Search Endpoint

class SearchRequest(BaseModel):
    query: str

@app.post("/search_query")
async def search_query(request: SearchRequest):

    context = ""
    query = request.query
    results = []

    #Check if document has been uploaded
    if os.path.exists("vector_db/faiss.index") and os.path.exists("vector_db/metadata.pkl"):
    #Load FAISS index
        index = faiss.read_index("vector_db/faiss.index")

        #Load metadata
        with open("vector_db/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        #Search Top 5 nearest chunks
        distances, indices = index.search(query_embedding, 5)

        DISTANCE_THRESHOLD = 0.8

        #Iterating through distances and indices using zip() to pair them together
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if distance <= DISTANCE_THRESHOLD:
                results.append(metadata[idx]["text"])
    
    
    context = "\n\n".join(results)

    if context:
        prompt = f"""
        You are a helpful AI assistant.

        Answer the user's question using the context below.

        If the answer isn't in the context,
        say so, then answer using your own knowledge.

        Context:
        {context}

        Question:
        {query}
        """
    else:
        prompt = f"""
        You are a helpful AI assistant.

        No uploaded document is available.

        Answer the user's question using your general knowledge.

        Question:
        {query}
        """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return {
            "response": response.text,
            "error": "No Error"
        }

    except Exception as e:
        return {
            "response": "Sorry something went wrong.",
            "error": str(e)
        }





