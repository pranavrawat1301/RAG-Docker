# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List
import os
from datetime import datetime
from query import user_input, response_generator
from save import PyMuPDFLoader, RecursiveCharacterTextSplitter
from upsert import process_batch

app = FastAPI()

# MongoDB connection (using MongoDB Atlas free tier)
MONGODB_URI = "mongodb+srv://rag:<db_password>@cluster0.4pkqz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGODB_URI)
db = client.ragdb

class Query(BaseModel):
    text: str
    generate_summary: bool = False

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()
        
        # Store metadata in MongoDB
        metadata = {
            "filename": file.filename,
            "upload_time": datetime.utcnow(),
            "page_count": len(documents),
            "status": "processed"
        }
        db.documents.insert_one(metadata)
        
        # Process and upsert to Pinecone (using your existing code)
        non_empty_documents = [doc for doc in documents if doc.page_content.strip()]
        full_text = " ".join([doc.page_content for doc in non_empty_documents])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)
        process_batch(chunks)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {"message": "Document processed successfully", "metadata": metadata}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: Query):
    try:
        # Get response using existing functions
        if query.generate_summary:
            response = response_generator(query.text)
        else:
            response = user_input(query.text)
        
        # Store query metadata
        query_metadata = {
            "query_text": query.text,
            "timestamp": datetime.utcnow(),
            "generated_summary": query.generate_summary
        }
        db.queries.insert_one(query_metadata)
        
        return {
            "response": response,
            "metadata": query_metadata
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    try:
        documents = list(db.documents.find({}, {"_id": 0}))
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))