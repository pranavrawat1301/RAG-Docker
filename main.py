from fastapi import FastAPI, UploadFile, File, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List
import os
from datetime import datetime
from query import user_input, response_generator
from save import PyMuPDFLoader, RecursiveCharacterTextSplitter
import ssl
from upsert import upsertion

app = FastAPI()

MONGODB_URI = "mongodb+srv://rag:Pranav04@cluster0.4pkqz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
try:
    client = MongoClient(MONGODB_URI,
        ssl=True,
        ssl_cert_reqs=ssl.CERT_NONE
    )
    # Test the connection
    client.admin.command('ping')
    db = client.ragdb
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise

class Query(BaseModel):
    text: str
    generate_summary: bool = False

def serialize_document(doc):
    """Convert MongoDB document to JSON-serializable format"""
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])  
    return doc

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()
        
        metadata = {
            "filename": file.filename,
            "upload_time": datetime.utcnow(),
            "page_count": len(documents),
            "status": "processed"
        }
        result = db.documents.insert_one(metadata)
        
        metadata['_id'] = str(result.inserted_id)
        
        non_empty_documents = [doc for doc in documents if doc.page_content.strip()]
        full_text = " ".join([doc.page_content for doc in non_empty_documents])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)
        upsertion(chunks)
        
        os.remove(temp_path)
        
        return {"message": "Document processed successfully", "metadata": metadata}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: Query):
    try:
        if query.generate_summary:
            response = response_generator(query.text)
        else:
            response = user_input(query.text)
        
        query_metadata = {
            "query_text": query.text,
            "timestamp": datetime.utcnow().isoformat(),  
            "generated_summary": query.generate_summary
        }
        
        result = db.queries.insert_one(query_metadata)
        query_metadata['_id'] = str(result.inserted_id) 
        
        if isinstance(response, dict) and 'output_text' in response:
            response_text = response['output_text']
        else:
            response_text = str(response)
        
        return {
            "response": response_text,
            "metadata": query_metadata
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    try:
        documents = list(db.documents.find({}))
        serialized_documents = [serialize_document(doc) for doc in documents]
        return serialized_documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
