# RAG Pipeline with FastAPI and Docker

A Retrieval-Augmented Generation (RAG) pipeline that enables document upload, vector storage, and intelligent querying using Google's Gemini Pro and Pinecone vector database.

## Features

- Document upload and processing (PDF support)
- Vector storage using Pinecone
- Question answering using Google's Gemini Pro
- FastAPI REST endpoints
- MongoDB for metadata storage
- Docker containerization

## Prerequisites

- Docker and Docker Compose
- Pinecone API key
- Google API key (Gemini Pro)
- MongoDB Atlas account


## Installation & Setup

1. Clone the repository:
```bash
git clone 
cd 
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

## API Endpoints

### 1. Upload Document
```http
POST /upload
Content-Type: multipart/form-data

file: PDF_FILE
```

### 2. Query Documents
```http
POST /query
Content-Type: application/json

{
    "text": "your question here",
    "generate_summary": false
}
```

### 3. List Documents
```http
GET /documents
```


## Configuration

### Vector Database Setup

1. Create a Pinecone account and get your API key
2. Create an index with:
   - Dimensions: 384 (for all-MiniLM-L6-v2 embeddings)
   - Metric: cosine
   - Environment: us-east-1

### LLM Provider Setup

1. Create a Google Cloud account
2. Enable the Gemini API
3. Generate an API key

## Architecture

The system follows a microservices architecture:
- FastAPI web server
- MongoDB for metadata storage
- Pinecone for vector storage
- Google Gemini Pro for LLM capabilities

## Performance Considerations

- Maximum document size: 1000 pages
- Maximum concurrent uploads: 20 documents
- Chunk size: 2000 characters
- Chunk overlap: 200 characters

## Error Handling

The API implements proper error handling for:
- Invalid file formats
- Failed uploads
- Database connection issues
- LLM API failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT
