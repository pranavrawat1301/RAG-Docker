from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import uuid
import concurrent.futures
from save import chunks
from save import file_name


model = SentenceTransformer('all-MiniLM-L6-v2')  
pc = Pinecone(api_key="pcsk_48jq17_8zsXqWFqrSZVSi9fFqMnxjsa8L3iP1CPDCZ88z7j1eq5y8MZvEjwrj7yd9T5ERH", pool_threads=50)
index_name = "fast-rag2"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,    #768
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

batch_size = 256
chunk_batches = chunk_list(chunks, batch_size)

def process_batch(batch):
    embeddings = model.encode(batch)  
    records = []
    for i, emb in enumerate(embeddings):
        record = {
            'id': f'doc_{str(uuid.uuid4())}',  
            'values': emb.tolist(),
            'metadata': {'file': file_name,
                         'text': batch[i]}
        }
        
        records.append(record)
    return records

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_batch, batch) for batch in chunk_batches]

    for future in concurrent.futures.as_completed(futures):

        records = future.result()
        index.upsert(vectors=records)  