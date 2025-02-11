import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain.schema import Document
from transformers import pipeline




pc = Pinecone(api_key="pinecone_api_key" , pool_threads=50)
index = pc.Index("fast-rag2")


GOOGLE_API_KEY = "gemini_api_key"
genai.configure(api_key=GOOGLE_API_KEY)
model = SentenceTransformer('all-MiniLM-L6-v2')  


def get_conversational_chain():

    prompt_template = """
    Answer the question as comprehensively as possible from the provided context. Include:
    - Detailed explanation of concepts
    - Examples where relevant
    - Any important related information
    - Practical applications or implications
    
    If any part of the answer is not available in the context, only provide information that is supported by the context.
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key= GOOGLE_API_KEY,
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(query):
    query_embedding = model.encode(query).tolist()
    
    results = index.query(
        vector=query_embedding,
        top_k=1,
        include_metadata=True
    )

    text_results = []
    metadata_results = []

    for match in results['matches']:
        text = match['metadata'].get('text', 'No text available')
        file_name = match['metadata'].get('file', 'Unknown file')
        text_results.append(text)
        metadata_results.append(file_name)

    full_text = " ".join(text_results)
    docs = [Document(page_content=full_text)]

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": query},
        return_only_outputs=True
    )

    if isinstance(response, dict) and 'output_text' in response:
        return response['output_text']
    return str(response)


summarizer = pipeline("summarization")

def response_generator(query):
    answer_pipe = user_input(query)
    summary = summarizer(query + answer_pipe, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']




user_input("What is cost accounting?")

response2 = response_generator("what is cost accounting?")
print(response2)
