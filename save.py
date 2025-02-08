from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

file_name = r"D:\Projects(main)\Pdf vocalquery bot\Bcom.Hons Financial&Accounting.pdf"


loader = PyMuPDFLoader(file_name)
documents = loader.load()

non_empty_documents = [doc for doc in documents if doc.page_content.strip()]

full_text = " ".join([doc.page_content for doc in non_empty_documents])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_text(full_text)