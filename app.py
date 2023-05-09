pip install -U openai pinecone-client flask flask-cors python-dotenv

import os
import openai
import pinecone
from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
import json
from PyPDF2 import PdfFileReader
from io import BytesIO
import requests

load_dotenv()


openai_api_key = os.environ.get('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')
pinecone_api_key = os.environ.get('2453a2f7-e073-4363-857d-95a159049b4c')
pinecone_env = os.environ.get('us-east4-gcp')

app = Flask(__name__)
CORS(app)

# Test default route
@app.route('/')
def hello_world():
    return {"Hello":"World"}


openai.api_key = openai_api_key
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

def load_and_split_pdf(file_url):
    response = requests.get(file_url)
    pdf = PdfFileReader(BytesIO(response.content))
    docs = []
    for page_num in range(pdf.getNumPages()):
        page_text = pdf.getPage(page_num).extractText()
        docs.append(page_text)
    return docs

def generate_embeddings(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"]["embedding"]

def upsert_pinecone(collection_name, docs, embeddings):
    index_name = collection_name
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)
    index = pinecone.Index(index_name)
    for doc, embedding in zip(docs, embeddings):
        index.upsert_item(doc, embedding)

@app.route('/embed', methods=['POST'])
def embed_pdf():
    collection_name = request.json.get("collection_name")
    file_url = request.json.get("file_url")
    docs = load_and_split_pdf(file_url)
    embeddings = [generate_embeddings(doc) for doc in docs]
    upsert_pinecone(collection_name, docs, embeddings)
    return {"status": "success"}


@app.route('/retrieve', methods=['POST'])
def retrieve_info():
    collection_name = request.json.get("collection_name")
    query = request.json.get("query")

    # Generate query embedding
    query_embedding = generate_embeddings(query)

    # Perform similarity search in Pinecone
    index = pinecone.Index(collection_name)
    search_results = index.fetch_top_k(query_embedding, k=2)

    # Prepare input for GPT-3.5 Turbo
    input_documents = [{"text": doc} for doc in search_results]

    # Query GPT-3.5 Turbo
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=f"Answer the following question based on the given documents: {query}",
        documents=input_documents,
        temperature=0.2,
        max_tokens=100,
    )

    # Extract the answer
    answer = response.choices[0].text.strip()

    return {"results": answer}