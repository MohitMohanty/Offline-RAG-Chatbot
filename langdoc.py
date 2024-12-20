import os
import shutil
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from PIL import Image
import pytesseract
import gradio as gr
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pymongo import MongoClient
import bson
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import hashlib



class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # Embed a list of documents
        return self.model.encode(texts)

    def embed_query(self, text):
        # Embed a single query
        return self.model.encode([text])[0]

# Instantiate the embedding class
embeddings = SentenceTransformerEmbeddings("sentence-transformers/all-mpnet-base-v2")


# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "document_database_doc"
COLLECTION_NAME = "documents_doc"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]


# Initialize variables
UPLOAD_FOLDER = "uploaded_files"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Llama model
MODEL_PATH = "codeup-llama-2-13b-chat-hf.Q4_K_M.gguf"
llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=4096, n_threads=8)


# Embedding model
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize FAISS vector store
vector_store = None

# Conversation history
conversation_history = []

# Generate a unique ID for the document
def generate_document_id(file_name, content):
    unique_string = f"{file_name}:{content}"
    return hashlib.sha256(unique_string.encode()).hexdigest()

# Load stored documents from MongoDB
def load_documents_from_db():
    global vector_store
    documents = []

    for record in collection.find():
        content = record["content"]
        metadata = record["metadata"]
        documents.append({
            "page_content": content,  # Expected by FAISS
            "metadata": metadata
        })

    if documents:
        vector_store = FAISS.from_documents(documents, embeddings)
        print(f"{len(documents)} documents loaded into the FAISS vector store.")
    else:
        vector_store = None
        print("No documents found in the database.")

# Save processed document to MongoDB
def save_document_to_db(file_name, file_path, content, metadata={}):
    document_id = generate_document_id(file_name, content)
    document = {
        "_id": document_id,
        "file_name": file_name,
        "file_path": file_path,
        "content": content,
        "metadata": metadata,
    }
    try:
        collection.insert_one(document)
        print(f"Document {file_name} saved to the database.")
    except Exception as e:
        print(f"Document {file_name} already exists in the database. Skipping. Error: {e}")


# OCR Function
def perform_ocr(file_path):
    if file_path.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    elif file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not text.strip():
            images = convert_from_path(file_path)
            text = "\n".join([pytesseract.image_to_string(img) for img in images])
        return text
    else:
        return None

# Document Processing Function
# Process document, store it in MongoDB, and update FAISS
def process_document(file_paths):
    global vector_store
    messages = []

    for file in file_paths:
        temp_path = os.path.join(UPLOAD_FOLDER, file.name)

        # Avoid SameFileError: Copy only if the destination file does not already exist
        if os.path.abspath(file.name) != os.path.abspath(temp_path):
            shutil.copy(file.name, temp_path)

        # Perform OCR if needed
        if file.name.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            content = perform_ocr(temp_path)
        else:
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

        # Generate a unique ID for the document
        document_id = generate_document_id(file.name, content)

        # Check if the document already exists in the database
        if collection.find_one({"_id": document_id}):
            messages.append(f"{file.name} is already processed and stored in the database.")
            continue

        # Save the document to MongoDB
        save_document_to_db(file.name, temp_path, content)

        # Create document chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = text_splitter.create_documents([content])

        # Initialize or update FAISS vector store
        if vector_store is None:
            vector_store = FAISS.from_documents(documents, embeddings)
        else:
            vector_store.add_documents(documents)

        messages.append(f"{file.name} processed and stored in the database successfully!")

    return "\n".join(messages)


# Generate Response Function
def generate_response(query):
    global conversation_history

    # Check if the vector store is empty
    if vector_store is None or vector_store.index.ntotal == 0:
        return "No documents available. Please upload and process a document first."

    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)

    # Combine the retrieved documents into a single context for the prompt
    context = "\n\n".join([doc.page_content for doc in docs])

    # Construct the prompt
    history_text = "\n".join([f"User: {q}\nBot: {r}" for q, r in conversation_history])
    prompt = (
        f"You are an expert assistant.You can use the context to generate response. Based on the following context , answer the question accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    print(prompt)
    # Pass the prompt to the LLM
    response = llm(prompt, max_tokens=4096)

    # Update conversation history
    conversation_history.append((query, response))

    return response

# Clear Conversation History
def clear_history():
    global conversation_history
    conversation_history = []
    return "History cleared!"

# Gradio Interface
def query_document(query):
    return generate_response(query)

with gr.Blocks(theme=gr.themes.Default()) as app:
    gr.Markdown("# 🚀 **OCR and RAG-Powered Assistant**")
    gr.Markdown("Upload your documents and interact with an AI assistant capable of answering questions based on the uploaded files.")

    with gr.Tab("📁 Document Upload"):
        with gr.Row():
            upload_files = gr.Files(label="Upload Files (PDF, Images, Text)")
            upload_button = gr.Button("Upload and Process")
        upload_output = gr.Textbox(label="Processing Status", lines=2, interactive=False)

    with gr.Tab("💬 Ask Questions"):
        with gr.Row():
            query_input = gr.Textbox(label="Ask a Question", placeholder="Enter your query here...", interactive=True)
            query_button = gr.Button("Generate Answer")
            clear_button = gr.Button("Clear History")
        query_output = gr.Textbox(label="AI Response", interactive=False, lines=6)

    with gr.Tab("🗑 Clear History"):
        clear_output = gr.Textbox(label="Clear Status", interactive=False)

    # Button Actions
    upload_button.click(process_document, inputs=upload_files, outputs=upload_output)
    query_button.click(query_document, inputs=query_input, outputs=query_output)
    clear_button.click(clear_history, inputs=[], outputs=clear_output)

app.launch()

