import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from pathlib import Path
import ollama  # Importing Ollama for using Nomic embeddings

# Load environment variables
load_dotenv()

# Set environment variables for Pinecone
api_key = os.getenv('PINECONE_API_KEY')
environment = os.getenv('PINECONE_ENVIRONMENT')
index_name = os.getenv('PINECONE_INDEX_NAME')

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Create Pinecone index with correct dimension if it doesn't exist
try:
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,  # Change to match the model's embedding size (based on your chosen model)
            metric='cosine',  # Use cosine similarity
            spec=ServerlessSpec(
                cloud='aws',
                region=environment
            )
        )
        st.success(f"Created Pinecone index '{index_name}' with dimension 768.")
    else:
        index = pc.Index(index_name)
        # Get index stats to check the dimension
        index_stats = index.describe_index_stats()
        if index_stats['dimension'] != 768:
            pc.delete_index(index_name)
            pc.create_index(
                name=index_name,
                dimension=768,  # Ensure dimension matches embeddings
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=environment
                )
            )
            st.success(f"Recreated Pinecone index '{index_name}' with dimension 768.")
except Exception as e:
    st.error(f"Error creating Pinecone index: {e}")
    exit(1)

# Initialize Ollama for embeddings using Nomic
  # Initialize with Nomic model

# Function to get embeddings using Nomic
def get_embeddings(text):
    response = ollama.embeddings(model='nomic-embed-text',prompt=text)  # Use Nomic model for embeddings
    return response['embedding']  # Adjust based on response structure

# Function to split text into chunks
def chunk_text(text, chunk_size=420):  # Define chunk size
    words = text.split(' ')
    chunks = []
    chunk = ''
    
    for word in words:
        if (len(chunk) + len(word)) <= chunk_size:
            chunk += f'{word} '
        else:
            chunks.append(chunk.strip())
            chunk = f'{word} '

    if chunk:
        chunks.append(chunk.strip())

    return chunks

# Function to store embeddings in Pinecone
def store_in_pinecone(embeddings, chunk, id):
    try:
        index = pc.Index(index_name)  # Access the index
        vector = {
            'id': id,
            'values': embeddings,  # Assuming embeddings is already a list
            'metadata': {'text': chunk}
        }
        index.upsert([vector])
        print(f"Stored chunk ID {id} in Pinecone.")
    except Exception as e:
        print(f"Error storing data in Pinecone: {e}")

# Function to process text and store embeddings
def process_text(text, file_name):
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embeddings = get_embeddings(chunk)
        store_in_pinecone(embeddings, chunk, f"{file_name}-chunk-{i + 1}")

# Function to read and process files from a directory
def process_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if Path(file_path).suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='utf8') as f:
                        data = f.read()
                    print(f"Processing file: {file}")
                    process_text(data, file)
                except Exception as e:
                    print(f"Error reading file: {e}")
    except Exception as e:
        print(f"Error reading directory: {e}")

# Pre-train the model with text files (backend)
def pre_train_with_files():
    directory_path = './documents'  # Your directory with text files
    process_directory(directory_path)
    print("Text files have been processed and stored in Pinecone.")

# Function to search Pinecone with a user's query
def search_in_pinecone(query):
    embeddings = get_embeddings(query)
    if not embeddings:  # Check if embeddings are empty
        st.warning('Failed to get embeddings for the query.')
        return

    try:
        index = pc.Index(index_name)  # Access the index
        query_response = index.query(
            vector=embeddings,
            top_k=1,  # Return top result
            include_metadata=True
        )

        if query_response['matches']:
            st.markdown("### 🔍 Similar Results Found:")
            for idx, match in enumerate(query_response['matches']):
                st.markdown(f"**Result {idx + 1}:**")
                st.markdown(f"📝 **Similar Text:** {match['metadata']['text']}")
        else:
            st.warning('No similar results found.')
    except Exception as e:
        st.error(f"Error searching Pinecone: {e}")

# Streamlit UI

st.title("VIDHYARTHI AI")

# Input field for user query
st.markdown("### Ask Questions About Colleges")
user_query = st.text_input("Type your query here:")

if st.button("Search"):
    if user_query:
        search_in_pinecone(user_query)
    else:
        st.warning("Please enter a query to search.")

# Backend process (pre-training with text files)
pre_train_with_files()
