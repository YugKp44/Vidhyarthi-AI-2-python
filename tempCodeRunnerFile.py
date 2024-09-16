import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

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
            dimension=384,  # Dimension of the embeddings
            metric='cosine',  # Use cosine similarity
            spec=ServerlessSpec(
                cloud='aws',
                region=environment
            )
        )
        print(f"Created Pinecone index '{index_name}' with dimension 384.")
except Exception as e:
    print(f"Error creating Pinecone index: {e}")
    exit(1)

# Function to split text into chunks
def chunk_text(text, chunk_size=512):
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

# Function to get embeddings using Hugging Face
def get_embeddings(text):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(text)

# Function to store embeddings in Pinecone
def store_in_pinecone(embeddings, chunk, id):
    try:
        index = pc.Index(index_name)  # Use Index class to access the index
        vector = {
            'id': id,
            'values': embeddings.tolist(),  # Convert numpy array to list
            'metadata': {'text': chunk}
        }
        index.upsert([vector])
        print(f"Stored chunk ID {id} in Pinecone.")
    except Exception as e:
        print(f"Error storing data in Pinecone: {e}")

# Main function to process text and store embeddings
def process_text(text, file_name):
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embeddings = get_embeddings(chunk)
        store_in_pinecone(embeddings, chunk, f"{file_name}-chunk-{i + 1}")

# Function to read and process files from a directory
def process_directory(directory_path):
    import os
    from pathlib import Path

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

# Function to search Pinecone with a user's query
def search_in_pinecone(query):
    embeddings = get_embeddings(query)
    if embeddings.size == 0:  # Check if embeddings are empty
        print('Failed to get embeddings for the query.')
        return

    try:
        index = pc.Index(index_name)  # Use Index class to access the index
        query_response = index.query(
            vector=embeddings.tolist(),
            top_k=4,  # Limit to top 4 results
            include_metadata=True
        )

        if query_response['matches']:
            print('Search Results:')
            for idx, match in enumerate(query_response['matches']):
                print(f"{idx + 1}. Score: {match['score']}")
                print(f"Text: {match['metadata']['text']}")
                print('--------------------------------')
        else:
            print('No similar results found.')
    except Exception as e:
        print(f"Error searching Pinecone: {e}")

# Example usage
directory_path = './documents'
process_directory(directory_path)

# Example search query
user_query = 'CAN YOU GIVE ME INFORMATION ABOUT IITJ,JODHPUR?'
search_in_pinecone(user_query)
