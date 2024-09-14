import axios from 'axios';
import dotenv from 'dotenv';
import { PineconeClient } from 'pinecone-client';
import fs from 'fs/promises';
import path from 'path';

// Load environment variables from .env file
dotenv.config();

// Ensure that the environment variables are loaded correctly
const apiKey = process.env.PINECONE_API_KEY;
const environment = process.env.PINECONE_ENVIRONMENT;
const indexName = process.env.PINECONE_INDEX_NAME;
const huggingfaceApiKey = process.env.HUGGINGFACE_API_KEY; // Hugging Face API key

// Check if the required environment variables are set
if (!apiKey || !environment || !indexName || !huggingfaceApiKey) {
  console.error('Missing configuration. Please check your .env file.');
  process.exit(1);
}

// Initialize Pinecone client
const pinecone = new PineconeClient({
  apiKey: apiKey,
  environment: environment,
});

try {
  console.log('Pinecone client initialized successfully.');
} catch (error) {
  console.error('Error initializing Pinecone client:', error);
  process.exit(1);
}

// Function to split text into chunks
function chunkText(text, chunkSize = 512) {
  const words = text.split(' ');
  const chunks = [];
  let chunk = '';

  for (const word of words) {
    if ((chunk + word).length <= chunkSize) {
      chunk += `${word} `;
    } else {
      chunks.push(chunk.trim());
      chunk = `${word} `;
    }
  }

  if (chunk) {
    chunks.push(chunk.trim());
  }

  return chunks;
}

// Function to get embeddings using Hugging Face
async function getEmbeddings(text) {
  try {
    const response = await axios.post(
      'https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2', // Replace with correct Hugging Face model URL
      { inputs: text },
      {
        headers: {
          Authorization: `Bearer ${huggingfaceApiKey}`,
        },
      }
    );

    const embeddings = response.data.embeddings; // Adjust based on the response structure
    return embeddings;
  } catch (error) {
    console.error('Error getting embeddings:', error.response ? error.response.data : error.message);
    return null;
  }
}

// Function to store embeddings in Pinecone
async function storeInPinecone(embeddings, chunk, id) {
  try {
    const index = pinecone.Index(indexName);

    const vector = {
      id,
      values: embeddings,
      metadata: { text: chunk },
    };

    await index.upsert([vector]);
    console.log(`Stored chunk ID ${id} in Pinecone.`);
  } catch (error) {
    console.error('Error storing data in Pinecone:', error);
  }
}

// Main function to process text and store embeddings
async function processText(text, fileName) {
  const chunks = chunkText(text);

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const embeddings = await getEmbeddings(chunk);

    if (embeddings) {
      await storeInPinecone(embeddings, chunk, `${fileName}-chunk-${i + 1}`);
    }
  }
}

// Function to read and process files from a directory
async function processDirectory(directoryPath) {
  try {
    const files = await fs.readdir(directoryPath);

    for (const file of files) {
      const filePath = path.join(directoryPath, file);

      if (path.extname(filePath) === '.txt') {
        try {
          const data = await fs.readFile(filePath, 'utf8');
          console.log(`Processing file: ${file}`);
          await processText(data, file);
        } catch (error) {
          console.error('Error reading file:', error);
        }
      }
    }
  } catch (error) {
    console.error('Error reading directory:', error);
  }
}

// Example usage
const directoryPath = './documents';
await processDirectory(directoryPath);

// Function to search Pinecone with a user's query
async function searchInPinecone(query) {
  const embeddings = await getEmbeddings(query);
  if (!embeddings) {
    console.error('Failed to get embeddings for the query.');
    return;
  }

  try {
    const index = pinecone.Index(indexName);
    const queryResponse = await index.query({
      topK: 5,
      vector: embeddings,
      includeMetadata: true,
    });

    if (queryResponse.matches && queryResponse.matches.length > 0) {
      console.log('Search Results:');
      queryResponse.matches.forEach((match, idx) => {
        console.log(`${idx + 1}. Score: ${match.score}`);
        console.log(`Text: ${match.metadata.text}`);
        console.log('--------------------------------');
      });
    } else {
      console.log('No similar results found.');
    }
  } catch (error) {
    console.error('Error searching Pinecone:', error);
  }
}

// Example search query
const userQuery = 'IIT, JODHPUR';
await searchInPinecone(userQuery);
