import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser
import weaviate
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import openai
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline)
import torch
from custom_model import Llama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import SimpleDirectoryReader

base_model_name = "NousResearch/Llama-2-7b-chat-hf"

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right" 

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,   
    device_map='cpu')

Settings.llm = Llama(model=base_model, tokenizer= llama_tokenizer)
Settings.embed_model = HuggingFaceEmbedding(model_name= 'BAAI/bge-small-en-v1.5')


# sample_text = 'what is your name? what do you do?'
# response = Llama(model= base_model, tokenizer = llama_tokenizer).complete(sample_text)
# print(response.text)

# Load data
documents = SimpleDirectoryReader(
        input_files=["./data/paul_graham_essay.txt"]
).load_data()


node_parser = SimpleNodeParser.from_defaults(chunk_size=256)

# Extract nodes from documents
nodes = node_parser.get_nodes_from_documents(documents)


# Connect to your Weaviate instance
client = weaviate.Client(
    embedded_options=weaviate.embedded.EmbeddedOptions(), 
)
print(f'client is ready : {client.is_ready()}')
index_name = "MyExternalContext"

# Construct vector store
vector_store = WeaviateVectorStore(
    weaviate_client = client, 
    index_name = index_name
)

# Set up the storage for the embeddings
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Setup the index
# build VectorStoreIndex that takes care of chunking documents
# and encoding chunks to embeddings for future retrieval
index = VectorStoreIndex(
    nodes,
    storage_context = storage_context,
)

query_engine = index.as_query_engine()
response = query_engine.query("what is the capital of India?")

breakpoint()