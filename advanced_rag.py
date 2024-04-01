from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
import weaviate
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline)
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


node_parser = SentenceWindowNodeParser.from_defaults(window_size= 3, window_metadata_key= 'window', original_text_metadata_key='original_text')

# Extract nodes from documents
nodes = node_parser.get_nodes_from_documents(documents)


# Connect to your Weaviate instance
client = weaviate.Client(
    embedded_options=weaviate.embedded.EmbeddedOptions(), 
)
print(f'client is ready : {client.is_ready()}')
index_name = "MyExternalContext"

postproc = MetadataReplacementPostProcessor(target_metadata_key= 'window')
rerank = SentenceTransformerRerank(top_n= 3, model= 'BAAI/bge-reranker-base')
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

query_engine = index.as_query_engine(node_postprocessors = [postproc, rerank], vector_store_query_mode = 'hybrid', alpha = 0.5, similarity_top_k = 6)
response = query_engine.query("what is the capital of India?")
print(response.response)
breakpoint()