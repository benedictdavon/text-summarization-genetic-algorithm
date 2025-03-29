import numpy as np
import os
from datasets import load_dataset

def print_file_size(filename):
    print(f"File Size: {os.path.getsize(filename) / 1e6:.2f} MB")

def load_custom_dataset(end_index=None):
    ds = load_dataset("antash420/text-summarization-alpaca-format")
    inputs = ds['train']['input'][:end_index]
    references = ds['train']['output'][:end_index]
    return inputs, references

def load_embeddings_in_chunks(filename, chunk_size=100000, end_index=None):
    try:
        # Memory-mapped file to reduce memory usage
        mmap_mode = 'r'
        
        # Load full array
        all_embeddings = np.load(filename, mmap_mode=mmap_mode)
        
        # Slice to prevent memory overflow
        embeddings_subset = all_embeddings[:chunk_size]
        
        if end_index:
            embeddings_subset = embeddings_subset[:end_index]
        
        print(f"Total embeddings: {len(all_embeddings)}")
        print(f"Loading first {chunk_size} embeddings")
        print(f"Embedding dimension: {embeddings_subset.shape[1]}")
        
        return embeddings_subset
    
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None