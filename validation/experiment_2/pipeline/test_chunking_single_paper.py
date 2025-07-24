#!/usr/bin/env python3
"""Test chunking on a single paper to debug issues"""

import os
import json
import torch
from transformers import AutoModel
from pathlib import Path

print("=" * 60)
print("SINGLE PAPER CHUNKING TEST")
print("=" * 60)

# Configuration
papers_dir = "/home/todd/olympus/Erebus/unstructured/papers"

# Find a paper with PDF content
test_paper_path = None
for json_path in Path(papers_dir).glob("*.json"):
    try:
        with open(json_path, 'r') as f:
            paper = json.load(f)
        if 'pdf_content' in paper and paper['pdf_content'].get('markdown'):
            test_paper_path = json_path
            break
    except:
        continue

if not test_paper_path:
    print("No papers with PDF content found!")
    exit(1)

print(f"\nTest paper: {test_paper_path.name}")

# Load the paper
with open(test_paper_path, 'r') as f:
    paper = json.load(f)

# Build full text
title = paper.get('title', '')
abstract = paper.get('abstract', '')
markdown = paper['pdf_content']['markdown']
full_text = f"Title: {title}\n\nAbstract: {abstract}\n\n{markdown}"

print(f"Full text length: {len(full_text)} characters")

# Load model
print("\nLoading Jina V4 model...")
model = AutoModel.from_pretrained(
    'jinaai/jina-embeddings-v4',
    trust_remote_code=True
)

device = torch.device('cuda:0')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Test chunking
print("\nTesting chunking...")
chunk_size = 2048
stride = 1536

chunks = []
for start in range(0, len(full_text), stride):
    end = min(start + chunk_size, len(full_text))
    chunk_text = full_text[start:end]
    chunks.append({
        'text': chunk_text,
        'start': start,
        'end': end
    })
    if end >= len(full_text):
        break

print(f"Created {len(chunks)} chunks")

# Test embedding generation
print("\nGenerating embeddings for first 3 chunks...")
with torch.no_grad():
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(chunk['text'])} chars")
        print(f"  Start: {chunk['text'][:50]}...")
        
        # Generate embedding
        embedding = model.encode_text(
            texts=[chunk['text']],
            task="retrieval",
            prompt_name="passage"
        )
        
        if isinstance(embedding, list):
            print(f"  Embedding length: {len(embedding[0])}")
            print(f"  Embedding sample: {embedding[0][:5]}")
        else:
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding sample: {embedding[0][:5].tolist()}")

print("\n" + "=" * 60)
print("TEST COMPLETE - Chunking works!")
print("=" * 60)