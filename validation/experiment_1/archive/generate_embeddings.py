#!/usr/bin/env python3
"""
Generate semantic embeddings for the WHAT dimension
Options for embedding generation
"""

import json
import os
import glob
import numpy as np
from typing import Dict, List
import time
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate embeddings for paper abstracts and titles"""
    
    def __init__(self, method='sentence-transformers'):
        self.method = method
        self.model = None
        
    def initialize_model(self):
        """Initialize the embedding model based on method"""
        
        if self.method == 'sentence-transformers':
            # Option 1: Local sentence-transformers (free, good quality)
            try:
                from sentence_transformers import SentenceTransformer
                # SPECTER is specifically trained on scientific papers
                self.model = SentenceTransformer('allenai/specter2')
                print("Loaded SPECTER2 model (768 dimensions)")
            except ImportError:
                print("Please install: pip install sentence-transformers")
                return False
                
        elif self.method == 'openai':
            # Option 2: OpenAI embeddings (requires API key)
            try:
                import openai
                # Requires: export OPENAI_API_KEY="your-key"
                self.model = openai.Embedding
                print("Using OpenAI embeddings (1536 dimensions)")
            except ImportError:
                print("Please install: pip install openai")
                return False
                
        elif self.method == 'jina':
            # Option 3: Jina embeddings (as specified in your framework)
            print("Jina embedding options:")
            print("1. Jina Cloud API: https://jina.ai/embeddings/")
            print("2. Local Jina: pip install jina")
            print("3. Use jina-embeddings-v2-base-en (512 dims)")
            return False
            
        return True
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        
        if self.method == 'sentence-transformers':
            # Returns numpy array directly
            return self.model.encode(text, convert_to_numpy=True)
            
        elif self.method == 'openai':
            # OpenAI API call
            response = self.model.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response['data'][0]['embedding'])
            
        elif self.method == 'test':
            # For testing: random embeddings
            return np.random.randn(1024)
    
    def process_papers(self, papers_dir: str, output_dir: str, limit: int = None):
        """Process papers and add embeddings to JSON files"""
        
        json_files = glob.glob(os.path.join(papers_dir, "*.json"))
        if limit:
            json_files = json_files[:limit]
            
        print(f"\nProcessing {len(json_files)} papers...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        processed = 0
        errors = 0
        
        for json_file in tqdm(json_files):
            try:
                # Load paper
                with open(json_file, 'r') as f:
                    paper = json.load(f)
                
                # Create text for embedding (title + abstract)
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                text = f"{title}. {abstract}"
                
                # Generate embedding
                embedding = self.generate_embedding(text)
                
                # Update paper data
                if 'dimensions' not in paper:
                    paper['dimensions'] = {}
                if 'WHAT' not in paper['dimensions']:
                    paper['dimensions']['WHAT'] = {}
                    
                paper['dimensions']['WHAT']['embeddings'] = embedding.tolist()
                paper['dimensions']['WHAT']['embedding_method'] = self.method
                paper['dimensions']['WHAT']['embedding_dim'] = len(embedding)
                
                # Save to output directory
                output_file = os.path.join(output_dir, os.path.basename(json_file))
                with open(output_file, 'w') as f:
                    json.dump(paper, f, indent=2)
                    
                processed += 1
                
                # Rate limiting for APIs
                if self.method in ['openai', 'jina']:
                    time.sleep(0.1)  # Avoid rate limits
                    
            except Exception as e:
                print(f"\nError processing {json_file}: {e}")
                errors += 1
                
        print(f"\nProcessed: {processed} papers")
        print(f"Errors: {errors}")
        
        return processed, errors


def estimate_time_and_cost(num_papers: int, method: str):
    """Estimate time and cost for embedding generation"""
    
    print(f"\n=== ESTIMATE FOR {num_papers} PAPERS ===")
    
    if method == 'sentence-transformers':
        # Local processing
        time_per_paper = 0.1  # seconds
        total_time = num_papers * time_per_paper
        print(f"Method: Local SPECTER2 (768 dims)")
        print(f"Cost: FREE")
        print(f"Time: ~{total_time/60:.1f} minutes")
        print(f"Quality: Good for scientific papers")
        
    elif method == 'openai':
        # OpenAI API
        cost_per_1k_tokens = 0.00002  # text-embedding-3-small
        avg_tokens_per_paper = 500
        total_cost = (num_papers * avg_tokens_per_paper / 1000) * cost_per_1k_tokens
        print(f"Method: OpenAI API (1536 dims)")
        print(f"Cost: ~${total_cost:.2f}")
        print(f"Time: ~{num_papers * 0.2 / 60:.1f} minutes (with rate limiting)")
        print(f"Quality: Excellent general purpose")
        
    elif method == 'jina':
        # Jina API
        print(f"Method: Jina v4 (1024 dims as specified)")
        print(f"Cost: Check https://jina.ai/embeddings/ for pricing")
        print(f"Time: Similar to OpenAI with rate limiting")
        print(f"Quality: Excellent, your specified model")


def main():
    """Generate embeddings for papers"""
    
    print("=== EMBEDDING GENERATION FOR INFORMATION RECONSTRUCTIONISM ===")
    
    # Show current status
    papers_dir = "/home/todd/olympus/Erebus/unstructured/papers"
    json_files = glob.glob(os.path.join(papers_dir, "*.json"))
    print(f"\nTotal papers available: {len(json_files)}")
    
    # Check a sample to see if any have embeddings
    sample_file = json_files[0]
    with open(sample_file, 'r') as f:
        sample = json.load(f)
    
    has_embeddings = (
        'dimensions' in sample and 
        'WHAT' in sample['dimensions'] and 
        'embeddings' in sample['dimensions']['WHAT'] and 
        sample['dimensions']['WHAT']['embeddings'] is not None
    )
    
    print(f"Sample paper has embeddings: {has_embeddings}")
    
    # Estimate for different options
    estimate_time_and_cost(1000, 'sentence-transformers')
    estimate_time_and_cost(1000, 'openai')
    estimate_time_and_cost(1000, 'jina')
    
    print("\n=== RECOMMENDED APPROACH ===")
    print("1. Start with sentence-transformers/SPECTER2 for FREE embeddings")
    print("2. Process 1000 papers as proof-of-concept")
    print("3. These embeddings will enable:")
    print("   - Semantic similarity calculation")
    print("   - Context score computation")
    print("   - Gravity well visualization")
    print("   - Full framework validation")
    
    print("\nTo proceed:")
    print("1. Install: pip install sentence-transformers")
    print("2. Run: python generate_embeddings.py --method sentence-transformers --limit 1000")
    
    # Uncomment to actually run:
    # generator = EmbeddingGenerator(method='sentence-transformers')
    # if generator.initialize_model():
    #     output_dir = "/home/todd/reconstructionism/validation/data/papers_with_embeddings"
    #     generator.process_papers(papers_dir, output_dir, limit=100)


if __name__ == "__main__":
    main()