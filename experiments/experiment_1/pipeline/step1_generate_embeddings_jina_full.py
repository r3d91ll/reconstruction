#!/usr/bin/env python3
"""
Generate embeddings using Jina V4 model with FULL PDF content
Handles enhanced JSON files with pdf_content field
"""

import os
import json
import torch
from glob import glob
from transformers import JinaEmbeddingsV4Model
import numpy as np

def extract_full_text(paper):
    """Extract all text content from enhanced paper JSON"""
    
    # Start with basic metadata
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    categories = ', '.join(paper.get('categories', []))
    
    # Build base text
    text_parts = [
        f"Title: {title}",
        f"Categories: {categories}",
        f"Abstract: {abstract}"
    ]
    
    # Add full PDF content if available
    if 'pdf_content' in paper:
        pdf_content = paper['pdf_content']
        
        # Add full markdown content
        if 'markdown' in pdf_content:
            text_parts.append("\n# Full Paper Content\n")
            text_parts.append(pdf_content['markdown'])
        
        # Add image captions
        if 'images' in pdf_content:
            text_parts.append("\n## Figures and Images\n")
            for img in pdf_content['images']:
                text_parts.append(f"Figure {img.get('figure_id', '')}: {img.get('caption', '')}")
                if 'context' in img:
                    text_parts.append(f"Context: {img['context']}")
        
        # Add table captions
        if 'tables' in pdf_content:
            text_parts.append("\n## Tables\n")
            for table in pdf_content['tables']:
                text_parts.append(f"Table {table.get('table_id', '')}: {table.get('caption', '')}")
                if 'markdown' in table:
                    text_parts.append(table['markdown'])
        
        # Add code blocks
        if 'code_blocks' in pdf_content:
            text_parts.append("\n## Code Examples\n")
            for code in pdf_content['code_blocks']:
                lang = code.get('language', 'unknown')
                section = code.get('section', 'Unknown section')
                text_parts.append(f"Code ({lang}) from {section}:")
                text_parts.append(code.get('content', ''))
        
        # Add references
        if 'references' in pdf_content:
            text_parts.append("\n## References\n")
            for ref in pdf_content['references'][:20]:  # Limit references
                text_parts.append(ref.get('text', ''))
    
    return '\n\n'.join(text_parts)

def main():
    # Configuration
    output_dir = os.environ.get('VALIDATION_OUTPUT_DIR', './data/papers_with_embeddings')
    papers_dir = "/home/todd/olympus/Erebus/unstructured/papers/"
    
    # Specify limit via environment variable or default
    limit = int(os.environ.get('PAPER_LIMIT', 100))
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Jina V4 model...")
    model = JinaEmbeddingsV4Model.from_pretrained(
        'jinaai/jina-embeddings-v4',
        trust_remote_code=True
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Model config: {model.config.hidden_size} dimensions")
    print(f"Max sequence length: {model.config.max_position_embeddings}")
    
    # Get papers
    json_files = glob(os.path.join(papers_dir, "*.json"))[:limit]
    print(f"\nProcessing {len(json_files)} papers with FULL content...")
    
    success = 0
    has_full_content = 0
    
    for i, json_file in enumerate(json_files):
        try:
            # Load paper
            with open(json_file, 'r') as f:
                paper = json.load(f)
            
            # Check if we have full content
            if 'pdf_content' in paper:
                has_full_content += 1
                print(f"\n[{i+1}/{len(json_files)}] Processing (with PDF): {paper.get('title', 'Unknown')[:60]}...")
            else:
                print(f"\n[{i+1}/{len(json_files)}] Processing (metadata only): {paper.get('title', 'Unknown')[:60]}...")
            
            # Extract all available text
            text = extract_full_text(paper)
            
            # Report content size
            print(f"  Text length: {len(text)} characters")
            if 'pdf_content' in paper:
                metadata = paper['pdf_content'].get('metadata', {})
                print(f"  Pages: {metadata.get('num_pages', 'unknown')}")
                print(f"  Figures: {metadata.get('num_figures', 0)}")
                print(f"  Tables: {metadata.get('num_tables', 0)}")
                print(f"  Code blocks: {metadata.get('num_code_blocks', 0)}")
            
            # Truncate if needed (Jina V4 can handle 128K tokens)
            # Approximate 1 token = 4 characters
            max_chars = 128000 * 4  # ~512K characters
            if len(text) > max_chars:
                print(f"  Warning: Truncating from {len(text)} to {max_chars} characters")
                text = text[:max_chars]
            
            # Generate embedding
            embeddings = model.encode_text(
                texts=[text],
                task="retrieval",
                prompt_name="passage",
            )
            
            # encode_text returns numpy array
            embedding = embeddings[0]
            print(f"  Generated embedding shape: {embedding.shape}")
            
            # Add to paper data
            if 'dimensions' not in paper:
                paper['dimensions'] = {}
            if 'WHAT' not in paper['dimensions']:
                paper['dimensions']['WHAT'] = {}
                
            paper['dimensions']['WHAT']['embeddings'] = embedding.tolist()
            paper['dimensions']['WHAT']['embedding_dim'] = len(embedding)
            paper['dimensions']['WHAT']['embedding_method'] = 'jina-v4-full-content'
            paper['dimensions']['WHAT']['context_length'] = len(text)
            paper['dimensions']['WHAT']['has_full_content'] = 'pdf_content' in paper
            
            # Save both embeddings and full content
            paper['embeddings'] = embedding.tolist()  # For compatibility
            
            # Save
            output_file = os.path.join(output_dir, os.path.basename(json_file))
            with open(output_file, 'w') as f:
                json.dump(paper, f)
                
            success += 1
            print(f"  ✓ Saved to {output_file}")
            
        except Exception as e:
            print(f"\n✗ Error processing {json_file}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total processed: {success}/{len(json_files)}")
    print(f"  With full PDF content: {has_full_content}")
    print(f"  Metadata only: {success - has_full_content}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()