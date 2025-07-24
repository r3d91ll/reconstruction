#!/usr/bin/env python3
"""
Simplified late chunking script - single GPU version
"""

import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoModel
from tqdm import tqdm
from datetime import datetime

def main():
    # Configuration from environment
    papers_dir = os.environ.get('PAPERS_DIR', "/home/todd/olympus/Erebus/unstructured/papers")
    output_dir = os.environ.get('OUTPUT_DIR', "/home/todd/reconstructionism/validation/experiment_2/data/chunks")
    num_papers = int(os.environ.get('LIMIT', '20'))
    
    print(f"Configuration:")
    print(f"  Papers dir: {papers_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Num papers: {num_papers}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        return 1
    
    device = torch.device('cuda:0')
    print(f"  Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("\nLoading Jina V4 model...")
    model = AutoModel.from_pretrained(
        'jinaai/jina-embeddings-v4',
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Get papers with PDF content
    papers_to_process = []
    for json_path in Path(papers_dir).glob("*.json"):
        try:
            with open(json_path, 'r') as f:
                paper = json.load(f)
            if 'pdf_content' in paper and paper['pdf_content'].get('markdown'):
                papers_to_process.append(str(json_path))
            if len(papers_to_process) >= num_papers:
                break
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {json_path}: {e}")
            continue
    
    print(f"\nProcessing {len(papers_to_process)} papers...")
    
    # Process papers
    for json_path in tqdm(papers_to_process, desc="Processing papers"):
        try:
            # Load paper
            with open(json_path, 'r') as f:
                paper = json.load(f)
            
            paper_id = paper.get('id', Path(json_path).stem)
            
            # Build full text
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            markdown = paper['pdf_content']['markdown']
            full_text = f"Title: {title}\n\nAbstract: {abstract}\n\n{markdown}"
            
            # Truncate if too long
            max_chars = int(os.environ.get('MAX_TEXT_CHARS', '512000'))
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars]
            
            # Create chunks
            chunk_size = 2048
            stride = 1536
            chunks_data = []
            
            for start in range(0, len(full_text), stride):
                end = min(start + chunk_size, len(full_text))
                chunk_text = full_text[start:end]
                
                # Generate embedding
                with torch.no_grad():
                    embeddings = model.encode_text(
                        texts=[chunk_text],
                        task="retrieval",
                        prompt_name="passage"
                    )
                    
                    # Handle embedding format
                    if isinstance(embeddings, list):
                        embedding_vector = embeddings[0]
                    else:
                        embedding_vector = embeddings[0]
                    
                    if hasattr(embedding_vector, 'cpu'):
                        embedding_list = embedding_vector.cpu().tolist()
                    else:
                        embedding_list = list(embedding_vector)
                
                chunk_data = {
                    '_key': f"{paper_id.replace('.', '_')}_chunk_{len(chunks_data)}",
                    'paper_id': paper_id,
                    'chunk_index': len(chunks_data),
                    'text': chunk_text[:int(os.environ.get('CHUNK_TEXT_PREVIEW_LENGTH', '1000'))] + "..." if len(chunk_text) > int(os.environ.get('CHUNK_TEXT_PREVIEW_LENGTH', '1000')) else chunk_text,
                    'embedding': embedding_list,
                    'metadata': {
                        'start_index': start,
                        'end_index': end,
                        'length': len(chunk_text),
                        'has_full_content': True
                    }
                }
                chunks_data.append(chunk_data)
                
                if end >= len(full_text):
                    break
            
            # Save chunks
            output_path = os.path.join(output_dir, f"{paper_id.replace('.', '_')}_chunks.json")
            result = {
                'paper_id': paper_id,
                'title': title,
                'num_chunks': len(chunks_data),
                'chunks': chunks_data,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(result, f)
                
        except json.JSONDecodeError as e:
            print(f"\nJSON decode error processing {Path(json_path).name}: {e}")
            continue
        except IOError as e:
            print(f"\nIO error processing {Path(json_path).name}: {e}")
            continue
        except torch.cuda.OutOfMemoryError as e:
            print(f"\nCUDA out of memory error processing {Path(json_path).name}: {e}")
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:
            print(f"\nRuntime error processing {Path(json_path).name}: {e}")
            continue
    
    # Summary
    chunk_files = list(Path(output_dir).glob("*_chunks.json"))
    print(f"\n{'='*60}")
    print(f"Late Chunking Complete!")
    print(f"Generated {len(chunk_files)} chunk files")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())