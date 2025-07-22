#!/usr/bin/env python3
"""
Extract transformer-related papers from ArXiv collection
For semantic gravity well analysis
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Set
import numpy as np
from collections import defaultdict
import shutil


class TransformerPaperExtractor:
    """Extract papers related to transformer evolution (2013-2024)"""
    
    def __init__(self, papers_dir: str, output_dir: str):
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Key transformer concepts to track
        self.transformer_keywords = {
            # Pre-transformer era (2013-2016)
            'word2vec': ['word2vec', 'word 2 vec', 'skip-gram', 'cbow'],
            'glove': ['glove', 'global vectors'],
            'lstm': ['lstm', 'long short-term memory'],
            'seq2seq': ['seq2seq', 'sequence to sequence'],
            
            # Transformer era (2017+)
            'attention': ['attention mechanism', 'attention is all you need', 'self-attention', 'multi-head attention'],
            'transformer': ['transformer', 'transformers'],
            'bert': ['bert', 'bidirectional encoder'],
            'gpt': ['gpt', 'generative pre-trained', 'generative pretrained'],
            'vit': ['vision transformer', 'vit'],
            't5': ['t5', 'text-to-text'],
            'llm': ['large language model', 'llm', 'foundation model']
        }
        
        # Gravity well papers (seminal works)
        self.gravity_wells = {
            '1301.3781': 'Word2Vec (Mikolov et al., 2013)',
            '1406.1078': 'GloVe (Pennington et al., 2014)',
            '1409.3215': 'Seq2Seq (Sutskever et al., 2014)',
            '1706.03762': 'Attention Is All You Need (Vaswani et al., 2017)',
            '1810.04805': 'BERT (Devlin et al., 2018)',
            '2005.14165': 'GPT-3 (Brown et al., 2020)',
            '2010.11929': 'Vision Transformer (Dosovitskiy et al., 2020)'
        }
        
    def extract_papers(self, limit: int = None) -> Dict:
        """Extract transformer-related papers"""
        json_files = glob.glob(os.path.join(self.papers_dir, "*.json"))
        if limit:
            json_files = json_files[:limit]
            
        print(f"Processing {len(json_files)} papers...")
        
        # Categories for analysis
        extracted = {
            'gravity_wells': [],
            'transformer_papers': [],
            'attention_papers': [],
            'pre_transformer': [],
            'timeline': defaultdict(list)
        }
        
        keyword_counts = defaultdict(int)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    paper = json.load(f)
                
                # Check if it's a gravity well paper
                arxiv_id = paper.get('arxiv_id', '').replace('v1', '').replace('v2', '').replace('v3', '')
                base_id = arxiv_id.split('v')[0]
                
                if any(base_id.startswith(gw_id) for gw_id in self.gravity_wells):
                    paper['is_gravity_well'] = True
                    paper['gravity_well_name'] = next(
                        self.gravity_wells[gw_id] for gw_id in self.gravity_wells 
                        if base_id.startswith(gw_id)
                    )
                    extracted['gravity_wells'].append(paper)
                    print(f"Found gravity well: {paper['gravity_well_name']}")
                
                # Check for transformer keywords
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()
                text = title + ' ' + abstract
                
                found_keywords = set()
                for category, keywords in self.transformer_keywords.items():
                    if any(kw in text for kw in keywords):
                        found_keywords.add(category)
                        keyword_counts[category] += 1
                
                if found_keywords:
                    paper['transformer_keywords'] = list(found_keywords)
                    year = paper.get('year', 0)
                    
                    # Categorize by era
                    if year < 2017:
                        extracted['pre_transformer'].append(paper)
                    elif 'transformer' in found_keywords or 'attention' in found_keywords:
                        extracted['transformer_papers'].append(paper)
                    
                    if 'attention' in found_keywords:
                        extracted['attention_papers'].append(paper)
                    
                    # Add to timeline
                    if 2013 <= year <= 2024:
                        extracted['timeline'][year].append(paper)
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # Summary statistics
        print("\n=== EXTRACTION SUMMARY ===")
        print(f"Gravity wells found: {len(extracted['gravity_wells'])}")
        print(f"Transformer papers: {len(extracted['transformer_papers'])}")
        print(f"Attention papers: {len(extracted['attention_papers'])}")
        print(f"Pre-transformer era: {len(extracted['pre_transformer'])}")
        
        print("\nKeyword distribution:")
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {keyword}: {count}")
        
        print("\nPapers by year:")
        for year in sorted(extracted['timeline'].keys()):
            print(f"  {year}: {len(extracted['timeline'][year])}")
        
        return extracted
    
    def save_extracted_papers(self, extracted: Dict):
        """Save extracted papers to output directory"""
        
        # Save gravity wells
        gravity_dir = os.path.join(self.output_dir, 'gravity_wells')
        os.makedirs(gravity_dir, exist_ok=True)
        
        for paper in extracted['gravity_wells']:
            json_name = paper.get('local_pdf', '').replace('.pdf', '.json')
            if json_name:
                src_json = os.path.join(self.papers_dir, json_name)
                src_pdf = os.path.join(self.papers_dir, paper.get('local_pdf', ''))
                
                if os.path.exists(src_json):
                    shutil.copy2(src_json, gravity_dir)
                if os.path.exists(src_pdf):
                    shutil.copy2(src_pdf, gravity_dir)
        
        # Save summary JSON
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'total_papers_processed': len(glob.glob(os.path.join(self.papers_dir, "*.json"))),
            'gravity_wells': [
                {
                    'arxiv_id': p.get('arxiv_id'),
                    'title': p.get('title'),
                    'year': p.get('year'),
                    'gravity_well_name': p.get('gravity_well_name')
                }
                for p in extracted['gravity_wells']
            ],
            'statistics': {
                'transformer_papers': len(extracted['transformer_papers']),
                'attention_papers': len(extracted['attention_papers']),
                'pre_transformer': len(extracted['pre_transformer'])
            },
            'timeline_summary': {
                str(year): len(papers) for year, papers in extracted['timeline'].items()
            }
        }
        
        with open(os.path.join(self.output_dir, 'extraction_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save paper lists for different categories
        categories = ['transformer_papers', 'attention_papers', 'pre_transformer']
        for category in categories:
            papers_list = [
                {
                    'arxiv_id': p.get('arxiv_id'),
                    'title': p.get('title'),
                    'year': p.get('year'),
                    'keywords': p.get('transformer_keywords', [])
                }
                for p in extracted[category][:100]  # Top 100 per category
            ]
            
            with open(os.path.join(self.output_dir, f'{category}.json'), 'w') as f:
                json.dump(papers_list, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}")
    
    def find_citation_networks(self, extracted: Dict) -> Dict:
        """Find papers that cite gravity wells (simplified version)"""
        # This would require parsing references from PDFs
        # For now, we'll identify potential citations by keyword overlap
        
        citation_clusters = defaultdict(list)
        
        for well in extracted['gravity_wells']:
            well_keywords = set(well.get('transformer_keywords', []))
            well_year = well.get('year', 0)
            
            for paper in extracted['transformer_papers']:
                paper_keywords = set(paper.get('transformer_keywords', []))
                paper_year = paper.get('year', 0)
                
                # Potential citation if: same keywords, published after, similar category
                if (paper_year > well_year and 
                    len(well_keywords & paper_keywords) > 0 and
                    paper.get('arxiv_id') != well.get('arxiv_id')):
                    
                    citation_clusters[well.get('gravity_well_name', 'Unknown')].append({
                        'arxiv_id': paper.get('arxiv_id'),
                        'title': paper.get('title'),
                        'year': paper.get('year'),
                        'shared_keywords': list(well_keywords & paper_keywords)
                    })
        
        return citation_clusters


def main():
    """Extract transformer papers for semantic gravity analysis"""
    
    papers_dir = "/home/todd/olympus/Erebus/unstructured/papers"
    output_dir = "/home/todd/reconstructionism/validation/data/transformer_papers"
    
    extractor = TransformerPaperExtractor(papers_dir, output_dir)
    
    # Extract papers
    extracted = extractor.extract_papers()
    
    # Find citation networks
    citation_clusters = extractor.find_citation_networks(extracted)
    
    print("\n=== CITATION CLUSTERS ===")
    for gravity_well, citing_papers in citation_clusters.items():
        print(f"\n{gravity_well}: {len(citing_papers)} potential citations")
        if citing_papers:
            print("  Sample citations:")
            for paper in citing_papers[:3]:
                print(f"    - {paper['title'][:60]}... ({paper['year']})")
    
    # Save results
    extractor.save_extracted_papers(extracted)
    
    # Save citation clusters
    with open(os.path.join(output_dir, 'citation_clusters.json'), 'w') as f:
        json.dump(citation_clusters, f, indent=2)
    
    print("\nExtraction complete! Ready for semantic gravity analysis.")


if __name__ == "__main__":
    main()