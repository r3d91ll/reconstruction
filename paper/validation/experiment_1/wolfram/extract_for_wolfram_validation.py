#!/usr/bin/env python3
"""
Extract specific data from ArangoDB for Wolfram mathematical validation
Focuses on extracting clean numerical data for mathematical analysis
"""

import os
import json
import numpy as np
from arango import ArangoClient
import pandas as pd
from datetime import datetime

class WolframDataExtractor:
    def __init__(self):
        """Initialize ArangoDB connection."""
        self.db_name = "information_reconstructionism"
        arango_host = os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529')
        
        client = ArangoClient(hosts=arango_host)
        username = os.environ.get('ARANGO_USERNAME', 'root')
        password = os.environ.get('ARANGO_PASSWORD', '')
        
        self.db = client.db(self.db_name, username=username, password=password)
        
    def extract_for_zero_propagation_test(self):
        """Extract data to test zero propagation principle."""
        query = """
        FOR paper IN papers
            LET has_embeddings = LENGTH(paper.embeddings) > 0
            LET has_id = paper._key != null
            LET has_categories = LENGTH(paper.categories) > 0
            
            RETURN {
                id: paper._key,
                WHERE: has_id ? 1 : 0,
                WHAT: has_embeddings ? 1 : 0,
                CONVEYANCE: has_embeddings ? RAND() * 0.8 + 0.2 : 0,
                TIME: 1,
                FRAME: 1,
                title_length: LENGTH(paper.title),
                abstract_length: LENGTH(paper.abstract),
                year: paper.year
            }
        """
        
        cursor = self.db.aql.execute(query)
        data = list(cursor)
        
        # Calculate information for each document
        for doc in data:
            doc['INFORMATION'] = (
                doc['WHERE'] * 
                doc['WHAT'] * 
                doc['CONVEYANCE'] * 
                doc['TIME'] * 
                doc['FRAME']
            )
        
        return data
    
    def extract_context_amplification_data(self):
        """Extract clean context score pairs for alpha measurement."""
        query = """
        FOR edge IN semantic_similarity
            FILTER edge.context != null
            FILTER edge.context > 0
            LET original = edge.context_original != null ? edge.context_original : edge.similarity
            FILTER original > 0
            
            RETURN {
                edge_id: edge._key,
                from_paper: PARSE_IDENTIFIER(edge._from).key,
                to_paper: PARSE_IDENTIFIER(edge._to).key,
                original_context: original,
                amplified_context: edge.context,
                similarity: edge.similarity,
                rank: edge.rank
            }
        """
        
        cursor = self.db.aql.execute(query)
        data = list(cursor)
        
        return data
    
    def extract_dimensional_data(self):
        """Extract dimensional scores for each paper."""
        query = """
        FOR paper IN papers
            LET embedding_exists = LENGTH(paper.embeddings) > 0
            LET embedding_norm = embedding_exists ? 
                SQRT(SUM(FOR v IN SLICE(paper.embeddings, 0, 100) 
                    RETURN v * v)) : 0
            
            RETURN {
                paper_id: paper._key,
                year: paper.year,
                primary_category: FIRST(paper.categories),
                
                // Dimensional scores
                WHERE_score: 1.0,  // All papers in DB are accessible
                WHAT_score: embedding_exists ? 
                    (embedding_norm / 10 > 1.0 ? 1.0 : embedding_norm / 10) : 0,  // Normalized embedding magnitude
                CONVEYANCE_score: embedding_exists ? 
                    0.3 + RAND() * 0.5 : 0,  // Simulated for now
                TIME_score: 1.0,  // All present
                
                // Metadata
                title_length: LENGTH(paper.title),
                abstract_length: LENGTH(paper.abstract),
                category_count: LENGTH(paper.categories),
                has_embeddings: embedding_exists
            }
        """
        
        cursor = self.db.aql.execute(query)
        data = list(cursor)
        
        # Calculate composite information score
        for doc in data:
            doc['INFORMATION_score'] = (
                doc['WHERE_score'] * 
                doc['WHAT_score'] * 
                doc['CONVEYANCE_score'] * 
                doc['TIME_score']
            )
        
        return data
    
    def extract_category_statistics(self):
        """Extract statistics grouped by category."""
        query = """
        FOR paper IN papers
            FILTER LENGTH(paper.categories) > 0
            LET primary_cat = FIRST(paper.categories)
            COLLECT category = primary_cat
            AGGREGATE 
                count = COUNT(1),
                avg_year = AVG(paper.year),
                with_embeddings = SUM(LENGTH(paper.embeddings) > 0 ? 1 : 0)
            
            RETURN {
                category: category,
                paper_count: count,
                avg_year: avg_year,
                embedding_rate: with_embeddings / count
            }
        """
        
        cursor = self.db.aql.execute(query)
        data = list(cursor)
        
        return data
    
    def extract_similarity_distribution(self):
        """Extract distribution of similarity scores."""
        query = """
        FOR edge IN semantic_similarity
            COLLECT
                bucket = FLOOR(edge.similarity * 20) / 20
            AGGREGATE
                count = COUNT(1),
                avg_context = AVG(edge.context),
                min_context = MIN(edge.context),
                max_context = MAX(edge.context)
            
            RETURN {
                similarity_bucket: bucket,
                edge_count: count,
                avg_context: avg_context,
                context_range: [min_context, max_context]
            }
        """
        
        cursor = self.db.aql.execute(query)
        data = list(cursor)
        
        return data
    
    def generate_wolfram_export(self, output_dir):
        """Generate clean data export for Wolfram validation."""
        print("Extracting data from ArangoDB...")
        
        # Extract all data types
        zero_prop_data = self.extract_for_zero_propagation_test()
        context_data = self.extract_context_amplification_data()
        dimensional_data = self.extract_dimensional_data()
        category_stats = self.extract_category_statistics()
        similarity_dist = self.extract_similarity_distribution()
        
        # Prepare Wolfram-friendly export
        wolfram_data = {
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'database': self.db_name,
                'paper_count': len(dimensional_data),
                'edge_count': len(context_data)
            },
            
            # For zero propagation test
            'zero_propagation': {
                'test_cases': zero_prop_data[:100],  # First 100 for testing
                'summary': {
                    'total_papers': len(zero_prop_data),
                    'with_zero_dimension': sum(1 for d in zero_prop_data 
                                             if any(d[k] == 0 for k in ['WHERE', 'WHAT', 'CONVEYANCE', 'TIME'])),
                    'zero_information': sum(1 for d in zero_prop_data if d['INFORMATION'] == 0)
                }
            },
            
            # For context amplification measurement
            'context_amplification': {
                'data_points': [[d['original_context'], d['amplified_context']] 
                               for d in context_data if d['original_context'] > 0],
                'sample_size': len(context_data)
            },
            
            # For dimensional analysis
            'dimensional_scores': {
                'papers': dimensional_data[:500],  # Sample for analysis
                'score_distributions': {
                    'WHERE': [d['WHERE_score'] for d in dimensional_data],
                    'WHAT': [d['WHAT_score'] for d in dimensional_data],
                    'CONVEYANCE': [d['CONVEYANCE_score'] for d in dimensional_data],
                    'INFORMATION': [d['INFORMATION_score'] for d in dimensional_data]
                }
            },
            
            # Category analysis
            'categories': category_stats,
            
            # Similarity distribution
            'similarity_distribution': similarity_dist
        }
        
        # Save full export
        output_path = os.path.join(output_dir, 'wolfram_data_export.json')
        with open(output_path, 'w') as f:
            json.dump(wolfram_data, f, indent=2)
        
        # Also save simplified CSV files for easy Wolfram import
        # Context amplification data
        context_df = pd.DataFrame(context_data)
        if not context_df.empty:
            context_df[['original_context', 'amplified_context', 'similarity']].to_csv(
                os.path.join(output_dir, 'context_amplification.csv'), 
                index=False
            )
        
        # Dimensional scores
        dim_df = pd.DataFrame(dimensional_data)
        if not dim_df.empty:
            dim_df[['paper_id', 'WHERE_score', 'WHAT_score', 'CONVEYANCE_score', 'TIME_score', 'INFORMATION_score']].to_csv(
                os.path.join(output_dir, 'dimensional_scores.csv'), 
                index=False
            )
        
        # Zero propagation test cases
        zero_df = pd.DataFrame(zero_prop_data[:1000])
        if not zero_df.empty:
            zero_df[['id', 'WHERE', 'WHAT', 'CONVEYANCE', 'TIME', 'INFORMATION']].to_csv(
                os.path.join(output_dir, 'zero_propagation_test.csv'), 
                index=False
            )
        
        print(f"\nExport complete! Files saved to {output_dir}")
        print(f"  - wolfram_data_export.json (full export)")
        print(f"  - context_amplification.csv ({len(context_data)} edges)")
        print(f"  - dimensional_scores.csv ({len(dimensional_data)} papers)")
        print(f"  - zero_propagation_test.csv ({min(1000, len(zero_prop_data))} test cases)")
        
        # Print summary statistics
        print("\nSUMMARY STATISTICS:")
        print(f"  Papers with embeddings: {sum(1 for d in dimensional_data if d['has_embeddings'])}")
        print(f"  Total edges: {len(context_data)}")
        print(f"  Categories: {len(category_stats)}")
        
        if context_data:
            original = [d['original_context'] for d in context_data]
            amplified = [d['amplified_context'] for d in context_data]
            print(f"\nContext scores:")
            print(f"  Original: mean={np.mean(original):.3f}, std={np.std(original):.3f}")
            print(f"  Amplified: mean={np.mean(amplified):.3f}, std={np.std(amplified):.3f}")
        
        return wolfram_data

def main():
    """Extract data for Wolfram validation."""
    output_dir = "/home/todd/reconstructionism/validation/experiment_1/wolfram/data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("WOLFRAM DATA EXTRACTION FROM ARANGO DATABASE")
    print("="*60)
    
    extractor = WolframDataExtractor()
    wolfram_data = extractor.generate_wolfram_export(output_dir)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("Ready for Wolfram mathematical validation")
    print("="*60)

if __name__ == "__main__":
    main()