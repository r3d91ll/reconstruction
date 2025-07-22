#!/usr/bin/env python3
"""
Compute Physical Grounding Factor for papers
Based on textual markers that predict implementation success
"""

import json
import re
import os
import glob
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd


class PhysicalGroundingCalculator:
    """Calculate Physical Grounding Factor based on paper content markers"""
    
    def __init__(self):
        # Markers weighted by implementation likelihood
        self.grounding_markers = {
            'high': {
                'algorithm': 0.9,
                'algorithm \\d+': 0.95,  # Algorithm 1, Algorithm 2, etc.
                'step \\d+': 0.9,        # Step 1, Step 2, etc.
                'procedure': 0.85,
                'pseudocode': 0.95,
                'implementation': 0.8,
                'github.com': 0.95,
                'source code': 0.9,
                'equation \\(\\d+\\)': 0.7,  # Equation (1), etc.
                'loss function': 0.8,
                'objective function': 0.8
            },
            'medium': {
                'framework': 0.6,
                'architecture': 0.6,
                'method': 0.5,
                'approach': 0.5,
                'technique': 0.5,
                'model': 0.6,
                'evaluation': 0.6,
                'experiment': 0.6,
                'dataset': 0.5,
                'baseline': 0.6
            },
            'low': {
                'theory': 0.3,
                'theoretical': 0.2,
                'conjecture': 0.1,
                'hypothesis': 0.3,
                'suggests': 0.2,
                'implies': 0.2,
                'philosophical': 0.1,
                'conceptual': 0.2,
                'abstract': 0.2,
                'potentially': 0.2
            }
        }
        
        # Domain-specific adjustments
        self.domain_multipliers = {
            'cs.': 1.2,      # Computer science: higher grounding
            'math.': 0.8,    # Mathematics: moderate grounding
            'physics.': 0.9, # Physics: good grounding
            'q-bio.': 1.1,   # Quantitative biology: high grounding
            'stat.': 1.0,    # Statistics: neutral
            'econ.': 0.7,    # Economics: lower grounding
            'phil.': 0.3     # Philosophy: very low grounding
        }
        
    def extract_text_features(self, paper: Dict) -> Dict[str, float]:
        """Extract grounding-relevant features from paper text"""
        features = defaultdict(float)
        
        # Combine title and abstract for analysis
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        text_lower = text.lower()
        
        # Check for implementation indicators
        for level, markers in self.grounding_markers.items():
            for marker, weight in markers.items():
                # Use regex for pattern matching
                if '\\' in marker:  # It's a regex pattern
                    matches = len(re.findall(marker, text_lower))
                else:  # Simple string search
                    matches = text_lower.count(marker)
                
                if matches > 0:
                    features[f'{level}_{marker}'] = min(matches * weight, 1.0)
        
        # Check for specific implementation signals
        if 'github.com' in text_lower or 'code available' in text_lower:
            features['has_code'] = 1.0
        
        if re.search(r'algorithm\s+\d+', text_lower):
            features['numbered_algorithm'] = 1.0
            
        if re.search(r'step\s+\d+', text_lower):
            features['numbered_steps'] = 1.0
            
        # Mathematical content
        math_symbols = text.count('$') / 2  # LaTeX math mode
        features['math_density'] = min(math_symbols / 100, 1.0)
        
        # Check comment field for implementation notes
        comment = paper.get('comment', '').lower()
        if 'code' in comment or 'implementation' in comment:
            features['comment_has_code'] = 1.0
            
        return features
    
    def calculate_grounding_score(self, features: Dict[str, float], 
                                 primary_category: str) -> float:
        """Calculate final Physical Grounding Factor"""
        
        # Aggregate scores by level
        high_score = sum(v for k, v in features.items() if k.startswith('high_'))
        medium_score = sum(v for k, v in features.items() if k.startswith('medium_'))
        low_score = sum(v for k, v in features.items() if k.startswith('low_'))
        
        # Special bonuses
        code_bonus = features.get('has_code', 0) * 0.3
        algo_bonus = features.get('numbered_algorithm', 0) * 0.2
        step_bonus = features.get('numbered_steps', 0) * 0.1
        math_bonus = features.get('math_density', 0) * 0.1
        
        # Weighted combination
        base_score = (
            high_score * 0.6 +      # High-grounding markers most important
            medium_score * 0.3 +    # Medium markers supportive
            (1 - low_score) * 0.1   # Penalize low-grounding markers
        )
        
        # Add bonuses
        total_score = base_score + code_bonus + algo_bonus + step_bonus + math_bonus
        
        # Apply domain multiplier
        domain_mult = 1.0
        for domain_prefix, mult in self.domain_multipliers.items():
            if primary_category.startswith(domain_prefix):
                domain_mult = mult
                break
                
        # Normalize to [0, 1]
        final_score = min(total_score * domain_mult, 1.0)
        
        return final_score
    
    def compute_for_paper(self, paper: Dict) -> Tuple[float, Dict]:
        """Compute Physical Grounding Factor for a single paper"""
        features = self.extract_text_features(paper)
        score = self.calculate_grounding_score(features, 
                                             paper.get('primary_category', ''))
        
        return score, features
    
    def compute_for_dataset(self, papers: List[Dict]) -> pd.DataFrame:
        """Compute Physical Grounding Factor for multiple papers"""
        results = []
        
        for paper in papers:
            score, features = self.compute_for_paper(paper)
            
            result = {
                'arxiv_id': paper.get('arxiv_id'),
                'title': paper.get('title', '')[:80] + '...',
                'year': paper.get('year'),
                'primary_category': paper.get('primary_category'),
                'physical_grounding_factor': score,
                'has_code_indicator': features.get('has_code', 0),
                'has_algorithm': features.get('numbered_algorithm', 0),
                'has_steps': features.get('numbered_steps', 0),
                'math_density': features.get('math_density', 0)
            }
            
            results.append(result)
            
        return pd.DataFrame(results)


def analyze_grounding_distribution(df: pd.DataFrame):
    """Analyze the distribution of Physical Grounding Factors"""
    print("\n=== PHYSICAL GROUNDING ANALYSIS ===")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Mean grounding: {df['physical_grounding_factor'].mean():.3f}")
    print(f"  Std deviation: {df['physical_grounding_factor'].std():.3f}")
    print(f"  Min: {df['physical_grounding_factor'].min():.3f}")
    print(f"  Max: {df['physical_grounding_factor'].max():.3f}")
    
    # Papers with code indicators
    with_code = df[df['has_code_indicator'] > 0]
    print(f"\nPapers with code indicators: {len(with_code)} ({100*len(with_code)/len(df):.1f}%)")
    if len(with_code) > 0:
        print(f"  Average grounding: {with_code['physical_grounding_factor'].mean():.3f}")
    
    # By category
    print("\nGrounding by Category (top 10):")
    category_stats = df.groupby('primary_category').agg({
        'physical_grounding_factor': ['mean', 'count']
    }).sort_values(('physical_grounding_factor', 'mean'), ascending=False)
    
    for cat, row in category_stats.head(10).iterrows():
        mean_score = row[('physical_grounding_factor', 'mean')]
        count = row[('physical_grounding_factor', 'count')]
        print(f"  {cat}: {mean_score:.3f} (n={count})")
    
    # Examples of high and low grounding papers
    print("\nHighest Grounding Papers:")
    for _, paper in df.nlargest(5, 'physical_grounding_factor').iterrows():
        print(f"  {paper['physical_grounding_factor']:.3f}: {paper['title']}")
    
    print("\nLowest Grounding Papers:")
    for _, paper in df.nsmallest(5, 'physical_grounding_factor').iterrows():
        print(f"  {paper['physical_grounding_factor']:.3f}: {paper['title']}")


def main():
    """Compute Physical Grounding Factor for papers"""
    
    # Load sample of papers
    papers_dir = "/home/todd/olympus/Erebus/unstructured/papers"
    calculator = PhysicalGroundingCalculator()
    
    # Load papers
    papers = []
    json_files = glob.glob(os.path.join(papers_dir, "*.json"))[:1000]  # First 1000
    
    print(f"Loading {len(json_files)} papers...")
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                papers.append(json.load(f))
        except:
            continue
    
    print(f"Computing Physical Grounding Factor for {len(papers)} papers...")
    
    # Compute grounding scores
    results_df = calculator.compute_for_dataset(papers)
    
    # Save results
    output_path = "/home/todd/reconstructionism/validation/data/physical_grounding_scores.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Analyze distribution
    analyze_grounding_distribution(results_df)
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribution histogram
    ax1.hist(results_df['physical_grounding_factor'], bins=30, alpha=0.7, color='blue')
    ax1.axvline(results_df['physical_grounding_factor'].mean(), color='red', 
                linestyle='--', label=f'Mean: {results_df["physical_grounding_factor"].mean():.3f}')
    ax1.set_xlabel('Physical Grounding Factor')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Physical Grounding Scores')
    ax1.legend()
    
    # Category comparison
    category_means = results_df.groupby('primary_category')['physical_grounding_factor'].mean()
    category_counts = results_df.groupby('primary_category').size()
    
    # Filter categories with enough samples
    valid_categories = category_counts[category_counts >= 10].index
    category_means_filtered = category_means[valid_categories].sort_values(ascending=False)[:15]
    
    ax2.bar(range(len(category_means_filtered)), category_means_filtered.values, alpha=0.7)
    ax2.set_xticks(range(len(category_means_filtered)))
    ax2.set_xticklabels(category_means_filtered.index, rotation=45, ha='right')
    ax2.set_ylabel('Mean Physical Grounding Factor')
    ax2.set_title('Grounding by Research Category')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/todd/reconstructionism/validation/python/physical_grounding_distribution.png', 
                dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: physical_grounding_distribution.png")


if __name__ == "__main__":
    main()