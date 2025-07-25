"""
Data Preparation for Experiment 1

Uses the validated infrastructure to load and prepare 2000 arXiv documents
for multiplicative model validation.
"""

import argparse
from pathlib import Path
import json
import logging
from typing import List, Dict

# Use our validated infrastructure
from irec_infrastructure import DocumentProcessor
from irec_infrastructure.data import ArxivLoader
from irec_infrastructure.embeddings import JinaConfig
from irec_infrastructure.monitoring import ProgressTracker


def prepare_experiment_data(
    num_papers: int = 2000,
    source_dir: str = "/mnt/data/arxiv_data/",
    output_dir: str = "./results"
) -> Dict:
    """
    Prepare data for experiment 1 using validated infrastructure.
    
    This function:
    1. Loads papers from arXiv dataset
    2. Ensures diverse sampling across domains
    3. Extracts metadata needed for dimensional measurements
    4. Does NOT generate embeddings yet (that's in dimensional_measurement.py)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing {num_papers} papers for experiment 1")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize infrastructure components
    loader = ArxivLoader(base_path=source_dir)
    progress = ProgressTracker(total=num_papers, desc="Loading papers")
    
    # Load papers with diversity sampling
    papers = loader.load_papers(
        num_papers=num_papers,
        categories=["cs.AI", "cs.LG", "physics.hep-th", "math.ST"],
        sampling_strategy="diverse",
        time_range=("2010-01-01", "2023-12-31")
    )
    
    # Extract metadata for dimensional measurements
    prepared_data = []
    for paper in papers:
        progress.update(1)
        
        # Extract information needed for WHERE dimension
        where_info = {
            "is_open_access": paper.get("is_open_access", True),  # arXiv is open
            "has_pdf": Path(paper["pdf_path"]).exists(),
            "language": "en",  # Assume English for now
        }
        
        # Basic metadata for other dimensions
        prepared_data.append({
            "paper_id": paper["id"],
            "title": paper["title"],
            "abstract": paper["abstract"],
            "categories": paper["categories"],
            "authors": paper["authors"],
            "date": paper["date"],
            "pdf_path": paper["pdf_path"],
            "where_info": where_info,
            # WHAT, CONVEYANCE, TIME will be computed in dimensional_measurement.py
        })
    
    progress.close()
    
    # Save prepared data
    output_file = output_path / "prepared_papers.json"
    with open(output_file, "w") as f:
        json.dump({
            "num_papers": len(prepared_data),
            "papers": prepared_data,
            "sampling_info": {
                "categories": ["cs.AI", "cs.LG", "physics.hep-th", "math.ST"],
                "time_range": ["2010-01-01", "2023-12-31"],
                "strategy": "diverse"
            }
        }, f, indent=2)
    
    logger.info(f"Saved {len(prepared_data)} papers to {output_file}")
    
    # Generate summary statistics
    stats = {
        "total_papers": len(prepared_data),
        "by_category": {},
        "by_year": {},
        "missing_pdfs": sum(1 for p in prepared_data if not p["where_info"]["has_pdf"])
    }
    
    # Count by primary category
    for paper in prepared_data:
        primary_cat = paper["categories"][0] if paper["categories"] else "unknown"
        stats["by_category"][primary_cat] = stats["by_category"].get(primary_cat, 0) + 1
        
        year = paper["date"][:4] if paper["date"] else "unknown"
        stats["by_year"][year] = stats["by_year"].get(year, 0) + 1
    
    # Save statistics
    stats_file = output_path / "data_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    return {
        "papers": prepared_data,
        "statistics": stats,
        "output_dir": str(output_path)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for experiment 1")
    parser.add_argument("--papers", type=int, default=2000, help="Number of papers to load")
    parser.add_argument("--source", default="/mnt/data/arxiv_data/", help="Source directory")
    parser.add_argument("--output", default="./results", help="Output directory")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run data preparation
    results = prepare_experiment_data(
        num_papers=args.papers,
        source_dir=args.source,
        output_dir=args.output
    )
    
    print(f"\nData preparation complete!")
    print(f"Prepared {results['statistics']['total_papers']} papers")
    print(f"Results saved to: {results['output_dir']}")