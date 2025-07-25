#!/bin/bash
# Cleanup script for experiment_1 to prepare for upstream push

echo "=== Cleaning experiment_1 for repository push ==="

# Create archive directory if it doesn't exist
mkdir -p archive

# Archive older pipeline versions
echo "Archiving older pipeline versions..."
mv pipeline/step1_generate_embeddings_jina.py archive/ 2>/dev/null
mv pipeline/step1_generate_embeddings_with_logging.py archive/ 2>/dev/null
mv pipeline/step1_run_4000.py archive/ 2>/dev/null
mv pipeline/step4_context_amplification.py archive/ 2>/dev/null
mv pipeline/run_pipeline.py archive/ 2>/dev/null
mv pipeline/run_pipeline_with_logging.py archive/ 2>/dev/null

# Archive analysis and visualization scripts
echo "Archiving analysis scripts..."
mv pipeline/analyze_temporal_graph.py archive/ 2>/dev/null
mv pipeline/temporal_visualization.py archive/ 2>/dev/null
mv pipeline/generate_graph_report.py archive/ 2>/dev/null
mv pipeline/verify_graph.py archive/ 2>/dev/null
mv pipeline/export_for_wolfram.py archive/ 2>/dev/null

# Delete one-time check and debug scripts
echo "Removing one-time check scripts..."
rm -f pipeline/check_actual_content.py
rm -f pipeline/check_milestone_papers.py
rm -f pipeline/check_papers_ready.py
rm -f pipeline/check_year_distribution.py
rm -f pipeline/cleanup_partial_extraction.py
rm -f pipeline/quick_edge_check.py
rm -f pipeline/compare_embedding_quality.py

# Delete debug and test scripts from main directory
echo "Removing debug scripts..."
rm -f debug_pipeline_output.py
rm -f check_docling_results.py
rm -f check_amplification_truth.py
rm -f test_pipeline_minimal.py

# Archive main directory analysis script
mv analyze_context_amplification.py archive/ 2>/dev/null

# Archive wolfram utility scripts
echo "Archiving wolfram utilities..."
mv wolfram/wolfram_api_validation.py archive/ 2>/dev/null
mv wolfram/wolfram_validation_queries.py archive/ 2>/dev/null
mv wolfram/analyze_arango_data.py archive/ 2>/dev/null

# Clean up __pycache__ directories
echo "Cleaning pycache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Clean up .pyc files
find . -name "*.pyc" -delete

# Create a clean structure report
echo -e "\n=== Cleaned Directory Structure ==="
tree -I '__pycache__|*.pyc' .

echo -e "\n=== Cleanup Complete ==="
echo "Scripts archived: $(ls archive/*.py 2>/dev/null | wc -l)"
echo "Main pipeline scripts remaining: $(ls pipeline/*.py 2>/dev/null | wc -l)"
echo "Ready for repository push!"