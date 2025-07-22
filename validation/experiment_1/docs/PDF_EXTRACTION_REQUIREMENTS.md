# PDF Extraction Technical Requirements

## Overview

Convert academic PDFs to structured Markdown with extracted images, adding the content to existing JSON metadata files for full-document embeddings.

## Input/Output Specification

### Input

- **PDF files**: Academic papers (typically 8-30 pages)
- **JSON files**: Existing metadata with title, abstract, categories, etc.
- **Location**: `/home/todd/olympus/Erebus/unstructured/papers/`

### Output

Enhanced JSON files with additional fields:

```json
{
  // Existing fields
  "id": "2301.12345",
  "title": "Paper Title",
  "abstract": "...",
  "categories": ["cs.AI", "cs.LG"],
  
  // New fields to add
  "pdf_content": {
    "markdown": "# Paper Title\n\n## Abstract\n...",
    "sections": [
      {
        "title": "Introduction",
        "content": "markdown content...",
        "subsections": []
      }
    ],
    "images": [
      {
        "figure_id": "fig1",
        "caption": "Figure 1: Architecture diagram",
        "path": "images/2301.12345_fig1.png",
        "context": "Referenced in Section 3.2",
        "type": "diagram"
      }
    ],
    "tables": [
      {
        "table_id": "tab1",
        "caption": "Table 1: Experimental Results",
        "markdown": "| Model | Accuracy | F1 Score |..."
      }
    ],
    "equations": [
      {
        "eq_id": "eq1",
        "latex": "E = mc^2",
        "context": "Einstein's mass-energy equivalence"
      }
    ],
    "code_blocks": [
      {
        "language": "python",
        "content": "def attention(Q, K, V):\n    ...",
        "section": "Implementation Details"
      }
    ],
    "references": [
      {
        "id": "ref1",
        "text": "Vaswani et al. (2017). Attention is all you need.",
        "type": "paper"
      }
    ],
    "metadata": {
      "num_pages": 12,
      "num_figures": 5,
      "num_tables": 3,
      "num_equations": 15,
      "num_code_blocks": 8,
      "extraction_timestamp": "2024-07-22T20:00:00Z",
      "extraction_method": "marker-pdf"
    }
  }
}
```

## Extraction Requirements

### 1. Text Extraction

- **Preserve structure**: Maintain section hierarchy (h1, h2, h3, etc.)
- **Clean formatting**: Remove headers, footers, page numbers
- **Handle multi-column**: Correctly order text from multi-column layouts
- **Unicode support**: Preserve mathematical symbols, special characters

### 2. Image Extraction

- **Format**: Save as PNG (lossless)
- **Naming**: `{paper_id}_fig{N}.png` (e.g., `2301.12345_fig1.png`)
- **Storage**: Create `images/` subdirectory in papers folder
- **Resolution**: Maintain original quality (min 150 DPI)
- **Types to extract**:
  - Figures and diagrams
  - Architecture diagrams
  - Plots and charts
  - Algorithm flowcharts
  - Screenshots

### 3. Table Extraction

- **Format**: Convert to Markdown tables
- **Complex tables**: For nested/complex tables, also save as image
- **Captions**: Preserve table captions and numbers

### 4. Equation Extraction

- **Format**: Preserve LaTeX notation
- **Display equations**: Mark as separate blocks
- **Inline equations**: Keep within text flow
- **Fallback**: For complex equations, extract as image

### 5. Code Block Extraction

- **Language detection**: Identify programming language
- **Syntax preservation**: Maintain indentation and formatting
- **Context**: Note which section contains the code

### 6. Reference Extraction

- **Format**: Structured list with identifiers
- **Linking**: Preserve in-text citation links
- **Types**: Differentiate papers, books, URLs, etc.

## Processing Pipeline

```python
# Pseudo-code for processing pipeline
def process_paper(pdf_path, json_path):
    # 1. Load existing JSON
    metadata = load_json(json_path)
    
    # 2. Extract PDF content
    content = extract_pdf_to_markdown(pdf_path)
    images = extract_images(pdf_path)
    tables = extract_tables(pdf_path)
    equations = extract_equations(pdf_path)
    code_blocks = extract_code(pdf_path)
    references = extract_references(pdf_path)
    
    # 3. Structure content
    pdf_content = {
        "markdown": content["full_text"],
        "sections": content["sections"],
        "images": images,
        "tables": tables,
        "equations": equations,
        "code_blocks": code_blocks,
        "references": references,
        "metadata": {
            "num_pages": content["num_pages"],
            "num_figures": len(images),
            "num_tables": len(tables),
            "num_equations": len(equations),
            "num_code_blocks": len(code_blocks),
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_method": "marker-pdf"
        }
    }
    
    # 4. Add to existing metadata
    metadata["pdf_content"] = pdf_content
    
    # 5. Save enhanced JSON
    save_json(json_path, metadata)
    
    # 6. Save images
    for img in images:
        save_image(img["data"], img["path"])
```

## Quality Requirements

### Text Quality

- **Accuracy**: >95% character accuracy
- **Structure**: Preserve document hierarchy
- **Readability**: Clean, well-formatted Markdown

### Image Quality

- **Completeness**: Extract ALL figures/diagrams
- **Resolution**: No quality loss from original
- **Captions**: Associate correct captions with images

### Performance

- **Speed**: ~30-60 seconds per paper (10-20 pages)
- **Memory**: Handle papers up to 200 pages
- **Robustness**: Gracefully handle corrupted/protected PDFs

## Integration with Embedding Pipeline

After extraction, the embedding pipeline will:

1. Load the enhanced JSON with full content
2. Combine all text sections into a single document
3. Include image captions and references
4. Feed to Jina V4 (128K token context)
5. Generate comprehensive document embeddings

## Recommended Tools

1. **marker-pdf**: Fast, accurate PDF to Markdown conversion
   - Handles academic papers well
   - Preserves equations and tables
   - Extracts images

2. **pypdf**: For PDF manipulation and image extraction

3. **Pillow**: For image processing and format conversion

## Error Handling

- **Missing PDFs**: Log and continue
- **Corrupted PDFs**: Try alternative extraction, log failures
- **Protected PDFs**: Skip with warning
- **Extraction failures**: Save partial results

## Storage Estimate

For 10,000 papers:

- **Images**: ~5 images/paper × 500KB = 25GB
- **Enhanced JSONs**: ~100KB/paper = 1GB
- **Total additional storage**: ~26GB

## Next Steps

1. Install required libraries (marker-pdf, pypdf, Pillow)
2. Create extraction script following these requirements
3. Test on sample papers (including the milestone papers)
4. Run on full dataset
5. Update embedding pipeline to use full content
