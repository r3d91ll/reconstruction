# arXiv Native Schema Analysis for Base Container Design

## Overview
Following the principle of letting each source database determine its initial schema, we'll analyze the native arXiv metadata structure and design our base containers to preserve all original fields exactly as they come from the source.

## Native arXiv Metadata Structure

Based on analysis of the actual JSON files, arXiv metadata contains these primary fields:

### Core Identification Fields
```json
{
  "id": "http://arxiv.org/abs/cs/0005007v1",
  "arxiv_id": "0005007v1",
  "entry_id": "http://arxiv.org/abs/cs/0005007v1",
  "updated": "2000-05-08T01:49:57+00:00",
  "published": "2000-05-08T01:49:57+00:00"
}
```

### Content Fields
```json
{
  "title": "Scientific Collaboratories as Socio-Technical Interaction Networks: A Theoretical Approach",
  "abstract": "Collaboratories refer to laboratories where scientists can work together...",
  "authors": [
    "Rob Kling",
    "Geoffrey McKim",
    "Joanna Fortuna",
    "Adam King"
  ],
  "categories": ["cs.CY", "H.5.3"],
  "primary_category": "cs.CY"
}
```

### Access Information
```json
{
  "pdf_url": "http://arxiv.org/pdf/cs/0005007v1",
  "doi": "10.1145/...",  // Optional
  "journal_ref": "ACM Computing Surveys...",  // Optional
  "comments": "15 pages, 2 figures"  // Optional
}
```

## Proposed Base Container Schemas

### 1. arxiv_metadata (Preserves Original Structure)

```javascript
// Collection: arxiv_metadata
{
  "_key": "0005007v1",  // Direct arXiv ID
  
  // === Original arXiv Fields (Preserved Exactly) ===
  "id": "http://arxiv.org/abs/cs/0005007v1",
  "arxiv_id": "0005007v1",
  "entry_id": "http://arxiv.org/abs/cs/0005007v1",
  "updated": "2000-05-08T01:49:57+00:00",
  "published": "2000-05-08T01:49:57+00:00",
  "title": "Scientific Collaboratories as Socio-Technical Interaction Networks: A Theoretical Approach",
  "abstract": "Collaboratories refer to laboratories where scientists can work together...",
  "authors": [
    "Rob Kling",
    "Geoffrey McKim"
  ],
  "categories": ["cs.CY", "H.5.3"],
  "primary_category": "cs.CY",
  "pdf_url": "http://arxiv.org/pdf/cs/0005007v1",
  "doi": null,
  "journal_ref": null,
  "comments": null,
  
  // === Processing Metadata (Added by Pipeline) ===
  "_meta": {
    "source": "arxiv",
    "ingested_at": "2024-07-30T08:45:00Z",
    "pipeline_version": "2.0",
    "source_file": "/path/to/original/0005007v1.json"
  }
}
```

### 2. arxiv_abstracts (Optimized for Text Processing)

```javascript
// Collection: arxiv_abstracts
{
  "_key": "0005007v1",
  "arxiv_id": "0005007v1",
  "abstract": "Collaboratories refer to laboratories where scientists can work together...",
  "title": "Scientific Collaboratories as Socio-Technical Interaction Networks: A Theoretical Approach",
  "language": "en",  // Detected
  "word_count": 250,
  "char_count": 1523,
  "_meta": {
    "processed_at": "2024-07-30T08:45:01Z"
  }
}
```

### 3. arxiv_abstract_embeddings (Jina Embeddings)

```javascript
// Collection: arxiv_abstract_embeddings  
{
  "_key": "0005007v1",
  "arxiv_id": "0005007v1",
  "embedding": [...],  // 1024-dim Jina v3 embedding
  "model": "jinaai/jina-embeddings-v3",
  "model_version": "3.0",
  "embedding_dim": 1024,
  "_meta": {
    "created_at": "2024-07-30T08:45:02Z",
    "gpu_id": 1,
    "batch_id": "w1_b42"
  }
}
```

## Schema Evolution Strategy

### Phase 1: Direct Import
1. Load raw arXiv JSON files
2. Preserve ALL original fields in arxiv_metadata
3. Extract minimal fields for specialized containers
4. Add only essential processing metadata

### Phase 2: Enhancement (Future)
```javascript
// Additional fields that might come from enhanced sources:
{
  "affiliations": [...],  // From author disambiguation services
  "citations": [...],     // From citation databases
  "semantic_scholar_id": "...",  // Cross-reference IDs
  "references": [...],    // Extracted from PDFs
  "funding": [...],       // From acknowledgments
  "datasets": [...]       // Linked datasets
}
```

### Phase 3: Analytical Layer
Only in the analytical layer do we normalize across sources:
```javascript
// Collection: unified_papers_metadata (Analytical Layer)
{
  "_key": "arxiv_0005007v1",
  "source": "arxiv",
  "source_id": "0005007v1",
  "title": "...",  // Normalized
  "authors": [...],  // Normalized structure
  "publication_date": "2000-05-08",  // ISO format
  "topics": ["collaboration", "distributed_systems"],  // Normalized
  "citations_count": 45,  // From citation services
  "cited_by": ["arxiv_0105023v2", "pubmed_12345678"]
}
```

## Implementation Guidelines

### 1. Schema Discovery Process
```python
def discover_arxiv_schema(sample_files: List[Path]) -> Dict:
    """Analyze sample files to discover all fields"""
    all_fields = set()
    field_types = {}
    field_examples = {}
    
    for file_path in sample_files[:1000]:  # Sample 1000 files
        with open(file_path) as f:
            data = json.load(f)
            
        # Recursively discover all fields
        def extract_fields(obj, prefix="", visited=None, depth=0):
            if visited is None:
                visited = set()
                
            # Limit recursion depth
            if depth > 10:
                return
                
            # Check for circular references
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    field_path = f"{prefix}.{key}" if prefix else key
                    all_fields.add(field_path)
                    field_types[field_path] = type(value).__name__
                    if field_path not in field_examples:
                        field_examples[field_path] = value
                    if isinstance(value, dict):
                        extract_fields(value, field_path, visited, depth + 1)
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                        # Handle lists of dicts
                        extract_fields(value[0], f"{field_path}[0]", visited, depth + 1)
                        
        extract_fields(data)
    
    return {
        "fields": sorted(all_fields),
        "types": field_types,
        "examples": field_examples
    }
```

### 2. Validation Rules
```python
ARXIV_REQUIRED_FIELDS = {
    "arxiv_id",  # or "id" 
    "title",
    "abstract",
    "authors"
}

ARXIV_OPTIONAL_FIELDS = {
    "categories",
    "primary_category", 
    "published",
    "updated",
    "doi",
    "journal_ref",
    "comments",
    "pdf_url"
}
```

### 3. Index Strategy
```javascript
// Primary indexes for arxiv_metadata
db._collection('arxiv_metadata').ensureIndex({
  type: 'persistent',
  fields: ['arxiv_id'],
  unique: true
});

db._collection('arxiv_metadata').ensureIndex({
  type: 'persistent',
  fields: ['published'],
  sparse: true
});

db._collection('arxiv_metadata').ensureIndex({
  type: 'persistent',
  fields: ['categories[*]']
});

// Full-text search on abstracts
db._collection('arxiv_abstracts').ensureIndex({
  type: 'fulltext',
  fields: ['abstract'],
  minLength: 3
});
```

## Benefits of Native Schema Preservation

### 1. Data Integrity
- No information loss during import
- Can always trace back to original source
- Supports data lineage requirements

### 2. Flexibility
- Can adapt to schema changes in source
- Easy to add new fields as arXiv evolves
- No need to predict future requirements

### 3. Multi-Source Compatibility
- Each source maintains its native structure
- Normalization happens only in analytical layer
- Easy to add new sources (PubMed, JSTOR, etc.)

### 4. Performance
- Specialized containers for specific operations
- Minimal data duplication
- Efficient indexing strategies

## Migration Path from V1

```python
# Migration script pseudocode
def migrate_v1_to_v2():
    # 1. Export current data
    existing_data = export_v1_collection()
    
    # 2. Transform to native schema
    for doc in existing_data:
        # Preserve original fields
        native_doc = {
            "_key": doc.get("arxiv_id"),
            **doc.get("raw_metadata", {}),  # If available
            "_meta": {
                "source": "arxiv",
                "migrated_from": "v1",
                "migration_date": datetime.now()
            }
        }
        
        # Create specialized docs
        abstract_doc = {
            "_key": doc.get("arxiv_id"),
            "arxiv_id": doc.get("arxiv_id"),
            "abstract": doc.get("abstract"),
            "title": doc.get("title")
        }
        
        embedding_doc = {
            "_key": doc.get("arxiv_id"),
            "arxiv_id": doc.get("arxiv_id"),
            "embedding": doc.get("jina_embedding_3")
        }
        
        # Insert into new collections
        insert_into_v2(native_doc, abstract_doc, embedding_doc)
```

## Recommended Next Steps

1. **Schema Discovery**
   - Run discovery script on 10,000 sample files
   - Document all discovered fields
   - Identify required vs optional fields

2. **Create Schema Documentation**
   - Generate JSON Schema for validation
   - Document field meanings and sources
   - Create data dictionary

3. **Build Import Pipeline**
   - Direct JSON import with validation
   - Parallel processing for performance
   - Progress tracking and resumability

4. **Test with Sample Data**
   - Import 1000 documents
   - Verify all fields preserved
   - Test query performance

This approach ensures that your base containers remain true to the source data while providing the flexibility to evolve and integrate with other sources in the future.