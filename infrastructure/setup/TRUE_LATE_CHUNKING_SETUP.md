# TRUE Late Chunking Infrastructure Setup

## Critical Difference from Traditional RAG

**Traditional RAG (WRONG for our use case):**
1. Chunk documents into pieces
2. Generate embeddings for each chunk
3. Store chunks + embeddings

**TRUE Late Chunking (Context-Safe):**
1. Extract FULL documents (no chunking!)
2. Send FULL documents to Jina API
3. Jina generates embeddings AND creates semantic chunks
4. Store Jina's chunks + embeddings

## Why This Matters

With TRUE late chunking:
- Jina sees the ENTIRE document before deciding chunk boundaries
- Chunks are semantically coherent because they're created with full context
- Embeddings are more accurate because they're generated with full document understanding
- This is what enables the "context safety" that makes Information Reconstructionism work

## Prerequisites

### Set JINA_API_KEY Environment Variable

You must set the JINA_API_KEY environment variable before proceeding. This key is required for accessing the Jina API.

**Set the key:**
```bash
export JINA_API_KEY="your-jina-api-key-here"
```

**Add to your shell profile (optional, for persistence):**
```bash
echo 'export JINA_API_KEY="your-jina-api-key-here"' >> ~/.bashrc
# or for zsh users:
echo 'export JINA_API_KEY="your-jina-api-key-here"' >> ~/.zshrc
```

**Validate the key is set:**
```bash
# This script will check if JINA_API_KEY is defined
if [ -z "$JINA_API_KEY" ]; then
    echo "ERROR: JINA_API_KEY environment variable is not set!"
    echo "Please set it using: export JINA_API_KEY='your-key-here'"
    exit 1
else
    echo "✓ JINA_API_KEY is set"
fi
```

## Phase 0: Infrastructure Setup with TRUE Late Chunking

### 0.1 Database Schema Creation (30 minutes)
```bash
python setup_database_schema.py \
    --db-name information_reconstructionism_base \
    --clean-start
```

**Creates:**
- Database: `information_reconstructionism_base`
- Collections designed for Jina's output format

### 0.2 Document Processing with TRUE Late Chunking (8-10 hours)
```bash
python process_documents.py \
    --input-dir /mnt/data/arxiv_data/pdf \
    --output-dir ./processed_documents \
    --num-docs 1960 \
    --jina-api-key $JINA_API_KEY
```

**Process:**
1. Docling extracts FULL text from each PDF
2. Send FULL text to Jina API with `late_chunking=True`
3. Jina returns:
   - Semantic chunks (it decides boundaries)
   - Embeddings for each chunk
   - Typically 50-100 chunks per document

**Output:**
- 1960 processed documents
- ~100,000-200,000 semantic chunks
- Each chunk has its embedding from Jina
- Full document text preserved

### 0.3 Load to Database (2 hours)
```bash
python load_processed_to_db.py \
    --input-dir ./processed_documents \
    --db-name information_reconstructionism_base
```

**Stores:**
- Full document text
- Jina-created chunks with their embeddings
- Document metadata
- Chunk-to-document relationships

### 0.4 Similarity Computation (3 hours)
```bash
python compute_similarities.py \
    --db-name information_reconstructionism_base \
    --chunk-threshold 0.7 \
    --document-threshold 0.6
```

**Computes:**
- Chunk-to-chunk similarities
- Document-level similarities (aggregated from chunks)
- Stores as sparse matrices

### 0.5 ISNE Model Training (4 hours)
```bash
python train_isne_model.py \
    --db-name information_reconstructionism_base \
    --eigenvectors 100 \
    --output-path ./models/isne_late_chunking.pkl
```

**Trains:**
- ISNE on chunk similarity graph
- Preserves semantic structure from Jina's chunking

## Total Time: ~20 hours

## Key Benefits of TRUE Late Chunking

1. **Context Safety**: Every chunk is created with full document understanding
2. **Semantic Coherence**: Chunks respect natural document boundaries
3. **Better Embeddings**: Generated with complete context
4. **No Information Loss**: Jina decides optimal chunk sizes

## Verification Steps

After setup, verify:
```bash
python verify_late_chunking.py --db-name information_reconstructionism_base
```

Should show:
- Average chunks per document: 50-100
- Chunk size distribution (varies based on semantic boundaries)
- No fixed-size chunks (proves Jina is deciding boundaries)

## Cost Estimate

Jina API pricing (July 2025):

**Free Tier:**
- 1M tokens/month free
- Good for testing and small experiments

**Starter Plan ($50/month):**
- 10,000 queries/month
- ~50M tokens/month
- Best for our use case

**Professional Plan ($500/month):**
- 100,000 queries/month
- ~500M tokens/month
- For production workloads

**Enterprise Plan (Custom pricing):**
- Unlimited queries
- SLA guarantees
- Custom support

**Our workload estimate:**
- 1960 documents × ~5000 tokens/doc average = ~10M tokens
- Approximately 2000 queries (one per document)
- **Cost: ~20% of Starter Plan capacity (~$10/month equivalent)**
- Worth it for TRUE semantic chunking

For latest pricing, see: https://jina.ai/pricing

## Next Steps

With this infrastructure:
1. All experiments can access pre-chunked, pre-embedded data
2. No need to re-process documents
3. Can test multiplicative model on semantically coherent chunks
4. Can discover α values empirically
5. Can analyze FRAME through citation patterns