#!/usr/bin/env python3
"""Inspect the arxiv_abstracts_61463 database to understand available metadata."""

import os
import statistics
from arango import ArangoClient

# Connect to database
client = ArangoClient(hosts='http://192.168.1.69:8529')
db = client.db('arxiv_abstracts_61463', username='root', password=os.environ.get('ARANGO_PASSWORD', ''))

# Get collection info
collection = db.collection('abstract_metadata')
print(f'Total documents: {collection.count()}')

# Sample a document to see structure
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
    LIMIT 1
    RETURN doc
''')

sample = list(cursor)[0]
print('\nSample document structure:')
for key in sample.keys():
    if key == 'abstract_embedding':
        print(f'  {key}: [vector of length {len(sample[key])}]')
    elif isinstance(sample[key], list) and key != 'abstract_embedding':
        print(f'  {key}: {sample[key][:3]}... (length: {len(sample[key])})')
    else:
        print(f'  {key}: {sample[key]}')

# Check metadata variety
print('\n\nMetadata statistics:')

# Category distribution
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        FOR cat IN doc.categories
        COLLECT category = cat WITH COUNT INTO count
        SORT count DESC
        LIMIT 15
        RETURN {category: category, count: count}
''')
print('\nTop 15 categories:')
for item in cursor:
    print(f'  {item["category"]}: {item["count"]}')

# Temporal distribution
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        FILTER doc.published != null
        COLLECT year = SUBSTRING(doc.published, 0, 4) WITH COUNT INTO count
        SORT year DESC
        LIMIT 10
        RETURN {year: year, count: count}
''')
print('\nPublication years (last 10):')
for item in cursor:
    print(f'  {item["year"]}: {item["count"]}')

# Author collaboration potential
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        RETURN LENGTH(doc.authors)
''')
author_counts = list(cursor)
print(f'\nAuthor statistics:')
print(f'  Average authors per paper: {statistics.mean(author_counts):.2f}')
print(f'  Max authors: {max(author_counts)}')
print(f'  Papers with 2+ authors: {sum(1 for c in author_counts if c >= 2)}')

# Check for other metadata fields
print('\n\nOther metadata availability:')
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        LIMIT 1000
        RETURN {
            has_doi: doc.doi != null,
            has_journal: doc.journal_ref != null,
            has_pdf_url: doc.pdf_url != null,
            has_abs_url: doc.abs_url != null
        }
''')

results = list(cursor)
print(f'  Papers with DOI: {sum(1 for r in results if r["has_doi"])}/{len(results)}')
print(f'  Papers with journal ref: {sum(1 for r in results if r["has_journal"])}/{len(results)}')
print(f'  Papers with PDF URL: {sum(1 for r in results if r["has_pdf_url"])}/{len(results)}')
print(f'  Papers with abstract URL: {sum(1 for r in results if r["has_abs_url"])}/{len(results)}')

# Check embedding quality
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        LIMIT 100
        RETURN LENGTH(doc.abstract_embedding)
''')
embedding_lengths = list(cursor)
print(f'\nEmbedding check:')
print(f'  All embeddings have length: {set(embedding_lengths)}')

# Check title/abstract lengths for quality
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        RETURN {
            title_length: LENGTH(doc.title),
            abstract_length: LENGTH(doc.abstract)
        }
''')
text_stats = list(cursor)
title_lengths = [s['title_length'] for s in text_stats]
abstract_lengths = [s['abstract_length'] for s in text_stats]

print(f'\nText quality:')
print(f'  Average title length: {statistics.mean(title_lengths):.0f} chars')
print(f'  Average abstract length: {statistics.mean(abstract_lengths):.0f} chars')
print(f'  Min abstract length: {min(abstract_lengths)} chars')