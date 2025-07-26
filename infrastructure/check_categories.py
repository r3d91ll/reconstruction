#!/usr/bin/env python3
"""Check why categories appear empty."""

import os
from arango import ArangoClient

# Connect to database
client = ArangoClient(hosts='http://192.168.1.69:8529')
db = client.db('arxiv_abstracts_61463', username='root', password=os.environ.get('ARANGO_PASSWORD', ''))

# Check actual data for categories
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        LIMIT 10
        RETURN {
            arxiv_id: doc.arxiv_id,
            categories: doc.categories,
            categories_length: LENGTH(doc.categories),
            all_keys: ATTRIBUTES(doc)
        }
''')

print("Sample documents with category info:")
for doc in cursor:
    print(f"\narxiv_id: {doc['arxiv_id']}")
    print(f"categories: {doc['categories']}")
    print(f"categories_length: {doc['categories_length']}")
    if 'category' in doc['all_keys']:
        print("  - Found 'category' field")
    
# Check for any non-empty categories
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        FILTER LENGTH(doc.categories) > 0
        LIMIT 10
        RETURN {arxiv_id: doc.arxiv_id, categories: doc.categories}
''')

results = list(cursor)
if results:
    print(f"\n\nFound {len(results)} papers with categories:")
    for r in results:
        print(f"  {r['arxiv_id']}: {r['categories']}")
else:
    print("\n\nNo papers found with non-empty categories!")
    
# Check for alternative category fields
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        LIMIT 1
        RETURN ATTRIBUTES(doc)
''')

print("\n\nAll fields in documents:")
fields = list(cursor)[0]
for field in sorted(fields):
    print(f"  - {field}")