#!/usr/bin/env python3
"""Check for institutional/affiliation data in author fields."""

import os
import re
from collections import Counter
from arango import ArangoClient

# Connect to database
client = ArangoClient(hosts='http://192.168.1.69:8529')
db = client.db('arxiv_abstracts_61463', username='root', password=os.environ.get('ARANGO_PASSWORD', ''))

print("Checking for institutional data in author fields...\n")

# Sample some author data to look for patterns
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        LIMIT 100
        RETURN {
            arxiv_id: doc.arxiv_id,
            authors: doc.authors,
            title: SUBSTRING(doc.title, 0, 50)
        }
''')

# Common institutional patterns
institution_patterns = [
    r'University',
    r'Institute',
    r'Laboratory',
    r'Lab\b',
    r'College',
    r'Department',
    r'School',
    r'Center',
    r'Centre',
    r'Research',
    r'Corporation',
    r'Corp\.',
    r'Inc\.',
    r'Company',
    r'Google',
    r'Microsoft',
    r'Meta',
    r'IBM',
    r'MIT\b',
    r'Stanford',
    r'Berkeley',
    r'CMU',
    r'Oxford',
    r'Cambridge',
    r'\([^)]+\)',  # Anything in parentheses
    r'\[[^\]]+\]',  # Anything in brackets
]

combined_pattern = '|'.join(institution_patterns)

print("Sample author fields (first 10 papers):")
has_institution_info = 0
sample_institutions = []

for i, doc in enumerate(cursor):
    if i < 10:  # Print first 10
        print(f"\nPaper: {doc['arxiv_id']} - {doc['title']}...")
        print(f"Authors ({len(doc['authors'])}):")
        for author in doc['authors'][:3]:  # First 3 authors
            print(f"  - {author}")
            # Check for institutional patterns
            if re.search(combined_pattern, author, re.IGNORECASE):
                has_institution_info += 1
                sample_institutions.append(author)
        if len(doc['authors']) > 3:
            print(f"  ... and {len(doc['authors']) - 3} more")

# Broader search for institutional data
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        LIMIT 1000
        FOR author IN doc.authors
            RETURN author
''')

all_authors = list(cursor)
authors_with_institutions = []
institution_matches = Counter()

for author in all_authors:
    match = re.search(combined_pattern, author, re.IGNORECASE)
    if match:
        authors_with_institutions.append(author)
        # Extract what looks like institution
        if '(' in author and ')' in author:
            inst = re.search(r'\(([^)]+)\)', author)
            if inst:
                institution_matches[inst.group(1)] += 1
        elif '[' in author and ']' in author:
            inst = re.search(r'\[([^\]]+)\]', author)
            if inst:
                institution_matches[inst.group(1)] += 1

print(f"\n\nInstitutional Data Analysis (from 1000 papers):")
print(f"Total author entries: {len(all_authors)}")
print(f"Authors with apparent institutional data: {len(authors_with_institutions)}")
print(f"Percentage with institutions: {len(authors_with_institutions)/len(all_authors)*100:.1f}%")

if institution_matches:
    print("\nTop 20 extracted institutions/affiliations:")
    for inst, count in institution_matches.most_common(20):
        print(f"  {inst}: {count}")
else:
    print("\nNo clear institutional patterns found in parentheses/brackets")

# Check if affiliations might be in separate field
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        LIMIT 1
        RETURN ATTRIBUTES(doc)
''')

fields = list(cursor)[0]
print(f"\n\nAll available fields:")
for field in sorted(fields):
    if 'affil' in field.lower() or 'inst' in field.lower() or 'org' in field.lower():
        print(f"  - {field} (POTENTIAL INSTITUTIONAL FIELD!)")
    else:
        print(f"  - {field}")

# Look for email domains as proxy for institutions
email_pattern = r'[\w\.-]+@([\w\.-]+\.\w+)'
cursor = db.aql.execute('''
    FOR doc IN abstract_metadata
        LIMIT 1000
        FOR author IN doc.authors
            RETURN author
''')

email_domains = Counter()
for author in cursor:
    email_match = re.search(email_pattern, author)
    if email_match:
        domain = email_match.group(1).lower()
        email_domains[domain] += 1

if email_domains:
    print(f"\n\nEmail domains found (top 20):")
    for domain, count in email_domains.most_common(20):
        print(f"  {domain}: {count}")