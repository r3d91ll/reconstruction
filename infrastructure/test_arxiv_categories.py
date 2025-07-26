#!/usr/bin/env python3
"""Test arXiv API to see if it returns categories."""

import urllib.request
import xml.etree.ElementTree as ET

# Test with a known paper
test_id = "2301.00001"
url = f"http://export.arxiv.org/api/query?id_list={test_id}"

print(f"Testing arXiv API with paper: {test_id}")
print(f"URL: {url}\n")

# Fetch from API
response = urllib.request.urlopen(url)
data = response.read().decode('utf-8')

# Parse XML
root = ET.fromstring(data)

# Print raw XML to see structure
print("Raw XML (first 2000 chars):")
print(data[:2000])
print("\n...")

# Find the entry
for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
    print("\n\nFound entry!")
    
    # Get ID
    id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
    if id_elem is not None:
        print(f"ID: {id_elem.text}")
    
    # Get title
    title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
    if title_elem is not None:
        print(f"Title: {title_elem.text.strip()}")
    
    # Look for categories with different approaches
    print("\nSearching for categories...")
    
    # Method 1: arxiv namespace
    categories = entry.findall('{http://arxiv.org/schemas/atom}category')
    if categories:
        print(f"Found {len(categories)} categories with arxiv namespace:")
        for cat in categories:
            print(f"  - {cat.get('term')}")
    else:
        print("No categories found with arxiv namespace")
    
    # Method 2: Look for any element with 'category' in tag
    for elem in entry.iter():
        if 'category' in elem.tag.lower():
            print(f"\nFound category element: {elem.tag}")
            print(f"  Attributes: {elem.attrib}")
            print(f"  Text: {elem.text}")
    
    # Method 3: Check all child elements
    print("\n\nAll child elements of entry:")
    for child in entry:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        print(f"  - {tag}")
        if tag == 'category':
            print(f"    Attributes: {child.attrib}")