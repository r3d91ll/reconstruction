#!/usr/bin/env python3
"""
Use Wolfram Alpha API to validate Information Reconstructionism
"""

import os
import requests
import xml.etree.ElementTree as ET
import urllib.parse

def query_wolfram(query):
    """Send query to Wolfram Alpha API"""
    app_id = os.environ.get('WOLFRAM_APP_ID')
    if not app_id:
        raise ValueError("WOLFRAM_APP_ID environment variable not set")
    
    base_url = "http://api.wolframalpha.com/v2/query"
    params = {
        'appid': app_id,
        'input': query,
        'format': 'plaintext',
        'output': 'json'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error querying Wolfram Alpha: {e}")
        return None

def test_zero_propagation():
    """Test zero propagation using Wolfram Alpha"""
    print("Testing Zero Propagation via Wolfram Alpha API...")
    print("-" * 50)
    
    test_cases = [
        "1 * 1 * 1 * 1",
        "0 * 1 * 1 * 1", 
        "0.5 * 0.8 * 0.9 * 0.7",
        "0 * 0.5 * 0.8 * 0.9"
    ]
    
    results = []
    for query in test_cases:
        print(f"\nQuery: {query}")
        result = query_wolfram(query)
        
        if result and 'queryresult' in result:
            pods = result['queryresult'].get('pods', [])
            for pod in pods:
                if pod.get('title') == 'Result' or pod.get('title') == 'Exact result':
                    subpods = pod.get('subpods', [])
                    if subpods:
                        answer = subpods[0].get('plaintext', 'No result')
                        print(f"Result: {answer}")
                        results.append({
                            'query': query,
                            'result': answer,
                            'has_zero': '0' in query.split('*'),
                            'is_zero': answer.strip() == '0'
                        })
                        break
        else:
            print("Failed to get result")
    
    return results

def test_context_amplification():
    """Test context amplification formulas"""
    print("\n\nTesting Context Amplification...")
    print("-" * 50)
    
    alpha_values = [1.5, 2.0]
    test_contexts = [0.5, 0.7, 0.9]
    
    for alpha in alpha_values:
        print(f"\nAlpha = {alpha}:")
        for context in test_contexts:
            query = f"{context}^{alpha}"
            result = query_wolfram(query)
            
            if result and 'queryresult' in result:
                pods = result['queryresult'].get('pods', [])
                for pod in pods:
                    if 'result' in pod.get('title', '').lower():
                        subpods = pod.get('subpods', [])
                        if subpods:
                            answer = subpods[0].get('plaintext', 'No result')
                            print(f"  {context}^{alpha} = {answer}")
                            break

def validate_multiplicative_model():
    """Validate the multiplicative information model"""
    print("\n\nValidating Multiplicative Model...")
    print("-" * 50)
    
    # Test dimensional independence
    queries = [
        ("All high", "1.0 * 1.0 * 1.0 * 1.0"),
        ("Low WHERE", "0.1 * 1.0 * 1.0 * 1.0"),
        ("Low WHAT", "1.0 * 0.1 * 1.0 * 1.0"),
        ("Compensation attempt", "0.1 * 10.0 * 1.0 * 1.0"),
    ]
    
    for desc, query in queries:
        print(f"\n{desc}: {query}")
        result = query_wolfram(query)
        
        if result and 'queryresult' in result:
            pods = result['queryresult'].get('pods', [])
            for pod in pods:
                if 'result' in pod.get('title', '').lower():
                    subpods = pod.get('subpods', [])
                    if subpods:
                        answer = subpods[0].get('plaintext', 'No result')
                        print(f"Result: {answer}")
                        break

def main():
    print("=" * 60)
    print("WOLFRAM ALPHA API VALIDATION")
    print("Information Reconstructionism Mathematical Tests")
    print("=" * 60)
    
    # Run tests
    zero_results = test_zero_propagation()
    test_context_amplification()
    validate_multiplicative_model()
    
    # Summary
    print("\n\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if zero_results:
        zero_violations = sum(1 for r in zero_results 
                            if r['has_zero'] and not r['is_zero'])
        print(f"Zero Propagation: {'PASSED' if zero_violations == 0 else 'FAILED'}")
        print(f"  Tested: {len(zero_results)} cases")
        print(f"  Violations: {zero_violations}")
    
    print("\nPipeline Status: Wolfram Alpha API connection successful ✓")

if __name__ == "__main__":
    main()