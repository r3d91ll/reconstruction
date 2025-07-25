# Methodology Appendix: Testing Context Amplification in Information Transfer

**Detailed experimental protocols for [Context-Driven Information Transfer Investigation](./reconstructionism_theory.md)**

This appendix provides comprehensive methodology for testing the hypothesis that context acts as an exponential amplifier (Context^α) in theory-to-practice information transfer.

## Table of Contents

- [A. Experimental Design for Context Amplification](#a-experimental-design-for-context-amplification)
- [B. Data Collection and Annotation Protocols](#b-data-collection-and-annotation-protocols)
- [C. Dimensional Measurement Methodology](#c-dimensional-measurement-methodology)
- [D. Statistical Analysis Plan](#d-statistical-analysis-plan)
- [E. Implementation and Computational Requirements](#e-implementation-and-computational-requirements)
- [F. Validation Experiments](#f-validation-experiments)

## A. Experimental Design for Context Amplification

### A.1 Core Hypothesis Testing

**H1: Context Amplification Hypothesis**

Null: Context contributes additively to information transfer

```
Transfer_rate = β₀ + β₁·WHERE + β₂·WHAT + β₃·CONVEYANCE + β₄·Context
```

Alternative: Context acts as exponential amplifier

```
Transfer_rate = WHERE × WHAT × (BaseConveyance × Context^α) × TIME
where α > 1 (empirically discovered, NOT predetermined)
```

**Experimental Design (Enhanced with Bottom-up Methodology):**

Following the bottom-up curriculum approach demonstrated by Dedhia et al. (2025), we structure our experiment with progressive context levels:

1. **Dataset**: 10,000 paper-implementation pairs from arXiv/GitHub (2015-2023)
2. **Treatment Groups (Progressive Context Curriculum)**:
   - Level 1: Papers with minimal context (math only)
   - Level 2: Math + pseudocode
   - Level 3: Math + pseudocode + examples
   - Level 4: Math + pseudocode + examples + code
3. **Implementation Path Tracking**:
   - Trace how developers progress from theory to implementation
   - Identify which context elements trigger implementation decisions
   - Map "implementation traces" showing reasoning chains
4. **Outcome Measures**:
   - Primary: Implementation success within 6 months
   - Secondary: Time to implementation, code quality score
5. **Analysis**: Compare model fits using AIC/BIC, test empirically discovered α significance

### A.1.1 Empirical α Discovery (Avoiding Circular Reasoning)

**Critical Methodology Update**: We do NOT predetermine α values. Instead, we discover them empirically from ground truth data:

```python
import numpy as np

def discover_optimal_alpha(dataset, ground_truth):
    """Discover alpha empirically from data using vectorized operations"""
    alpha_candidates = np.arange(1.0, 3.0, 0.1)
    
    # Precompute base conveyance values for all papers (without alpha)
    base_conveyances = np.array([compute_base_conveyance(paper) for paper in dataset])
    
    # Vectorize alpha search: compute conveyances for all alphas at once
    # Shape: (n_alphas, n_papers)
    alphas_matrix = alpha_candidates[:, np.newaxis]
    conveyances_matrix = base_conveyances[np.newaxis, :] ** alphas_matrix
    
    # Apply prediction function vectorized across all conveyance values
    predictions_matrix = vectorized_predict_implementation(conveyances_matrix)
    
    # Evaluate all alpha candidates at once
    accuracies = np.array([
        evaluate_predictions(predictions_matrix[i], ground_truth)
        for i in range(len(alpha_candidates))
    ])
    
    # Find best alpha
    best_idx = np.argmax(accuracies)
    best_alpha = alpha_candidates[best_idx]
    best_accuracy = accuracies[best_idx]
    
    return best_alpha, best_accuracy

def compute_base_conveyance(paper):
    """Compute base conveyance value before alpha amplification"""
    # Extract conveyance features without applying alpha
    return paper.get('base_conveyance', 1.0)

def vectorized_predict_implementation(conveyances):
    """Vectorized prediction function for implementation impact"""
    # Apply threshold-based prediction or more complex model
    # Works on entire array of conveyance values
    return (conveyances > np.median(conveyances)).astype(float)
```

**Ground Truth Metrics** (independent of model):
- Implementation existence (GitHub repositories)
- Adoption metrics (stars, forks, citations)
- Time to implementation
- Cross-domain application

### A.2 Zero Propagation Testing

**H2: Multiplicative Dependency Hypothesis**

Null: Dimensions combine additively (missing dimension reduces effectiveness)
Alternative: Any dimension = 0 → Transfer rate = 0

**Experimental Protocol:**

```python
def test_zero_propagation(papers, implementations):
    """Test if zeroing any dimension prevents transfer"""
    
    # Identify paper-implementation pairs
    pairs = match_papers_to_implementations(papers, implementations)
    
    # For each dimension, find natural zeros
    zero_where = pairs[pairs.accessibility_score == 0]  # Paywalled
    zero_what = pairs[pairs.semantic_overlap < 0.1]     # Different domain  
    zero_conveyance = pairs[pairs.actionability == 0]   # Pure theory
    
    # Create boolean mask for control group (all dimensions > 0)
    all_dimensions_positive = (
        (pairs.accessibility_score > 0) & 
        (pairs.semantic_overlap > 0) & 
        (pairs.actionability > 0)
    )
    
    # Measure implementation rates
    results = {
        'zero_where': implementation_rate(zero_where),
        'zero_what': implementation_rate(zero_what),
        'zero_conveyance': implementation_rate(zero_conveyance),
        'control': implementation_rate(pairs[all_dimensions_positive])
    }
    
    # Test: Do any zero dimensions have >0 implementation?
    return chi_square_test(results)
```

### A.3 Implementation Success Prediction

**RQ3: Predictive Model Development**

Can dimensional scores predict which papers will achieve implementation?

**Training Protocol:**

```python
def train_implementation_predictor(historical_data):
    """Train classifier to predict implementation success"""
    
    # Feature extraction for each paper
    features = []
    labels = []
    
    for paper in historical_data:
        # Calculate dimensional scores
        where = calculate_accessibility(paper)
        what = calculate_semantic_clarity(paper) 
        conveyance = calculate_actionability(paper)
        context = extract_context_score(paper)
        
        # Create feature vectors for different models
        additive_features = [where, what, conveyance, context]
        multiplicative_features = [where * what * conveyance * (context ** 1.5)]
        
        # Label: Did this paper get implemented within 6 months?
        label = 1 if paper.implementation_count > 0 else 0
        
        features.append({
            'additive': additive_features,
            'multiplicative': multiplicative_features,
            'paper_id': paper.id
        })
        labels.append(label)
    
    # Extract feature arrays
    X_additive = [f['additive'] for f in features]
    X_multiplicative = [f['multiplicative'] for f in features]
    
    # Train competing models
    additive_model = LogisticRegression()
    multiplicative_model = LogisticRegression()
    
    # Compare performance using cross-validation
    from sklearn.model_selection import cross_val_score
    
    return {
        'additive_auc': cross_val_score(additive_model, X_additive, labels, cv=5, scoring='roc_auc'),
        'multiplicative_auc': cross_val_score(multiplicative_model, X_multiplicative, labels, cv=5, scoring='roc_auc'),
        'feature_importance': analyze_feature_importance(multiplicative_model.fit(X_multiplicative, labels))
    }
```

### A.4 Implementation Path Generation

**Structured Path Analysis (Inspired by KG Methodology):**

Adapting the knowledge graph path traversal approach from Dedhia et al. (2025), we generate and analyze "implementation paths" that trace theory-to-practice transfer:

```python
import networkx as nx
from typing import List, Dict, Tuple, Any
import numpy as np
from collections import defaultdict

class ImplementationPathAnalyzer:
    """Knowledge graph-based path analysis for theory-to-practice transfer"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_types = ['theory', 'algorithm', 'demonstration', 'reference', 'implementation']
        self.edge_weights = self._initialize_edge_weights()
        
    def _initialize_edge_weights(self) -> Dict[Tuple[str, str], float]:
        """Define transition probabilities between node types"""
        return {
            ('theory', 'algorithm'): 0.8,
            ('theory', 'demonstration'): 0.6,
            ('algorithm', 'demonstration'): 0.9,
            ('algorithm', 'reference'): 0.7,
            ('demonstration', 'reference'): 0.8,
            ('demonstration', 'implementation'): 0.7,
            ('reference', 'implementation'): 0.9,
            ('theory', 'implementation'): 0.3,  # Direct theory->impl is harder
        }
    
    def build_knowledge_graph(self, papers: List[Dict], implementations: List[Dict]):
        """Construct knowledge graph from papers and implementations"""
        
        # Add paper nodes with their content elements
        for paper in papers:
            paper_id = paper['id']
            
            # Add theory node
            if paper.get('has_math'):
                self.graph.add_node(f"{paper_id}_theory", 
                                  type='theory',
                                  content=paper.get('math_formulas', ''),
                                  paper_id=paper_id)
            
            # Add algorithm node
            if paper.get('has_pseudocode'):
                self.graph.add_node(f"{paper_id}_algorithm",
                                  type='algorithm', 
                                  content=paper.get('pseudocode', ''),
                                  paper_id=paper_id)
            
            # Add demonstration node
            if paper.get('has_examples'):
                self.graph.add_node(f"{paper_id}_demonstration",
                                  type='demonstration',
                                  content=paper.get('examples', ''),
                                  paper_id=paper_id)
            
            # Add reference implementation node
            if paper.get('has_code'):
                self.graph.add_node(f"{paper_id}_reference",
                                  type='reference',
                                  content=paper.get('code_snippets', ''),
                                  paper_id=paper_id)
        
        # Add implementation nodes
        for impl in implementations:
            self.graph.add_node(f"impl_{impl['id']}",
                              type='implementation',
                              content=impl.get('code', ''),
                              success_metric=impl.get('stars', 0))
        
        # Connect nodes based on semantic similarity and type transitions
        self._connect_nodes()
    
    def _connect_nodes(self):
        """Create edges between nodes based on content similarity and type transitions"""
        nodes = list(self.graph.nodes(data=True))
        
        for i, (node1, data1) in enumerate(nodes):
            for j, (node2, data2) in enumerate(nodes[i+1:], i+1):
                # Check if transition is allowed
                type1, type2 = data1['type'], data2['type']
                if (type1, type2) in self.edge_weights:
                    # Calculate semantic similarity
                    similarity = self._calculate_similarity(data1['content'], data2['content'])
                    
                    # Add edge if similarity exceeds threshold
                    if similarity > 0.5:
                        weight = self.edge_weights[(type1, type2)] * similarity
                        self.graph.add_edge(node1, node2, weight=weight)
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between content elements"""
        # Option 1: Use Jina embeddings for semantic similarity (preferred)
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # Load Jina model (cache this in __init__ for production)
            model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en')
            tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
            
            # Generate embeddings
            with torch.no_grad():
                inputs1 = tokenizer(content1, return_tensors='pt', truncation=True, max_length=512)
                inputs2 = tokenizer(content2, return_tensors='pt', truncation=True, max_length=512)
                
                emb1 = model(**inputs1).pooler_output
                emb2 = model(**inputs2).pooler_output
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
                return cos_sim.item()
                
        except ImportError:
            # Option 2: Fallback to improved bag-of-words with preprocessing
            import re
            from collections import Counter
            
            # Common English stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                         'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
                         'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
            
            def preprocess(text):
                # Convert to lowercase and extract words
                words = re.findall(r'\b\w+\b', text.lower())
                # Remove stop words and short words
                return [w for w in words if w not in stop_words and len(w) > 2]
            
            # Preprocess both texts
            words1 = preprocess(content1)
            words2 = preprocess(content2)
            
            if not words1 or not words2:
                return 0.0
            
            # Calculate Jaccard similarity on preprocessed words
            set1, set2 = set(words1), set(words2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return intersection / union if union > 0 else 0.0
    
    def generate_implementation_paths(self, papers: List[Dict], implementations: List[Dict]) -> List[Dict]:
        """Generate and analyze paths from theory to implementation"""
        
        # Build the knowledge graph
        self.build_knowledge_graph(papers, implementations)
        
        paths = []
        
        # For each paper-implementation pair
        for paper, impl in self.match_pairs(papers, implementations):
            paper_nodes = [n for n, d in self.graph.nodes(data=True) 
                          if d.get('paper_id') == paper['id']]
            impl_node = f"impl_{impl['id']}"
            
            # Find all paths from paper nodes to implementation
            all_paths = []
            for start_node in paper_nodes:
                try:
                    # Use shortest path weighted by transition probabilities
                    path = nx.shortest_path(self.graph, start_node, impl_node, weight='weight')
                    path_data = self._analyze_path(path, impl)
                    all_paths.append(path_data)
                except nx.NetworkXNoPath:
                    continue
            
            if all_paths:
                # Select best path based on multiple criteria
                best_path = max(all_paths, key=lambda p: p['quality_score'])
                paths.append(best_path)
        
        return paths
    
    def _analyze_path(self, path: List[str], implementation: Dict) -> Dict:
        """Analyze a single path for quality metrics"""
        
        nodes_data = [self.graph.nodes[node] for node in path]
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        # Calculate path metrics
        path_data = {
            'nodes': path,
            'node_types': [n['type'] for n in nodes_data],
            'length': len(path),
            'success': implementation.get('stars', 0) > 10,
            
            # Path completeness: how many node types are covered
            'completeness': len(set(n['type'] for n in nodes_data)) / len(self.node_types),
            
            # Path coherence: average edge weight (transition probability * similarity)
            'coherence': (np.mean([self.graph.edges[e]['weight'] for e in edges])
                          if edges else 0.0),
            
            # Context richness: amount of content in path nodes
            'context_richness': sum(len(n.get('content', '')) for n in nodes_data),
            
            # Direct vs indirect: whether path takes shortcuts
            # Use exponential decay to avoid sharp jumps for short paths
            'directness': np.exp(-0.3 * (len(path) - 2)) if len(path) >= 2 else 1.0,
        }
        
        # Calculate overall quality score
        path_data['quality_score'] = (
            0.3 * path_data['completeness'] +
            0.3 * path_data['coherence'] +
            0.2 * path_data['context_richness'] / 1000 +  # Normalize
            0.2 * path_data['directness']
        )
        
        return path_data
    
    def match_pairs(self, papers: List[Dict], implementations: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Match papers to their implementations based on metadata"""
        pairs = []
        
        for paper in papers:
            # Find implementations that cite or reference this paper
            matching_impls = [
                impl for impl in implementations
                if paper['id'] in impl.get('references', []) or
                paper['title'].lower() in impl.get('description', '').lower()
            ]
            
            for impl in matching_impls:
                pairs.append((paper, impl))
        
        return pairs
    
    def analyze_path_patterns(self, paths: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns across all implementation paths"""
        
        successful_paths = [p for p in paths if p['success']]
        failed_paths = [p for p in paths if not p['success']]
        
        analysis = {
            'total_paths': len(paths),
            'success_rate': len(successful_paths) / len(paths) if paths else 0,
            
            # Compare metrics between successful and failed paths
            'successful_patterns': {
                'avg_length': np.mean([p['length'] for p in successful_paths]) if successful_paths else 0,
                'avg_completeness': np.mean([p['completeness'] for p in successful_paths]) if successful_paths else 0,
                'avg_coherence': np.mean([p['coherence'] for p in successful_paths]) if successful_paths else 0,
                'common_sequences': self._find_common_sequences(successful_paths),
            },
            
            'failed_patterns': {
                'avg_length': np.mean([p['length'] for p in failed_paths]) if failed_paths else 0,
                'avg_completeness': np.mean([p['completeness'] for p in failed_paths]) if failed_paths else 0,
                'avg_coherence': np.mean([p['coherence'] for p in failed_paths]) if failed_paths else 0,
                'missing_elements': self._find_missing_elements(failed_paths),
            },
            
            # Identify critical transitions
            'critical_transitions': self._identify_critical_transitions(paths),
        }
        
        return analysis
    
    def _find_common_sequences(self, paths: List[Dict]) -> List[Tuple[str, ...]]:
        """Find common node type sequences in paths"""
        sequences = defaultdict(int)
        
        for path in paths:
            types = tuple(path['node_types'])
            # Check all subsequences of length 2-3
            for length in [2, 3]:
                for i in range(len(types) - length + 1):
                    subseq = types[i:i+length]
                    sequences[subseq] += 1
        
        # Return most common sequences
        return sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _find_missing_elements(self, paths: List[Dict]) -> List[str]:
        """Identify which node types are commonly missing in failed paths"""
        all_types = set(self.node_types)
        missing_counts = defaultdict(int)
        
        for path in paths:
            present_types = set(path['node_types'])
            for missing in all_types - present_types:
                missing_counts[missing] += 1
        
        return sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)
    
    def _identify_critical_transitions(self, paths: List[Dict]) -> List[Tuple[str, str, float]]:
        """Identify transitions that correlate with success"""
        transition_success = defaultdict(lambda: {'count': 0, 'success': 0})
        
        for path in paths:
            types = path['node_types']
            for i in range(len(types) - 1):
                transition = (types[i], types[i+1])
                transition_success[transition]['count'] += 1
                if path['success']:
                    transition_success[transition]['success'] += 1
        
        # Calculate success rates for each transition
        critical = []
        for transition, stats in transition_success.items():
            if stats['count'] >= 5:  # Minimum sample size
                success_rate = stats['success'] / stats['count']
                critical.append((transition[0], transition[1], success_rate))
        
        return sorted(critical, key=lambda x: x[2], reverse=True)

# Usage example
def analyze_implementation_paths(papers, implementations):
    """Main function to perform path analysis"""
    analyzer = ImplementationPathAnalyzer()
    
    # Generate paths
    paths = analyzer.generate_implementation_paths(papers, implementations)
    
    # Analyze patterns
    patterns = analyzer.analyze_path_patterns(paths)
    
    return {
        'paths': paths,
        'patterns': patterns,
        'graph_stats': {
            'nodes': analyzer.graph.number_of_nodes(),
            'edges': analyzer.graph.number_of_edges(),
            'density': nx.density(analyzer.graph),
        }
    }
```

**Path-based Metrics:**

- Path length: Number of context elements traversed
- Path completeness: Percentage of possible nodes included
- Path coherence: Semantic similarity between adjacent nodes
- Path effectiveness: Correlation with implementation success

### A.5 A/B Testing Protocol for Retrieval Systems

**RQ4: Conveyance-Weighted vs Semantic Similarity Retrieval**

**Experimental Setup:**

```python
def ab_test_retrieval_systems():
    """A/B test comparing retrieval approaches"""
    
    # Participant recruitment
    participants = recruit_developers(n=100, experience_level='intermediate')
    
    # Task design
    tasks = [
        "Find papers to implement a recommendation system",
        "Find research for building a chatbot",
        "Find papers on efficient neural network training",
        # ... 10 total tasks
    ]
    
    # Random assignment to conditions
    control_group = participants[:50]  # Semantic similarity
    treatment_group = participants[50:]  # Conveyance-weighted
    
    # Measurement protocol
    for participant in participants:
        task = random.choice(tasks)
        
        if participant in control_group:
            results = semantic_similarity_search(task)
        else:
            results = conveyance_weighted_search(task)
        
        # Present top-5 results
        participant.review_papers(results[:5])
        
        # Outcome measures (30-day follow-up)
        outcomes = {
            'attempted_implementation': bool,  # Did they try?
            'successful_implementation': bool,  # Did they succeed?
            'time_to_implementation': float,   # Hours spent
            'self_rated_usefulness': int,      # 1-10 scale
            'code_quality_score': float        # External review
        }
        
    # Statistical analysis
    return {
        'implementation_rate_difference': treatment_rate - control_rate,
        'p_value': chi_square_test(treatment_outcomes, control_outcomes),
        'effect_size': cohens_d(treatment_times, control_times)
    }
```

## B. Data Collection and Annotation Protocols

### B.1 Paper-Implementation Pair Identification

**Objective**: Create ground truth dataset of verified paper→implementation links

**Data Sources:**

1. **Papers**: arXiv CS categories (cs.AI, cs.LG, cs.CL, cs.CV)
2. **Implementations**: GitHub repositories with >10 stars
3. **Linking Methods**:
   - Explicit citations in README/documentation
   - Paper title in repository description
   - Author overlap verification
   - Semantic similarity > 0.9 with manual verification

**Diversity Sampling Protocol:**

Following the inverse frequency weighting approach of Dedhia et al. (2025), we ensure broad coverage and avoid clustering around popular papers:

```python
def diversity_aware_sampling(papers, target_size=10000):
    """Sample papers with inverse frequency weighting"""
    
    # Track sampling frequency by domain and year
    frequency = defaultdict(lambda: defaultdict(int))
    selected = []
    
    while len(selected) < target_size:
        # Calculate sampling weights
        weights = []
        for paper in papers:
            domain = paper.primary_category
            year = paper.year
            
            # Inverse frequency weighting
            f = frequency[domain][year]
            weight = 1.0 / (1 + f)
            
            # Adjust for domain balance
            domain_weight = 1.0 / (1 + len([p for p in selected 
                                           if p.primary_category == domain]))
            
            weights.append(weight * domain_weight)
        
        # Sample paper
        paper = np.random.choice(papers, p=weights/sum(weights))
        selected.append(paper)
        frequency[paper.primary_category][paper.year] += 1
    
    return selected
```

**Collection Protocol:**

```python
def collect_paper_implementation_pairs():
    """Systematic collection of verified pairs"""
    
    pairs = []
    
    # Method 1: GitHub search for arXiv links
    for repo in github.search_repositories("arxiv.org"):
        arxiv_ids = extract_arxiv_references(repo.readme)
        for arxiv_id in arxiv_ids:
            paper = fetch_arxiv_metadata(arxiv_id)
            pairs.append({
                'paper': paper,
                'implementation': repo,
                'link_type': 'explicit_citation',
                'confidence': 1.0
            })
    
    # Method 2: Paper title search
    for paper in arxiv_papers:
        repos = github.search_repositories(paper.title)
        for repo in repos:
            if verify_implementation(paper, repo):
                pairs.append({
                    'paper': paper,
                    'implementation': repo,
                    'link_type': 'title_match',
                    'confidence': calculate_match_confidence(paper, repo)
                })
    
    return filter_high_confidence_pairs(pairs, threshold=0.8)
```

### B.2 Context Element Annotation

**Annotation Schema:**

Each paper annotated for presence/absence of context elements:

```python
CONTEXT_ELEMENTS = {
    'mathematical_formulas': {
        'present': bool,
        'clarity': float,  # 0-1 scale
        'count': int
    },
    'pseudocode': {
        'present': bool,
        'detail_level': ['high', 'medium', 'low'],
        'executable': bool
    },
    'examples': {
        'present': bool,
        'concrete': bool,  # Specific numbers vs abstract
        'count': int
    },
    'code_snippets': {
        'present': bool,
        'language': str,
        'runnable': bool
    },
    'diagrams': {
        'present': bool,
        'types': ['architecture', 'flowchart', 'results'],
        'count': int
    },
    'hyperparameters': {
        'present': bool,
        'complete': bool,  # All params specified
        'justified': bool  # Reasoning provided
    }
}
```

**Inter-annotator Agreement Protocol:**

- 3 independent annotators per paper
- Cohen's kappa > 0.7 required
- Disagreements resolved by majority vote

## C. Dimensional Measurement Methodology

### C.1 WHERE Dimension: Accessibility Scoring (Within-Dimension Entropy)

**Measurement Protocol:**

```python
def measure_where_dimension(paper):
    """Calculate accessibility score using within-dimension entropy normalization"""
    
    # Calculate entropy for each accessibility component
    access_components = {
        'open_access': 1.0 if paper.is_open_access else 0.0,
        'preprint_available': 1.0 if paper.has_arxiv else 0.5,
        'institution_access': estimate_institutional_coverage(paper),
        'language': 1.0 if paper.language == 'en' else 0.3,
        'format_accessibility': check_pdf_quality(paper)
    }
    
    # Calculate Shannon entropy within WHERE dimension
    H_WHERE = calculate_shannon_entropy(access_components)  # bits
    H_max_WHERE = np.log2(len(access_components))  # Maximum possible entropy
    
    # Normalize to [0,1] for multiplicative model
    WHERE_norm = H_WHERE / H_max_WHERE
    
    return WHERE_norm
```

### C.2 WHAT Dimension: Semantic Clarity (Within-Dimension Entropy)

**Measurement Using Embeddings:**

```python
def measure_what_dimension(paper, implementation):
    """Calculate semantic overlap using within-dimension entropy normalization"""
    
    # Extract text for embedding
    paper_text = f"{paper.title} {paper.abstract} {paper.introduction}"
    impl_text = f"{implementation.readme} {implementation.description}"
    
    # Generate embeddings
    paper_embedding = embed_with_jina(paper_text)
    impl_embedding = embed_with_jina(impl_text)
    
    # Calculate semantic entropy distribution
    semantic_distribution = calculate_semantic_distribution(paper_embedding, impl_embedding)
    
    # Calculate Shannon entropy within WHAT dimension
    H_WHAT = calculate_shannon_entropy(semantic_distribution)  # bits
    H_max_WHAT = calculate_max_semantic_entropy()  # Maximum possible entropy
    
    # Normalize to [0,1] for multiplicative model
    WHAT_norm = H_WHAT / H_max_WHAT
    
    return WHAT_norm
```

### C.3 CONVEYANCE Dimension: Actionability Measurement

**Comprehensive Scoring:**

```python
def measure_conveyance_dimension(paper, context_elements):
    """Calculate actionability with context amplification"""
    
    # Base conveyance from implementation guidance
    base_scores = {
        'has_algorithm': 0.2 if paper.has_clear_algorithm else 0.0,
        'has_pseudocode': 0.3 if context_elements['pseudocode']['present'] else 0.0,
        'has_examples': 0.2 if context_elements['examples']['concrete'] else 0.0,
        'has_code': 0.3 if context_elements['code_snippets']['runnable'] else 0.0
    }
    
    base_conveyance = sum(base_scores.values())
    
    # Context score calculation
    context_score = calculate_context_score(context_elements)
    
    # Apply exponential amplification with empirically discovered α
    # NOTE: α is discovered from data, not predetermined
    alpha = discover_optimal_alpha_for_domain(paper.domain)
    amplified_conveyance = base_conveyance * (context_score ** alpha)
    
    return min(amplified_conveyance, 1.0)  # Cap at 1.0
```

### C.4 Context Score Calculation

```python
def calculate_context_score(elements):
    """Aggregate context elements into single score"""
    
    element_weights = {
        'mathematical_formulas': 0.15,
        'pseudocode': 0.25,
        'examples': 0.20,
        'code_snippets': 0.20,
        'diagrams': 0.10,
        'hyperparameters': 0.10
    }
    
    score = 0.0
    for element, weight in element_weights.items():
        if elements[element]['present']:
            quality = elements[element].get('clarity', 0.8)
            score += weight * quality
    
    return score

```

## D. Statistical Analysis Plan

### D.1 Model Comparison Framework

**Comparing Additive vs Multiplicative Models:**

```python
def compare_models(data):
    """Statistical comparison of competing hypotheses"""
    
    # Prepare data
    X = data[['where', 'what', 'conveyance', 'context']]
    y = data['implemented_within_6_months']
    
    # Model 1: Additive (null hypothesis)
    additive_formula = 'implemented ~ where + what + conveyance + context'
    additive_model = smf.logit(additive_formula, data).fit()
    
    # Model 2: Multiplicative with exponential context
    data['context_exp'] = data['context'] ** 1.67
    data['multiplicative_score'] = (data['where'] * data['what'] * 
                                    data['conveyance'] * data['context_exp'])
    multiplicative_formula = 'implemented ~ multiplicative_score'
    multiplicative_model = smf.logit(multiplicative_formula, data).fit()
    
    # Model comparison
    likelihood_ratio = -2 * (additive_model.llf - multiplicative_model.llf)

    results = {
        'additive_aic': additive_model.aic,
        'multiplicative_aic': multiplicative_model.aic,
        'likelihood_ratio': likelihood_ratio,
        'p_value': chi2.sf(likelihood_ratio, df=3)
    }
    
    # Cross-validation
    additive_cv = cross_val_score(LogisticRegression(), X, y, cv=10)
    mult_cv = cross_val_score(LogisticRegression(), 
                              data[['multiplicative_score']], y, cv=10)
    
    results['additive_cv_mean'] = additive_cv.mean()
    results['multiplicative_cv_mean'] = mult_cv.mean()
    
    return results
```

### D.2 Power Analysis

**Sample Size Determination:**

```python
def calculate_required_sample_size():
    """Determine sample size for adequate power"""
    
    # Expected effect sizes from pilot
    effect_size = 0.3  # Medium effect
    alpha = 0.05
    power = 0.80
    
    # For logistic regression with 4 predictors
    from statsmodels.stats.power import zt_ind_solve_power
    
    n_required = zt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1,
        alternative='two-sided'
    )
    
    # Adjust for expected implementation rate (~40%)
    n_adjusted = n_required / 0.4
    
    return int(np.ceil(n_adjusted))
```

## E. Implementation and Computational Requirements

### E.1 Technical Infrastructure

**System Requirements:**

```python
INFRASTRUCTURE = {
    'compute': {
        'gpus': 2,  # NVIDIA A6000 or equivalent
        'cpu_cores': 24,
        'ram_gb': 256
    },
    'storage': {
        'papers_raw': '500GB',  # 10K papers with PDFs
        'implementations': '2TB',  # GitHub repos
        'embeddings': '100GB',   # Jina V4 embeddings
        'results': '50GB'
    },
    'software': {
        'python': '3.10+',
        'frameworks': ['pytorch', 'transformers', 'scikit-learn'],
        'databases': ['PostgreSQL', 'Redis'],
        'apis': ['arXiv', 'GitHub', 'Jina']
    }
}
```

### E.2 Embedding Generation Pipeline

```python
def generate_embeddings_pipeline():
    """Parallel embedding generation for papers and implementations"""
    
    from transformers import AutoModel
    import torch.multiprocessing as mp
    
    # Load Jina model on each GPU
    model_gpu0 = AutoModel.from_pretrained('jinaai/jina-embeddings-v4').to('cuda:0')
    model_gpu1 = AutoModel.from_pretrained('jinaai/jina-embeddings-v4').to('cuda:1')
    
    # Split workload
    papers_gpu0, papers_gpu1 = split_dataset(papers, n_splits=2)
    
    # Parallel processing
    with mp.Pool(processes=2) as pool:
        embeddings_0 = pool.apply_async(embed_batch, (papers_gpu0, model_gpu0))
        embeddings_1 = pool.apply_async(embed_batch, (papers_gpu1, model_gpu1))
        
        all_embeddings = embeddings_0.get() + embeddings_1.get()
    
    return all_embeddings
```

## F. Validation Experiments

### F.1 Pilot Study Replication

**Validate Initial Findings on Larger Scale:**

```python
def replicate_pilot_study(n_papers=1000):
    """Replicate pilot findings with larger sample"""
    
    # Sample papers stratified by year and category
    papers = stratified_sample(arxiv_papers, n=n_papers)
    
    # Collect implementation data
    implementations = []
    for paper in papers:
        impls = find_implementations(paper)
        implementations.extend(impls)
    
    # Annotate context elements
    annotated_papers = parallel_annotate(papers, n_annotators=3)
    
    # Calculate implementation rates by context level
    results = {}
    for context_level in ['minimal', 'moderate', 'rich', 'complete']:
        subset = filter_by_context_level(annotated_papers, context_level)
        rate = calculate_implementation_rate(subset)
        results[context_level] = rate
    
    # Fit models
    linear_fit = fit_linear_model(results)
    power_fit = fit_power_law(results)
    
    return {
        'implementation_rates': results,
        'linear_r2': linear_fit.r2,
        'power_r2': power_fit.r2,
        'estimated_alpha': power_fit.exponent
    }
```

### F.2 Cross-Domain Validation

**Test Generalizability Across CS Subfields:**

```python
def validate_across_domains():
    """Test if α varies by domain"""
    
    domains = {
        'ml': ['cs.LG', 'cs.AI'],
        'systems': ['cs.OS', 'cs.DC'],
        'theory': ['cs.DS', 'cs.CC'],
        'hci': ['cs.HC', 'cs.CY']
    }
    
    domain_alphas = {}
    
    for domain_name, categories in domains.items():
        # Get domain-specific papers
        papers = get_papers_by_category(categories, n=500)
        
        # Measure context amplification
        alpha = estimate_context_exponent(papers)
        domain_alphas[domain_name] = alpha
        
        print(f"{domain_name}: α = {alpha:.2f}")
    
    # Test if alphas significantly differ
    anova_result = f_oneway(*[domain_alphas[d] for d in domains])
    
    return {
        'domain_alphas': domain_alphas,
        'significant_difference': anova_result.pvalue < 0.05
    }
```

### F.3 Temporal Validation

**Ensure Findings Hold Over Time:**

```python
def temporal_validation():
    """Test stability of findings across years"""
    
    years = range(2015, 2024)
    yearly_results = {}
    
    for year in years:
        papers = get_papers_by_year(year, n=200)
        
        # Measure key metrics
        results = {
            'implementation_rate': calculate_overall_implementation_rate(papers),
            'context_alpha': estimate_context_exponent(papers),
            'zero_propagation_verified': test_zero_propagation(papers)
        }
        
        yearly_results[year] = results
    
    # Trend analysis
    trend = linear_regression(years, [r['context_alpha'] for r in yearly_results.values()])
    
    return {
        'yearly_results': yearly_results,
        'alpha_trend': trend.slope,
        'trend_significant': trend.pvalue < 0.05
    }
```

## Conclusion

This methodology provides rigorous experimental protocols for testing whether context acts as an exponential amplifier in theory-to-practice information transfer. The multi-method approach combining observational data, controlled experiments, and predictive modeling will provide converging evidence for or against our hypotheses.

Key strengths:

- Large-scale data collection (10,000+ papers)
- Multiple validation approaches
- Clear statistical criteria
- Reproducible protocols

Expected timeline:

- Months 1-3: Data collection and annotation
- Months 4-5: Experimental execution
- Month 6: Analysis and reporting

The methodology balances theoretical rigor with practical feasibility, providing a clear path to testing our core hypotheses about information transfer dynamics.

## Methodological Acknowledgments

This experimental design incorporates several key methodological innovations from recent work in domain-specific AI:

1. **Bottom-up Curriculum Design**: Our progressive context levels (math → pseudocode → examples → code) adapt the hop-based complexity progression demonstrated by Dedhia et al. (2025) in their knowledge graph traversal approach.

2. **Path-based Analysis**: The concept of tracing "implementation paths" from theory to practice parallels their KG path methodology, where multi-hop traversals capture increasingly complex relationships.

3. **Diversity Sampling**: Our inverse frequency weighting ensures broad domain coverage, preventing the clustering effects observed in naive sampling approaches.

4. **Multi-stage Quality Control**: The filtering pipeline ensures high-quality data while maintaining the scientific rigor necessary for hypothesis testing.

These methodological choices position our work within the broader context of structured knowledge representation and transfer, while maintaining our unique focus on context as an exponential amplifier in information dynamics.

## G. FRAME Discovery Through Citation Networks

### G.1 Citation-Based FRAME Measurement

**Empirical Discovery of Directional Compatibility:**

```python
def discover_frame_from_citations(citation_network):
    """Discover FRAME as directional compatibility from citation patterns"""
    
    frame_edges = []
    
    for source_paper in citation_network.nodes():
        for target_paper in source_paper.cites:
            # Temporal constraint (information flows forward)
            if source_paper.timestamp > target_paper.timestamp:
                frame_score = 0
            else:
                # Measure actual information flow
                semantic_overlap = measure_chunk_propagation(source_paper, target_paper)
                citation_strength = get_citation_context_strength(source_paper, target_paper)
                temporal_distance = target_paper.timestamp - source_paper.timestamp
                
                # FRAME emerges from these measurements
                frame_score = compute_directional_compatibility(
                    semantic_overlap, citation_strength, temporal_distance
                )
            
            frame_edges.append({
                'source': source_paper.id,
                'target': target_paper.id,
                'frame_forward': frame_score,
                'frame_backward': 0 if frame_score > 0 else compute_reverse_frame()
            })
    
    return frame_edges
```

### G.2 Asynchronous Decay Measurement

**Tracking Dimensional Decay Rates:**

```python
def measure_dimensional_decay(paper_cohort, time_window):
    """Measure asynchronous decay rates for each dimension"""
    
    decay_rates = {}
    
    for dimension in ['WHERE', 'WHAT', 'CONVEYANCE', 'TIME']:
        values_over_time = []
        
        for t in time_window:
            # Measure dimension value at time t
            if dimension == 'WHERE':
                value = measure_infrastructure_persistence(paper_cohort, t)
            elif dimension == 'WHAT':
                value = measure_semantic_drift(paper_cohort, t)
            elif dimension == 'CONVEYANCE':
                value = measure_method_obsolescence(paper_cohort, t)
            else:  # TIME
                value = 1.0  # TIME doesn't decay, it progresses
            
            values_over_time.append(value)
        
        # Fit exponential decay model
        if dimension != 'TIME':
            lambda_decay, r_squared = fit_exponential_decay(values_over_time, time_window)
            decay_rates[dimension] = {
                'lambda': lambda_decay,
                'half_life': np.log(2) / lambda_decay,
                'r_squared': r_squared
            }
    
    return decay_rates
```

## H. Updated Conclusion

This methodology provides rigorous experimental protocols for testing whether context acts as an exponential amplifier in theory-to-practice information transfer, while addressing all major theoretical critiques:

**Key Methodological Innovations:**
1. **Empirical α Discovery**: Avoiding circular reasoning by discovering α from ground truth
2. **Within-Dimension Entropy**: Preserving Shannon's formalism while enabling multiplication
3. **FRAME as Emergent Property**: Discovering directional compatibility from citation networks
4. **Asynchronous Decay Tracking**: Measuring different decay rates per dimension

The methodology maintains theoretical rigor while ensuring all critiques from the review are addressed through empirical, data-driven approaches.
