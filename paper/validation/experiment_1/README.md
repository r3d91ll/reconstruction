# Experiment 1: Validating the Multiplicative Model of Information Transfer

**Dataset**: 2000 real arXiv papers from /mnt/data/arxiv_data/

## Primary Objectives

### 1. Prove the Multiplicative Nature of Information Transfer

**Hypothesis**: Information transfer requires ALL dimensions to be non-zero:
```
Information(i→j) = WHERE × WHAT × CONVEYANCE × TIME
```

**What We're Testing**:
- If ANY dimension = 0, then Information = 0 (zero propagation)
- Dimensions interact multiplicatively, not additively
- No dimension can compensate for another being zero

**How Our Code Tests This**:
```python
# zero_propagation_test.py
def test_zero_propagation(papers):
    """
    For each dimension, find natural zeros:
    - WHERE = 0: Papers behind paywalls or unavailable
    - WHAT = 0: Papers with <10% semantic overlap with implementation
    - CONVEYANCE = 0: Pure theoretical papers with no actionable content
    - TIME = 0: Future papers (temporal impossibility)
    
    Measure: Do ANY of these achieve implementation?
    Expected: Implementation rate = 0% when any dimension = 0
    """
```

### 2. Establish Baseline Dimensional Measurements

**What We're Testing**:
- Reliable measurement of each dimension using real data
- Within-dimension entropy normalization
- Dimensional independence (changing one doesn't affect others)

**How Our Code Tests This**:
```python
# dimensional_measurement.py
def measure_dimensions(paper):
    """
    WHERE: Shannon entropy of accessibility distribution → [0,1]
    WHAT: Shannon entropy of semantic overlap → [0,1]  
    CONVEYANCE: Actionability score from content → [0,1]
    TIME: Temporal availability (binary for now) → {0,1}
    
    Each dimension measured independently
    Each normalized by its maximum entropy
    """
```

### 3. Validate Against Real Implementation Data

**What We're Testing**:
- Can we predict which papers get implemented?
- Does multiplicative model outperform additive baseline?
- What's the actual implementation rate in our dataset?

**How Our Code Tests This**:
```python
# implementation_prediction.py
def compare_models(papers_with_implementations):
    """
    Model A (Additive): Score = WHERE + WHAT + CONVEYANCE + TIME
    Model B (Multiplicative): Score = WHERE × WHAT × CONVEYANCE × TIME
    
    Measure: Which better predicts actual implementations?
    Metric: AUC, precision/recall, log-likelihood
    """
```

### 4. Discover Natural Zero Occurrences

**What We're Testing**:
- How often do natural zeros occur in each dimension?
- Which dimensions are most likely to be zero?
- Do zeros cluster in certain domains/time periods?

**How Our Code Tests This**:
```python
# zero_distribution_analysis.py
def analyze_zero_distribution(papers):
    """
    Count and categorize natural zeros:
    - Paywalled papers (WHERE = 0)
    - Cross-domain papers (WHAT ≈ 0)
    - Pure theory papers (CONVEYANCE = 0)
    
    Map distribution across categories and years
    """
```

## Secondary Objectives

### 5. Prepare for Context Amplification Discovery

**What We're Testing**:
- Baseline CONVEYANCE scores WITHOUT predetermined α
- Distribution of context elements in papers
- Correlation between context and implementation

**How Our Code Tests This**:
```python
# context_baseline.py
def measure_context_distribution(papers):
    """
    Count context elements WITHOUT applying α:
    - Mathematical formulas
    - Pseudocode blocks
    - Examples
    - Code snippets
    
    Store raw counts for later α discovery
    """
```

### 6. Validate Measurement Stability

**What We're Testing**:
- Are our measurements consistent?
- Do embeddings capture semantic overlap reliably?
- Is the pipeline deterministic?

**How Our Code Tests This**:
```python
# measurement_validation.py
def validate_measurements(papers_subset):
    """
    Run measurements multiple times
    Check consistency of:
    - Embedding similarities
    - Dimension scores
    - Implementation predictions
    
    Report variance and stability metrics
    """
```

## What This Experiment Does NOT Test

1. **Context amplification α** - We don't predetermine α values
2. **FRAME discovery** - Requires citation network (experiment_2)
3. **Temporal decay** - Requires longitudinal data
4. **Cross-domain bridges** - Requires full corpus analysis

## Success Criteria

1. **Zero Propagation Confirmed**: 
   - Implementation rate = 0% when any dimension = 0
   - Statistical significance p < 0.001

2. **Multiplicative Model Superior**:
   - AUC(multiplicative) > AUC(additive) + 0.1
   - Log-likelihood ratio test significant

3. **Reliable Measurements**:
   - Test-retest reliability > 0.9
   - Dimension independence confirmed

4. **Natural Zero Distribution**:
   - At least 50 examples per dimension
   - Covers multiple domains and years

## Code Structure

```
experiment_1/
├── README.md                    # This file
├── data_preparation.py          # Load 2000 papers from arXiv
├── dimensional_measurement.py   # Measure WHERE, WHAT, CONVEYANCE, TIME
├── zero_propagation_test.py     # Test zero → zero hypothesis
├── implementation_tracking.py   # Find paper→implementation pairs
├── model_comparison.py          # Additive vs multiplicative
├── context_baseline.py          # Count context elements (no α)
├── measurement_validation.py    # Ensure measurement stability
├── analysis.py                  # Statistical analysis and reporting
└── results/
    ├── dimension_distributions.json
    ├── zero_propagation_results.json
    ├── model_comparison_results.json
    └── validation_report.md
```

## How Each Component Proves Our Hypothesis

### `data_preparation.py`
- Loads real arXiv papers (not mock data)
- Ensures diverse sampling across domains/years
- **Proves**: Results apply to real academic literature

### `dimensional_measurement.py`
- Implements within-dimension Shannon entropy
- Normalizes to [0,1] for multiplication
- **Proves**: Dimensions can be measured independently and combined

### `zero_propagation_test.py`
- Identifies natural zeros in data
- Tracks implementation outcomes
- **Proves**: Zero in any dimension → zero information transfer

### `implementation_tracking.py`
- Links papers to GitHub/citations
- Defines "successful implementation"
- **Proves**: Our model predicts real-world outcomes

### `model_comparison.py`
- Tests multiplicative vs additive
- Uses proper train/test splits
- **Proves**: Multiplication captures information dynamics better

### `context_baseline.py`
- Counts context elements objectively
- No predetermined α values
- **Proves**: We're not reverse-engineering results

## Running the Experiment

```bash
# 1. Prepare data
python data_preparation.py --papers 2000 --source /mnt/data/arxiv_data/

# 2. Measure dimensions
python dimensional_measurement.py

# 3. Test zero propagation
python zero_propagation_test.py

# 4. Compare models
python model_comparison.py

# 5. Generate report
python analysis.py --output results/validation_report.md
```

## Expected Outcomes

1. **Clear evidence** that zeros in any dimension prevent information transfer
2. **Quantitative proof** that multiplicative model predicts better than additive
3. **Baseline measurements** for all dimensions on real data
4. **Foundation** for discovering α empirically in experiment_2

This experiment establishes the core multiplicative model on real data, setting the stage for more sophisticated analyses of context amplification and FRAME discovery.