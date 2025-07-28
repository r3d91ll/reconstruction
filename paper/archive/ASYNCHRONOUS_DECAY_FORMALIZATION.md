# Asynchronous Dimensional Decay: A Dynamic View of Information Evolution

## Core Insight

Each dimension of information has its own decay rate, creating complex interaction dynamics that drive both information preservation and transformation.

## Mathematical Formalization

### Individual Dimension Decay

Each dimension follows its own decay function:

```
WHERE(t) = WHERE(0) × e^(-λ_WHERE × t)
WHAT(t) = WHAT(0) × e^(-λ_WHAT × t)  
CONVEYANCE(t) = CONVEYANCE(0) × e^(-λ_CONVEYANCE × t)
TIME(t) = TIME(0) × f_TIME(t)  # Not exponential decay
FRAME(i→j,t) = FRAME(i→j,0) × e^(-λ_FRAME × t)
```

Where λ represents decay constants that vary by:
- Domain (physics vs. computer science)
- Medium (paper vs. code vs. infrastructure)
- Context (well-documented vs. obscure)

### Differential Decay Rates

**Observed patterns:**

1. **WHERE (Slow Decay)**: λ_WHERE ≈ 0.01-0.05
   - Infrastructure persists (journals, repositories)
   - Access methods remain stable
   - Example: arXiv URLs remain valid for decades

2. **WHAT (Medium Decay)**: λ_WHAT ≈ 0.1-0.3
   - Semantic drift occurs gradually
   - Core concepts persist, terminology evolves
   - Example: "Neural networks" → "Deep learning" → "Transformers"

3. **CONVEYANCE (Fast Decay)**: λ_CONVEYANCE ≈ 0.5-2.0
   - Implementation methods quickly outdated
   - Tools and libraries change rapidly
   - Example: TensorFlow → PyTorch migration

4. **TIME (Linear Progress)**: No decay, but creates pressure
   - Increases distance between source and receiver
   - Amplifies other decay effects

5. **FRAME (Variable Decay)**: λ_FRAME depends on field dynamics
   - Stable fields: slow decay
   - Rapidly evolving fields: fast decay

## Interaction Dynamics

### 1. Decay Interference Patterns

When dimensions decay at different rates, interference patterns emerge:

```
I(t) = ∏ D_i(t) = ∏ D_i(0) × e^(-λ_i × t)
     = I(0) × e^(-Σλ_i × t)
```

But the multiplicative nature means fast-decaying dimensions dominate:

```
If λ_CONVEYANCE >> λ_WHERE, then I(t) ≈ 0 when CONVEYANCE ≈ 0
```

### 2. Resonance Effects

When decay rates align, information can be "revived":

```
Revival occurs when:
- New CONVEYANCE methods make old WHAT accessible again
- Updated FRAME allows reinterpretation of old ideas
- WHERE improvements (digitization) restore access
```

### 3. Critical Transitions

Phase transitions occur when one dimension crosses a threshold:

```
If D_i(t) < θ_critical, then I(t) → 0 (information collapse)
```

## Empirical Measurement

### From arXiv Data

1. **WHERE Decay**: Track URL persistence, repository availability
2. **WHAT Decay**: Measure semantic drift in citations over time
3. **CONVEYANCE Decay**: Monitor implementation language/tool obsolescence
4. **FRAME Decay**: Analyze citation patterns over decades

### Decay Signature Discovery

```python
def measure_decay_rate(dimension, paper_cohort, time_window):
    """
    Measure decay rate for a specific dimension
    """
    values = []
    for t in time_window:
        # Measure dimension value at time t
        value = measure_dimension(dimension, paper_cohort, t)
        values.append(value)
    
    # Fit exponential decay model
    λ, r_squared = fit_exponential_decay(values, time_window)
    return λ, r_squared
```

## Implications for Information Bridges

### 1. Bridge Timing

Optimal bridges form when:
- Source CONVEYANCE is still high (methods accessible)
- Target FRAME has evolved to appreciate source
- Temporal gap allows for perspective

### 2. Bridge Characteristics

Strong bridges have:
- Low CONVEYANCE decay (timeless methods)
- High WHAT stability (fundamental concepts)
- Increasing FRAME compatibility over time

### 3. Revival Patterns

Information can be revived when:
- New tools reduce CONVEYANCE barriers
- Paradigm shifts increase FRAME compatibility
- Digitization efforts restore WHERE access

## Predictive Model

### Information Survival Probability

```
P(survival, t) = ∏ P(D_i > θ_i, t)
               = ∏ (1 - F_i(θ_i, t))
```

Where F_i is the CDF of dimension i's decay distribution.

### Expected Information Lifetime

```
τ = argmin_t {P(survival, t) < 0.5}
```

The time at which information has 50% chance of being inaccessible.

## Experimental Validation

### Using Experiment 2 Data

1. **Track chunk persistence** across citations
2. **Measure semantic drift** in propagated chunks  
3. **Identify decay patterns** by discipline
4. **Validate revival events** in citation networks

### Expected Findings

1. **CS/AI**: Fast CONVEYANCE decay (λ ≈ 1.5) due to rapid tool evolution
2. **Mathematics**: Slow WHAT decay (λ ≈ 0.05) due to stable notation
3. **Physics**: Moderate uniform decay across dimensions
4. **Cross-disciplinary**: FRAME decay creates isolation over time

## Integration with Context Amplification

Decay rates affect optimal α values:

```
α_optimal(t) = α_0 × g(λ_dominant, t)
```

Where dominant decay dimension requires higher amplification to maintain information transfer.

## Conclusion

Asynchronous dimensional decay creates a rich dynamics where information doesn't simply "fade" but evolves through complex interaction patterns. Understanding these patterns enables:

1. **Prediction** of information lifetime
2. **Identification** of revival opportunities  
3. **Design** of persistent information systems
4. **Discovery** of optimal bridging strategies

This framework transforms static information measurement into a dynamic understanding of knowledge evolution.