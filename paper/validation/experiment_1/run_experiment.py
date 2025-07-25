#!/usr/bin/env python3
"""
Experiment 1: Multiplicative Model Validation

With all infrastructure pre-computed, this experiment focuses purely on
testing the multiplicative nature of information transfer.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy import stats

from irec_infrastructure.database import ExperimentBase


class MultiplicativeModelExperiment(ExperimentBase):
    """
    Test the core hypothesis that information transfer requires ALL dimensions > 0.
    
    Uses pre-computed infrastructure to focus on hypothesis validation.
    """
    
    def run_experiment(self) -> Dict:
        """Run the multiplicative model validation."""
        self.logger.info("Starting Experiment 1: Multiplicative Model Validation")
        
        results = {
            "zero_propagation": self.test_zero_propagation(),
            "model_comparison": self.compare_additive_vs_multiplicative(),
            "dimensional_independence": self.test_dimensional_independence(),
            "natural_zeros": self.analyze_natural_zero_distribution()
        }
        
        # Save results
        self.save_results(results, "experiment_1_results.json")
        
        return results
    
    def test_zero_propagation(self) -> Dict:
        """
        Test: If ANY dimension = 0, then Information = 0
        
        This is the core claim of the multiplicative model.
        """
        self.logger.info("Testing zero propagation hypothesis...")
        
        # Get papers with dimensional scores
        papers = self.get_papers()
        dim_scores = self.get_dimensional_scores()
        implementations = self.get_implementations()
        
        # Create implementation lookup
        impl_lookup = {impl['paper_id']: impl for impl in implementations}
        
        # Find papers with zero dimensions
        zero_where = []
        zero_what = []
        zero_conveyance = []
        control_group = []
        
        for paper in papers:
            paper_id = paper['_id']
            if paper_id not in dim_scores:
                continue
                
            scores = dim_scores[paper_id]
            
            # Classify papers by zero dimensions
            if scores['WHERE_score'] == 0:
                zero_where.append(paper_id)
            elif scores['WHAT_score'] == 0:
                zero_what.append(paper_id)
            elif scores['CONVEYANCE_score'] == 0:
                zero_conveyance.append(paper_id)
            elif all(scores[f'{dim}_score'] > 0 for dim in ['WHERE', 'WHAT', 'CONVEYANCE', 'TIME']):
                control_group.append(paper_id)
        
        # Calculate implementation rates
        def impl_rate(paper_ids):
            if not paper_ids:
                return 0.0
            implemented = sum(1 for pid in paper_ids if pid in impl_lookup)
            return implemented / len(paper_ids)
        
        results = {
            "zero_WHERE": {
                "count": len(zero_where),
                "implementation_rate": impl_rate(zero_where)
            },
            "zero_WHAT": {
                "count": len(zero_what),
                "implementation_rate": impl_rate(zero_what)
            },
            "zero_CONVEYANCE": {
                "count": len(zero_conveyance),
                "implementation_rate": impl_rate(zero_conveyance)
            },
            "control_group": {
                "count": len(control_group),
                "implementation_rate": impl_rate(control_group)
            }
        }
        
        # Statistical test
        # Chi-square test: Are implementation rates significantly different?
        observed = [
            len(zero_where) * results["zero_WHERE"]["implementation_rate"],
            len(zero_what) * results["zero_WHAT"]["implementation_rate"],
            len(zero_conveyance) * results["zero_CONVEYANCE"]["implementation_rate"],
            len(control_group) * results["control_group"]["implementation_rate"]
        ]
        
        # Expected under null hypothesis (all equal to control rate)
        control_rate = results["control_group"]["implementation_rate"]
        expected = [
            len(zero_where) * control_rate,
            len(zero_what) * control_rate,
            len(zero_conveyance) * control_rate,
            len(control_group) * control_rate
        ]
        
        chi2, p_value = stats.chisquare(observed[:3], expected[:3])
        
        results["statistical_test"] = {
            "chi_square": chi2,
            "p_value": p_value,
            "significant": p_value < 0.001
        }
        
        self.logger.info(f"Zero propagation test complete. p-value: {p_value:.6f}")
        
        return results
    
    def compare_additive_vs_multiplicative(self) -> Dict:
        """
        Compare predictive power of additive vs multiplicative models.
        """
        self.logger.info("Comparing additive vs multiplicative models...")
        
        # Get data
        papers = self.get_papers()
        dim_scores = self.get_dimensional_scores()
        implementations = self.get_implementations()
        
        # Create implementation labels
        impl_set = {impl['paper_id'] for impl in implementations}
        
        # Prepare data for modeling
        X_additive = []
        X_multiplicative = []
        y = []
        
        for paper in papers:
            paper_id = paper['_id']
            if paper_id not in dim_scores:
                continue
            
            scores = dim_scores[paper_id]
            
            # Additive features
            additive = sum(scores[f'{dim}_score'] for dim in ['WHERE', 'WHAT', 'CONVEYANCE', 'TIME'])
            
            # Multiplicative feature
            multiplicative = 1.0
            for dim in ['WHERE', 'WHAT', 'CONVEYANCE', 'TIME']:
                multiplicative *= scores[f'{dim}_score']
            
            X_additive.append(additive)
            X_multiplicative.append(multiplicative)
            y.append(1 if paper_id in impl_set else 0)
        
        # Convert to arrays
        X_additive = np.array(X_additive).reshape(-1, 1)
        X_multiplicative = np.array(X_multiplicative).reshape(-1, 1)
        y = np.array(y)
        
        # Train simple logistic regression models
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import roc_auc_score
        
        # Additive model
        lr_additive = LogisticRegression()
        auc_additive = cross_val_score(lr_additive, X_additive, y, cv=5, scoring='roc_auc').mean()
        
        # Multiplicative model
        lr_multiplicative = LogisticRegression()
        auc_multiplicative = cross_val_score(lr_multiplicative, X_multiplicative, y, cv=5, scoring='roc_auc').mean()
        
        results = {
            "additive_model": {
                "auc": float(auc_additive),
                "feature_type": "sum"
            },
            "multiplicative_model": {
                "auc": float(auc_multiplicative),
                "feature_type": "product"
            },
            "improvement": float(auc_multiplicative - auc_additive),
            "multiplicative_superior": auc_multiplicative > auc_additive + 0.05
        }
        
        self.logger.info(f"Model comparison complete. Multiplicative AUC: {auc_multiplicative:.3f}")
        
        return results
    
    def test_dimensional_independence(self) -> Dict:
        """
        Test that dimensions are independent (changing one doesn't affect others).
        """
        self.logger.info("Testing dimensional independence...")
        
        dim_scores = self.get_dimensional_scores()
        
        # Extract dimension values
        dimensions = ['WHERE', 'WHAT', 'CONVEYANCE', 'TIME']
        dim_values = {dim: [] for dim in dimensions}
        
        for scores in dim_scores.values():
            for dim in dimensions:
                dim_values[dim].append(scores[f'{dim}_score'])
        
        # Calculate correlation matrix
        import pandas as pd
        df = pd.DataFrame(dim_values)
        corr_matrix = df.corr()
        
        # Test for significant correlations
        n = len(df)
        correlations = {}
        
        for i, dim1 in enumerate(dimensions):
            for j, dim2 in enumerate(dimensions):
                if i < j:  # Upper triangle only
                    r = corr_matrix.loc[dim1, dim2]
                    # Test significance
                    t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
                    p = 2 * (1 - stats.t.cdf(abs(t), n - 2))
                    
                    correlations[f"{dim1}_vs_{dim2}"] = {
                        "correlation": float(r),
                        "p_value": float(p),
                        "significant": p < 0.05
                    }
        
        results = {
            "correlation_matrix": corr_matrix.to_dict(),
            "pairwise_tests": correlations,
            "dimensions_independent": all(not c["significant"] for c in correlations.values())
        }
        
        return results
    
    def analyze_natural_zero_distribution(self) -> Dict:
        """
        Analyze how often natural zeros occur in each dimension.
        """
        self.logger.info("Analyzing natural zero distribution...")
        
        papers = self.get_papers()
        dim_scores = self.get_dimensional_scores()
        
        # Count zeros by dimension and category
        zero_counts = {
            'WHERE': {'total': 0, 'by_category': {}},
            'WHAT': {'total': 0, 'by_category': {}},
            'CONVEYANCE': {'total': 0, 'by_category': {}},
            'TIME': {'total': 0, 'by_category': {}}
        }
        
        for paper in papers:
            paper_id = paper['_id']
            if paper_id not in dim_scores:
                continue
            
            scores = dim_scores[paper_id]
            primary_category = paper['categories'][0] if paper['categories'] else 'unknown'
            
            for dim in ['WHERE', 'WHAT', 'CONVEYANCE', 'TIME']:
                if scores[f'{dim}_score'] == 0:
                    zero_counts[dim]['total'] += 1
                    
                    if primary_category not in zero_counts[dim]['by_category']:
                        zero_counts[dim]['by_category'][primary_category] = 0
                    zero_counts[dim]['by_category'][primary_category] += 1
        
        # Calculate percentages
        total_papers = len(papers)
        results = {}
        
        for dim, counts in zero_counts.items():
            results[dim] = {
                'total_zeros': counts['total'],
                'percentage': counts['total'] / total_papers * 100,
                'by_category': counts['by_category']
            }
        
        return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run experiment
    experiment = MultiplicativeModelExperiment()
    results = experiment.run_experiment()
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT 1 RESULTS: Multiplicative Model Validation")
    print("="*60)
    
    # Zero propagation results
    zp = results["zero_propagation"]
    print("\nZero Propagation Test:")
    print(f"  Papers with WHERE=0: {zp['zero_WHERE']['count']} (impl rate: {zp['zero_WHERE']['implementation_rate']:.1%})")
    print(f"  Papers with WHAT=0: {zp['zero_WHAT']['count']} (impl rate: {zp['zero_WHAT']['implementation_rate']:.1%})")
    print(f"  Papers with CONVEYANCE=0: {zp['zero_CONVEYANCE']['count']} (impl rate: {zp['zero_CONVEYANCE']['implementation_rate']:.1%})")
    print(f"  Control group: {zp['control_group']['count']} (impl rate: {zp['control_group']['implementation_rate']:.1%})")
    print(f"  Statistical significance: p = {zp['statistical_test']['p_value']:.6f}")
    
    # Model comparison
    mc = results["model_comparison"]
    print(f"\nModel Comparison:")
    print(f"  Additive model AUC: {mc['additive_model']['auc']:.3f}")
    print(f"  Multiplicative model AUC: {mc['multiplicative_model']['auc']:.3f}")
    print(f"  Improvement: {mc['improvement']:.3f}")
    print(f"  Multiplicative superior: {mc['multiplicative_superior']}")
    
    print("\nExperiment complete!")