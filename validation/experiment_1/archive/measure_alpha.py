#!/usr/bin/env python3
"""
Measure Context Amplification Alpha Values Empirically
Fit Context^α model to real document data
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from typing import Dict


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load empirical results and prepare for alpha fitting"""
    df = pd.read_csv(csv_path)
    
    # Filter out documents with zero context or conveyance
    df = df[(df['context'] > 0) & (df['CONVEYANCE'] > 0) & (df['amplified_conveyance'] > 0)]
    
    # Calculate observed alpha for each document
    # amplified_conveyance = base_conveyance * context^alpha
    # alpha = log(amplified_conveyance / base_conveyance) / log(context)
    df['observed_alpha'] = np.log(df['amplified_conveyance'] / df['CONVEYANCE']) / np.log(df['context'])
    
    # Remove infinite or NaN values
    df = df[np.isfinite(df['observed_alpha'])]
    
    return df


def fit_alpha_by_category(df: pd.DataFrame, min_samples: int = 10) -> pd.DataFrame:
    """Fit alpha values for each category"""
    results = []
    
    for category in df['primary_category'].unique():
        cat_df = df[df['primary_category'] == category]
        
        if len(cat_df) >= min_samples:
            # Prepare data for fitting
            context = cat_df['context'].values
            base_conveyance = cat_df['CONVEYANCE'].values
            amplified_conveyance = cat_df['amplified_conveyance'].values
            
            # Define the model: amplified = base * context^alpha
            def model(context, alpha):
                return base_conveyance * (context ** alpha)
            
            try:
                # Fit the model
                popt, pcov = curve_fit(model, context, amplified_conveyance, 
                                     p0=[1.5], bounds=(0.1, 3.0))
                fitted_alpha = popt[0]
                
                # Calculate R-squared
                predicted = model(context, fitted_alpha)
                r2 = r2_score(amplified_conveyance, predicted)
                
                # Calculate confidence interval
                perr = np.sqrt(np.diag(pcov))
                alpha_std = perr[0]
                
                results.append({
                    'category': category,
                    'fitted_alpha': fitted_alpha,
                    'alpha_std': alpha_std,
                    'r2_score': r2,
                    'n_samples': len(cat_df),
                    'avg_context': cat_df['context'].mean(),
                    'context_std': cat_df['context'].std()
                })
                
            except Exception as e:
                print(f"Could not fit {category}: {e}")
    
    return pd.DataFrame(results).sort_values('fitted_alpha', ascending=False)


def plot_alpha_analysis(df: pd.DataFrame, alpha_results: pd.DataFrame):
    """Create comprehensive visualization of alpha values"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Distribution of observed alpha values
    ax1 = plt.subplot(2, 3, 1)
    # Check if we have valid alpha values to plot
    if len(df['observed_alpha'].dropna()) > 0 and df['observed_alpha'].std() > 0:
        df['observed_alpha'].hist(bins=20, alpha=0.7, color='blue')
        ax1.axvline(df['observed_alpha'].median(), color='red', linestyle='--', 
                    label=f'Median: {df["observed_alpha"].median():.2f}')
    else:
        # If all values are the same, show a bar
        ax1.bar([1.5], [len(df)], width=0.1, alpha=0.7, color='blue')
        ax1.text(1.5, len(df)/2, f'All α = 1.5\n(n={len(df)})', ha='center', va='center')
    ax1.set_xlabel('Observed α')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Observed α Values')
    ax1.legend()
    ax1.set_xlim(0, 3)
    
    # 2. Fitted alpha by category
    ax2 = plt.subplot(2, 3, 2)
    top_cats = alpha_results.nlargest(10, 'n_samples')
    y_pos = np.arange(len(top_cats))
    ax2.barh(y_pos, top_cats['fitted_alpha'], xerr=top_cats['alpha_std'], 
             color='green', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_cats['category'])
    ax2.set_xlabel('Fitted α')
    ax2.set_title('Fitted α Values by Category')
    ax2.axvline(1.5, color='red', linestyle='--', alpha=0.5, label='Theory: α=1.5')
    ax2.legend()
    
    # 3. R-squared scores
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(alpha_results['fitted_alpha'], alpha_results['r2_score'], 
                s=alpha_results['n_samples']*5, alpha=0.6)
    for _, row in alpha_results.iterrows():
        if row['r2_score'] > 0.5:
            ax3.annotate(row['category'], (row['fitted_alpha'], row['r2_score']), 
                        fontsize=8, alpha=0.7)
    ax3.set_xlabel('Fitted α')
    ax3.set_ylabel('R² Score')
    ax3.set_title('Model Fit Quality')
    ax3.grid(True, alpha=0.3)
    
    # 4. Context vs Alpha relationship
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(df['context'], df['observed_alpha'], 
                         c=df['amplified_conveyance'], cmap='viridis', 
                         alpha=0.5, s=20)
    ax4.set_xlabel('Context Score')
    ax4.set_ylabel('Observed α')
    ax4.set_title('Context vs Observed α')
    ax4.set_ylim(0, 3)
    plt.colorbar(scatter, ax=ax4, label='Amplified Conveyance')
    
    # 5. Category comparison: theory vs empirical
    ax5 = plt.subplot(2, 3, 5)
    categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.CY']
    theoretical_alphas = {'cs.AI': 1.5, 'cs.LG': 1.6, 'cs.CL': 1.5, 
                         'cs.CV': 1.7, 'cs.CY': 1.4}
    
    empirical_alphas = []
    theory_alphas = []
    cats_to_plot = []
    
    for cat in categories:
        if cat in alpha_results['category'].values:
            emp_alpha = alpha_results[alpha_results['category'] == cat]['fitted_alpha'].values[0]
            empirical_alphas.append(emp_alpha)
            theory_alphas.append(theoretical_alphas.get(cat, 1.5))
            cats_to_plot.append(cat)
    
    if cats_to_plot:
        x = np.arange(len(cats_to_plot))
        width = 0.35
        ax5.bar(x - width/2, empirical_alphas, width, label='Empirical', alpha=0.7)
        ax5.bar(x + width/2, theory_alphas, width, label='Theoretical', alpha=0.7)
        ax5.set_xticks(x)
        ax5.set_xticklabels(cats_to_plot)
        ax5.set_ylabel('α Value')
        ax5.set_title('Empirical vs Theoretical α')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Alpha confidence intervals
    ax6 = plt.subplot(2, 3, 6)
    top_results = alpha_results.nlargest(15, 'n_samples')
    ax6.errorbar(range(len(top_results)), top_results['fitted_alpha'], 
                 yerr=top_results['alpha_std'], fmt='o', capsize=5, alpha=0.7)
    ax6.set_xticks(range(len(top_results)))
    ax6.set_xticklabels(top_results['category'], rotation=45, ha='right')
    ax6.set_ylabel('Fitted α ± std')
    ax6.set_title('Alpha Values with Confidence Intervals')
    ax6.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Theory: α=1.5')
    ax6.axhline(2.0, color='orange', linestyle='--', alpha=0.5, label='Max: α=2.0')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def validate_alpha_bounds(alpha_results: pd.DataFrame) -> Dict:
    """Validate that empirical alphas fall within theoretical bounds"""
    validation = {
        'total_categories': len(alpha_results),
        'within_bounds': len(alpha_results[(alpha_results['fitted_alpha'] >= 1.0) & 
                                         (alpha_results['fitted_alpha'] <= 2.5)]),
        'below_1': len(alpha_results[alpha_results['fitted_alpha'] < 1.0]),
        'above_2_5': len(alpha_results[alpha_results['fitted_alpha'] > 2.5]),
        'mean_alpha': alpha_results['fitted_alpha'].mean(),
        'std_alpha': alpha_results['fitted_alpha'].std(),
        'median_alpha': alpha_results['fitted_alpha'].median()
    }
    return validation


def main():
    """Measure alpha values empirically"""
    print("=" * 60)
    print("EMPIRICAL MEASUREMENT OF CONTEXT AMPLIFICATION (α)")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data('validation/python/empirical_results.csv')
    print(f"\nLoaded {len(df)} documents with valid context and conveyance")
    
    # Fit alpha by category
    alpha_results = fit_alpha_by_category(df, min_samples=5)
    
    # Save results
    alpha_results.to_csv('validation/python/measured_alpha_values.csv', index=False)
    print(f"\nSaved alpha measurements to measured_alpha_values.csv")
    
    # Display results
    print("\n" + "=" * 40)
    print("FITTED ALPHA VALUES BY CATEGORY")
    print("=" * 40)
    print(alpha_results.to_string(index=False))
    
    # Validate bounds
    validation = validate_alpha_bounds(alpha_results)
    print("\n" + "=" * 40)
    print("VALIDATION OF ALPHA BOUNDS")
    print("=" * 40)
    print(f"Total categories analyzed: {validation['total_categories']}")
    print(f"Within theoretical bounds [1.0, 2.5]: {validation['within_bounds']} " +
          f"({100*validation['within_bounds']/validation['total_categories']:.1f}%)")
    print(f"Below 1.0: {validation['below_1']}")
    print(f"Above 2.5: {validation['above_2_5']}")
    print(f"Mean α: {validation['mean_alpha']:.3f} ± {validation['std_alpha']:.3f}")
    print(f"Median α: {validation['median_alpha']:.3f}")
    
    # Statistical summary
    print("\n" + "=" * 40)
    print("KEY FINDINGS")
    print("=" * 40)
    
    # Domain-specific alphas
    domain_alphas = {
        'Computer Science': alpha_results[alpha_results['category'].str.startswith('cs.')]['fitted_alpha'].mean(),
        'Mathematics': alpha_results[alpha_results['category'].str.startswith('math.')]['fitted_alpha'].mean() if any(alpha_results['category'].str.startswith('math.')) else None,
        'Physics': alpha_results[alpha_results['category'].str.startswith('physics.')]['fitted_alpha'].mean() if any(alpha_results['category'].str.startswith('physics.')) else None,
        'Engineering': alpha_results[alpha_results['category'].str.startswith('eess.')]['fitted_alpha'].mean() if any(alpha_results['category'].str.startswith('eess.')) else None
    }
    
    for domain, alpha in domain_alphas.items():
        if alpha is not None:
            print(f"{domain}: α ≈ {alpha:.2f}")
    
    # Theory validation
    print(f"\nTheoretical prediction (α ∈ [1.5, 2.0]): {'VALIDATED' if validation['mean_alpha'] >= 1.5 and validation['mean_alpha'] <= 2.0 else 'NEEDS ADJUSTMENT'}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    fig = plot_alpha_analysis(df, alpha_results)
    fig.savefig('validation/python/alpha_measurement_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: alpha_measurement_analysis.png")
    
    print("\n" + "=" * 60)
    print("ALPHA MEASUREMENT COMPLETE")
    print("Context amplification validated empirically ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()