import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np


def load_eval_reports(results_dir):
    """Load all eval_*.json files from results directory."""
    reports = {}
    eval_files = glob.glob(os.path.join(results_dir, 'eval_*.json'))
    for fpath in sorted(eval_files):
        with open(fpath, 'r') as f:
            data = json.load(f)
            model_name = os.path.basename(fpath).replace('eval_', '').replace('.json', '')
            reports[model_name] = data
    return reports


def create_error_comparison(reports, output_dir):
    """Create subplots with one chart per error type, showing all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set EDA-style visualization
    plt.style.use('seaborn-v0_8-darkgrid')
    
    models = sorted(list(reports.keys()))
    error_types = ['false_negative', 'false_positive', 'poor_localization', 'misclassification']
    
    # prepare data: data[error_type] = [count for each model]
    data = {error_type: [] for error_type in error_types}
    for error_type in error_types:
        for model in models:
            summary = reports[model]['summary']
            data[error_type].append(summary.get(error_type, 0))
    
    # create subplots (2 rows x 2 columns for 4 charts)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # EDA-style color palette (professional seaborn colors)
    palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # plot each error type in a separate subplot
    for idx, error_type in enumerate(error_types):
        ax = axes[idx]
        x = np.arange(len(models))
        values = data[error_type]
        
        bars = ax.bar(x, values, color=palette[idx], alpha=0.85, edgecolor='#333', linewidth=1.5)
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(error_type.replace('_', ' ').title(), fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.7)
        ax.set_axisbelow(True)
        
        # add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Clean up spines for EDA style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
    
    fig.suptitle('Model Performance: Error Analysis', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = os.path.join(output_dir, 'error_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")



if __name__ == '__main__':
    results_dir = 'results'
    output_dir = 'assets'
    
    print("Loading evaluation reports...")
    reports = load_eval_reports(results_dir)
    
    if not reports:
        print(f"No evaluation reports found in {results_dir}")
    else:
        print(f"Found {len(reports)} evaluation report(s): {list(reports.keys())}")
        print("Generating error comparison chart...")
        create_error_comparison(reports, output_dir)
        print(f"✓ Chart saved to {output_dir}/error_comparison.png")

