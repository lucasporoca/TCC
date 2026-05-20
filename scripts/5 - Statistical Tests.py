import re
import itertools
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots 
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

plt.style.use(['science'])

BASE_DIR = Path(__file__).resolve().parent.parent

DIRS = {
    'ranks': BASE_DIR / "data/3 - tests/ranks",
    'pvalues': BASE_DIR / "data/3 - tests/pvalues",
    'images_models': BASE_DIR / "images/cd_plots/models",
    'images_preprocessors': BASE_DIR / "images/cd_plots/preprocessors"
}
TARGET_METRIC = "Test_MCC"

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

def clean_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

def wilcoxon_bh_test(df_matrix):
    cols = df_matrix.columns
    comparisons = list(itertools.combinations(cols, 2))
    
    p_unadj = [
        1.0 if np.allclose(df_matrix[c1], df_matrix[c2]) 
        else wilcoxon(df_matrix[c1], df_matrix[c2], zero_method='pratt').pvalue
        for c1, c2 in comparisons
    ]
            
    _, p_adj, _, _ = multipletests(p_unadj, alpha=0.05, method='fdr_bh')
    
    p_matrix = pd.DataFrame(1.0, index=cols, columns=cols)
    for (c1, c2), p in zip(comparisons, p_adj):
        p_matrix.loc[c1, c2] = p_matrix.loc[c2, c1] = p
        
    return p_matrix

def plot_cd_diagram(ranks, p_values, save_path):
    fig = plt.figure(figsize=(3.3, max(2.5, len(ranks) * 0.4)))
    
    props = {
        'marker': 'o', 
        'markersize': 4, 
        'markerfacecolor': 'white', 
        'markeredgecolor': 'black', 
        'color': 'black', 
        'linewidth': 1.0, 
        'zorder': 3
    }
    
    sp.critical_difference_diagram(
        ranks, p_values, 
        crossbar_props=props,
        text_h_margin=0.2
    )
    
    save_path = Path(save_path)
    
    plt.savefig(
        save_path.with_suffix('.pdf'), 
        format='pdf', 
        transparent=True, 
        bbox_inches='tight'
    )
    
    plt.close(fig)

def evaluate_group(df_group, perspective, fixed_val, compare_col):
    test_matrix = df_group.pivot(index='Dataset', columns=compare_col, values=TARGET_METRIC)
        
    p_values = wilcoxon_bh_test(test_matrix)
    ranks = test_matrix.rank(axis=1, ascending=False).mean()
    
    prefix = f"{clean_name(perspective)}_{clean_name(fixed_val)}_{clean_name(compare_col)}"
    
    p_values.to_csv(DIRS['pvalues'] / f"{prefix}_pvalues.csv")
    ranks.to_csv(DIRS['ranks'] / f"{prefix}_ranks.csv", header=['Rank'])
    
    if compare_col == 'Model':
        save_dir = DIRS['images_models']
    else:
        save_dir = DIRS['images_preprocessors']
        
    plot_cd_diagram(
        ranks=ranks, 
        p_values=p_values, 
        save_path=save_dir / f"{prefix}_cdplot"
    )

def main():
    df = pd.read_csv(BASE_DIR / 'data/aggregated_results.csv')
    df = df[~df['Preprocessor'].str.contains(r'PCA \(Raw\)|SMOTE \(Raw\)', regex=True, na=False)].copy()
    df['Preprocessor'] = df['Preprocessor'].replace({'PCA (Scaled)': 'PCA', 'SMOTE (Scaled)': 'SMOTE'})

    for (persp, prep), group in df.groupby(['Perspective', 'Preprocessor']):
        evaluate_group(group, persp, prep, compare_col='Model')

    for (persp, model), group in df.groupby(['Perspective', 'Model']):
        evaluate_group(group, persp, model, compare_col='Preprocessor')

if __name__ == "__main__": 
    main()