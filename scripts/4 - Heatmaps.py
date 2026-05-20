import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots 

plt.style.use(['science'])

BASE_DIR = Path(__file__).resolve().parent.parent

DIRS = {
    'heatmaps': BASE_DIR / "images/heatmaps"
}
TARGET_METRIC = "Test_MCC"
DISPLAY_METRIC = "MCC"
BASELINE_PREP = "Baseline"

MODELS_ORDER = ['GNB', 'LR', 'KNN', 'SVM', 'RF', 'MLP', 'XGB']

PREPROCESSORS_ORDER = [
    'Baseline', 'Standard Scaling', 'MinMax Scaling', 'IQR Capping', 
    'IQR Removal', 'Yeo-Johnson', 'Quantile Transform', 'Uniform Binning', 
    'Quantile Binning', 'PCA', 'LDA', 'Random Undersampling', 'SMOTE', 
    'Select Percentile (ANOVA)', 'Select Percentile (MI)'
]

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

def plot_heatmap(df_matrix, save_path, cmap='RdBu'):
    fig, ax = plt.subplots(figsize=(7.1, 3.8)) 
    
    max_val = df_matrix.abs().max().max()
    if pd.isna(max_val) or max_val == 0:
        max_val = 0.1
        
    sns.heatmap(
        df_matrix, 
        annot=True, 
        fmt=".3f",
        cmap=cmap, 
        center=0,
        vmin=-max_val,
        vmax=max_val,
        cbar_kws={'label': rf'$\Delta$ {DISPLAY_METRIC}'},
        linewidths=0.5,
        linecolor='black',
        annot_kws={"size": 6.5},
        ax=ax
    )
    
    ax.set_ylabel("Model")
    ax.set_xlabel("Preprocessor")
    
    ax.set_xticks(ax.get_xticks()) 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    
    save_path = Path(save_path)
    plt.savefig(save_path.with_suffix('.pdf'), format='pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)

def main():
    df = pd.read_csv(BASE_DIR / 'data/aggregated_results.csv')
    df = df[~df['Preprocessor'].str.contains(r'PCA \(Raw\)|SMOTE \(Raw\)', regex=True, na=False)].copy()
    df['Preprocessor'] = df['Preprocessor'].replace({'PCA (Scaled)': 'PCA', 'SMOTE (Scaled)': 'SMOTE'})

    models_in_df = [m for m in MODELS_ORDER if m in df['Model'].unique()]
    preps_in_df = [p for p in PREPROCESSORS_ORDER if p in df['Preprocessor'].unique()]
    
    df['Model'] = pd.Categorical(df['Model'], categories=models_in_df, ordered=True)
    df['Preprocessor'] = pd.Categorical(df['Preprocessor'], categories=preps_in_df, ordered=True)

    df_median = df.groupby(['Perspective', 'Model', 'Preprocessor'], observed=True)[TARGET_METRIC].median().reset_index()

    df_eq = df_median[df_median['Perspective'] == 'Equality'].pivot(index='Model', columns='Preprocessor', values=TARGET_METRIC)
    df_eqt = df_median[df_median['Perspective'] == 'Equity'].pivot(index='Model', columns='Preprocessor', values=TARGET_METRIC)

    diff_columns = [p for p in preps_in_df if p != BASELINE_PREP]

    if BASELINE_PREP in df_eq.columns:
        diff_eq = df_eq.sub(df_eq[BASELINE_PREP], axis=0).drop(columns=[BASELINE_PREP])
        diff_eq = diff_eq.reindex(index=models_in_df, columns=diff_columns)
        plot_heatmap(diff_eq, DIRS['heatmaps'] / "heatmap_equality_vs_baseline")

    if BASELINE_PREP in df_eqt.columns:
        diff_eqt = df_eqt.sub(df_eqt[BASELINE_PREP], axis=0).drop(columns=[BASELINE_PREP])
        diff_eqt = diff_eqt.reindex(index=models_in_df, columns=diff_columns)
        plot_heatmap(diff_eqt, DIRS['heatmaps'] / "heatmap_equity_vs_baseline")

    diff_perspectives = df_eqt - df_eq
    diff_perspectives = diff_perspectives.reindex(index=models_in_df, columns=preps_in_df)
    plot_heatmap(diff_perspectives, DIRS['heatmaps'] / "heatmap_equity_vs_equality", cmap='PRGn')

if __name__ == "__main__": 
    main()