import os
import sys
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import GroupKFold, cross_validate

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots 

plt.style.use(['science'])

sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

DIRS = {
    'data': BASE_DIR / "data",
    'heatmaps': BASE_DIR / "images/heatmaps"
}

for directory in DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)


def create_pipeline_flags(df):
    flags = pd.DataFrame(index=df.index) 
    
    flags['SVM'] = (df['Model'] == 'SVM').astype(int)
    flags['LR'] = (df['Model'] == 'LR').astype(int)
    flags['KNN'] = (df['Model'] == 'KNN').astype(int)
    flags['MLP'] = (df['Model'] == 'MLP').astype(int)
    flags['GNB'] = (df['Model'] == 'GNB').astype(int)
    flags['RF'] = (df['Model'] == 'RF').astype(int)
    flags['XGB'] = (df['Model'] == 'XGB').astype(int)

    flags['Standard_Scaler'] = (df['Preprocessor'] == 'Standard Scaling').astype(int)
    flags['MinMax_Scaler'] = (df['Preprocessor'] == 'MinMax Scaling').astype(int)
    flags['IQR_Capping'] = (df['Preprocessor'] == 'IQR Capping').astype(int)
    flags['IQR_Removal'] = (df['Preprocessor'] == 'IQR Removal').astype(int)
    flags['Yeo_Johnson'] = (df['Preprocessor'] == 'Yeo-Johnson').astype(int)
    flags['Quantile_Transform'] = (df['Preprocessor'] == 'Quantile Transform').astype(int)
    flags['Uniform_Binning'] = (df['Preprocessor'] == 'Uniform Binning').astype(int)
    flags['Quantile_Binning'] = (df['Preprocessor'] == 'Quantile Binning').astype(int)
    flags['PCA'] = (df['Preprocessor'] == 'PCA').astype(int)
    flags['LDA'] = (df['Preprocessor'] == 'LDA').astype(int)
    flags['RUS'] = (df['Preprocessor'] == 'Random Undersampling').astype(int)
    flags['SMOTE'] = (df['Preprocessor'] == 'SMOTE').astype(int)
    flags['FS_ANOVA'] = (df['Preprocessor'] == 'Select Percentile (ANOVA)').astype(int)
    flags['FS_MI'] = (df['Preprocessor'] == 'Select Percentile (MI)').astype(int)

    return flags


def process_results_data(df):
    df = df[~df['Preprocessor'].str.contains(r'PCA \(Raw\)|SMOTE \(Raw\)', regex=True, na=False)].copy()
    df['Preprocessor'] = df['Preprocessor'].replace({'PCA (Scaled)': 'PCA', 'SMOTE (Scaled)': 'SMOTE'})

    df = df[df['Perspective'] == 'Equity'].copy()

    baseline_df = df[df['Preprocessor'] == 'Baseline'][['Dataset', 'Model', 'Test_MCC']]
    baseline_df = baseline_df.rename(columns={'Test_MCC': 'Baseline_MCC'})
    
    df = df.merge(baseline_df, on=['Dataset', 'Model'], how='inner')
    df['Delta_MCC'] = df['Test_MCC'] - df['Baseline_MCC']
    
    df = df.dropna(subset=['Delta_MCC']).copy()
    df = df[df['Preprocessor'] != 'Baseline'].copy()
    
    return df


def process_metadata(df):
    if 'dataset_name' in df.columns:
        df = df.rename(columns={'dataset_name': 'Dataset'})

    if 'features_clean' in df.columns and 'num_features_clean' in df.columns:
        df['prop_num_features'] = np.where(
            df['features_clean'] > 0, 
            df['num_features_clean'] / df['features_clean'], 
            0
        )

    columns_to_keep = [
        'Dataset', 'instances_clean', 'features_clean',
        'prop_num_features', 'prop_missing_clean', 'imbalance_ratio_clean'
    ]

    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns].copy()

    df.columns = df.columns.str.replace('_clean', '')
    
    rename_dict = {col: f'MF_{col}' for col in df.columns if col != 'Dataset'}
    df = df.rename(columns=rename_dict)
    
    return df


def plot_heatmap(matrix, save_path, ylabel, xlabel):
    fig, ax = plt.subplots(figsize=(7.1, 4.2)) 
    
    max_val = matrix.max().max()
    
    if pd.isna(max_val) or max_val == 0:
        max_val = 0.01
        
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".4f",
        cmap='Reds',
        vmin=0,
        vmax=max_val,
        cbar_kws={'label': r'Mean |SHAP Interaction|'},
        linewidths=0.5,
        linecolor='black',
        annot_kws={"size": 6.5},
        ax=ax
    )
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    ax.set_xticks(ax.get_xticks()) 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    
    save_path = Path(save_path)
    plt.savefig(save_path.with_suffix('.pdf'), format='pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)


def generate_shap_heatmap(X, shap_interactions, prep_cols, meta_cols):
    interaction_matrix = pd.DataFrame(index=prep_cols, columns=meta_cols)
    
    for prep in prep_cols:
        for meta in meta_cols:
            idx_p = X.columns.get_loc(prep)
            idx_m = X.columns.get_loc(meta)
            interaction_matrix.loc[prep, meta] = np.abs(shap_interactions[:, idx_p, idx_m]).mean()

    # alterado: Cria uma cópia da matriz apenas para o plot e remove o "MF_" das colunas
    plot_matrix = interaction_matrix.copy()
    plot_matrix.columns = [col.replace('MF_', '') for col in plot_matrix.columns]

    plot_heatmap(
        plot_matrix.astype(float), 
        DIRS['heatmaps'] / "shap_heatmap_preprocessors_vs_metafeatures", 
        ylabel="Preprocessor", 
        xlabel="Meta-Feature"
    )


def plot_top_metafeature_dynamics(X, shap_interactions, prep_cols, meta_name, save_dir, top_k=5):
    idx_m = X.columns.get_loc(meta_name)
    
    display_meta_name = meta_name.replace('MF_', '')
    
    prep_impacts = {}
    for prep_name in prep_cols:
        idx_p = X.columns.get_loc(prep_name)
        mask = X[prep_name] == 1 
        
        if mask.sum() > 0:
            mean_abs_impact = np.abs(shap_interactions[mask, idx_p, idx_m]).mean()
            prep_impacts[prep_name] = mean_abs_impact
            
    if not prep_impacts:
        return
        
    sorted_preps = sorted(prep_impacts.items(), key=lambda x: x[1], reverse=True)
    top_preprocessors = [item[0] for item in sorted_preps[:top_k]]
    
    plot_data = []
    for prep_name in top_preprocessors:
        idx_p = X.columns.get_loc(prep_name)
        mask = X[prep_name] == 1 
        
        x_vals = X.loc[mask, meta_name].values
        y_vals = shap_interactions[mask, idx_p, idx_m]
        
        for x, y in zip(x_vals, y_vals):
            plot_data.append({'Meta_Feature': x, 'SHAP_Interaction': y, 'Preprocessor': prep_name})
            
    df_plot = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    
    sns.lineplot(
        data=df_plot,
        x='Meta_Feature', 
        y='SHAP_Interaction', 
        hue='Preprocessor',
        marker='o',
        alpha=0.8, 
        linewidth=1.5,
        palette='Set1',
        ax=ax
    )
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1.2, zorder=1)
    
    ax.set_xlabel(f"{display_meta_name} (Original Value)")
    ax.set_ylabel(f"Impact on Delta MCC (SHAP Value)")
    ax.set_title(f"Top {top_k} Preprocessors Impact vs {display_meta_name}")
    
    plt.legend(title="Top Preprocessors", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    
    save_path = Path(save_dir) / f"lineplot_top{top_k}_{display_meta_name}.pdf"
    plt.savefig(save_path, format='pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)


def main():
    results_path = DIRS['data'] / 'aggregated_results.csv'
    meta_path = DIRS['data'] / 'metadata.csv'
    shap_path = DIRS['data'] / 'shap_results.pkl'

    results_df = pd.read_csv(results_path)
    results_df = process_results_data(results_df)

    meta_df = pd.read_csv(meta_path)
    meta_df = process_metadata(meta_df)

    merged_df = results_df.merge(meta_df, on='Dataset', how='inner')
    flags_df = create_pipeline_flags(merged_df)
    
    meta_cols = [col for col in merged_df.columns if col.startswith('MF_')]
    meta_df_only = merged_df[meta_cols]

    X = pd.concat([flags_df, meta_df_only], axis=1)
    y = merged_df['Delta_MCC']
    groups = merged_df['Dataset'].values

    model = xgb.XGBRegressor(
        tree_method="hist",
        random_state=42,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8
    )

    gkf = GroupKFold(n_splits=10)
    scoring = ['neg_root_mean_squared_error', 'r2']
    
    cv_results = cross_validate(model, X, y, groups=groups, cv=gkf, scoring=scoring)
    
    print("\n=== Cross-Validation Results (10-GroupKFold) ===")
    print(f"-> Mean RMSE: {-np.mean(cv_results['test_neg_root_mean_squared_error']):.4f} (+/- {np.std(-cv_results['test_neg_root_mean_squared_error']):.4f})")
    print(f"-> Mean R²:   {np.mean(cv_results['test_r2']):.4f} (+/- {np.std(cv_results['test_r2']):.4f})")
    print("================================================\n")

    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X)
    shap_interaction_values = explainer.shap_interaction_values(X)

    with open(shap_path, 'wb') as f:
        pickle.dump({
            'shap_explanation': shap_explanation,
            'shap_interaction_values': shap_interaction_values,
            'X': X,
            'y': y,
            'model': model
        }, f)
        
    algo_cols = ['GNB', 'LR', 'KNN', 'SVM', 'RF', 'MLP', 'XGB']
    prep_cols = [col for col in X.columns if not col.startswith('MF_') and col not in algo_cols]
    
    generate_shap_heatmap(X, shap_interaction_values, prep_cols, meta_cols)
    
    for meta in meta_cols:
        plot_top_metafeature_dynamics(
            X, 
            shap_interaction_values, 
            prep_cols, 
            meta, 
            DIRS['heatmaps'],
            top_k=3
        )
    
if __name__ == "__main__":
    main()