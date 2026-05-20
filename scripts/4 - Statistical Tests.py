import os
import re
import itertools
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, norm
import scikit_posthocs as sp
import matplotlib.pyplot as plt


def friedman_finner_test(df_matrix):
    k = df_matrix.shape[1]
    n = df_matrix.shape[0]
    
    stat, p_friedman = friedmanchisquare(*[df_matrix.iloc[:, i].values for i in range(k)])
    
    ranks = df_matrix.rank(axis=1, ascending=False).mean()
    
    m = k * (k - 1) / 2.0
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    comparisons = list(itertools.combinations(df_matrix.columns, 2))
    
    p_unadj_list = []
    for c1, c2 in comparisons:
        z = abs(ranks[c1] - ranks[c2]) / se
        p_unadj = 2 * (1 - norm.cdf(z))
        p_unadj_list.append((c1, c2, p_unadj))
        
    p_unadj_list.sort(key=lambda x: x[2])
    
    p_adj_dict = {}
    running_max = 0.0
    for j, (c1, c2, p_val) in enumerate(p_unadj_list, start=1):
        adj_p = 1.0 - (1.0 - p_val)**(m / j)
        running_max = max(running_max, adj_p) 
        final_p = min(1.0, running_max)
        
        p_adj_dict[(c1, c2)] = final_p
        p_adj_dict[(c2, c1)] = final_p
        
    finner_matrix = pd.DataFrame(np.ones((k, k)), index=df_matrix.columns, columns=df_matrix.columns)
    for c1, c2 in comparisons:
        finner_matrix.loc[c1, c2] = p_adj_dict[(c1, c2)]
        finner_matrix.loc[c2, c1] = p_adj_dict[(c1, c2)]
        
    return p_friedman, stat, finner_matrix, ranks


def clean_name(name):
    name = name.lower()
    name = re.sub(r"[\[\]\(\)]", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    output_dir_tests = os.path.abspath(os.path.join(script_dir, "../data/3 - tests/"))
    output_dir_ranks = os.path.join(output_dir_tests, "ranks/")
    output_dir_pvalues = os.path.join(output_dir_tests, "pvalues/")
    output_dir_images = os.path.abspath(os.path.join(script_dir, "../images/cd_plots"))

    os.makedirs(output_dir_ranks, exist_ok=True)
    os.makedirs(output_dir_pvalues, exist_ok=True)
    os.makedirs(output_dir_images, exist_ok=True)

    input_data_path = os.path.abspath(os.path.join(script_dir, '../data/aggregated_results.csv'))
    
    results_df = pd.read_csv(input_data_path)

    results_df = results_df[~results_df['Preprocessor'].str.contains('PCA (Raw)', regex=False)]
    results_df = results_df[~results_df['Preprocessor'].str.contains('SMOTE (Raw)', regex=False)]
    results_df['Preprocessor'] = results_df['Preprocessor'].str.replace('PCA (Scaled)', 'PCA', regex=False)
    results_df['Preprocessor'] = results_df['Preprocessor'].str.replace('SMOTE (Scaled)', 'SMOTE', regex=False)

    target_metric = "Test_MCC"
    perspectives = ["Equality", "Equity"]
    models = results_df['Model'].unique()
    
    base_preps = results_df['Preprocessor'].unique()

    friedman_summary = []

    for persp in perspectives:
        for base_prep in base_preps:
            
            mask = (results_df['Perspective'] == persp) & (results_df['Preprocessor'] == base_prep)
            test_df = results_df[mask].copy()
            if test_df.empty: continue
                
            test_matrix = test_df.pivot(index='Dataset', columns='Model', values=target_metric).dropna(axis=1)
            if test_matrix.shape[1] < 2: continue
                
            p_friedman, stat, p_values, ranks = friedman_finner_test(test_matrix)
            
            friedman_summary.append({
                'Perspective': persp, 'Comparison': 'Algorithms',
                'Fixed_Factor': base_prep, 'Friedman_Statistic': stat, 'Friedman_P_Value': p_friedman
            })
            
            prefix = f"{persp.lower()}_{clean_name(base_prep)}_algorithms"
            
            ranks.to_csv(os.path.join(output_dir_ranks, f"{prefix}_ranks.csv"), header=['Rank'])
            p_values.to_csv(os.path.join(output_dir_pvalues, f"{prefix}_pvalues.csv"))
            
            plt.figure(figsize=(10, 4), dpi=120)
            sp.critical_difference_diagram(ranks, p_values)
            plt.title(f"Algoritmos: {base_prep} ({persp})", pad=20)
            plt.savefig(os.path.join(output_dir_images, f"{prefix}_cdplot.png"), bbox_inches='tight')
            plt.close()

    for persp in perspectives:
        for model_name in models:
            
            mask = (results_df['Perspective'] == persp) & (results_df['Model'] == model_name)
            test_df = results_df[mask].copy()
            if test_df.empty: continue
                
            test_matrix = test_df.pivot(index='Dataset', columns='Preprocessor', values=target_metric).dropna(axis=1)
            if test_matrix.shape[1] < 2: continue
            
            p_friedman, stat, p_values, ranks = friedman_finner_test(test_matrix)
            
            friedman_summary.append({
                'Perspective': persp, 'Comparison': 'Preprocessors',
                'Fixed_Factor': model_name, 'Friedman_Statistic': stat, 'Friedman_P_Value': p_friedman
            })
            
            prefix = f"{persp.lower()}_{clean_name(model_name)}_preprocessors"
            
            ranks.to_csv(os.path.join(output_dir_ranks, f"{prefix}_ranks.csv"), header=['Rank'])
            p_values.to_csv(os.path.join(output_dir_pvalues, f"{prefix}_pvalues.csv"))
            
            plt.figure(figsize=(12, 6), dpi=120)
            sp.critical_difference_diagram(ranks, p_values)
            plt.title(f"Preprocessadores: {model_name} ({persp})", pad=20)
            plt.savefig(os.path.join(output_dir_images, f"{prefix}_cdplot.png"), bbox_inches='tight')
            plt.close()

    summary_df = pd.DataFrame(friedman_summary)
    summary_df.to_csv(os.path.join(output_dir_tests, "friedman_summary_results.csv"), index=False)


if __name__ == "__main__":
    main()