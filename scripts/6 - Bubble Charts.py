import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from pathlib import Path

plt.style.use(['science']) 

BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "images/bubble_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

def plot_bubble_chart(df, x_col, y_col, size_col, hue_col, x_label, size_label, save_name):
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    max_size_val = df[size_col].max()
    
    sns.scatterplot(
        data=df, x=x_col, y=y_col, size=size_col, hue=hue_col,
        sizes=(10, 80), size_norm=(0, max_size_val), alpha=0.65,      
        edgecolor='black', linewidth=0.4, ax=ax
    )

    ax.set_xscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel('MCC')
    
    h, l = ax.get_legend_handles_labels()
    for i, label in enumerate(l):
        if label == hue_col: l[i] = 'Algorithm'
        elif label == size_col: l[i] = size_label

    ax.legend(h, l, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)
    plt.savefig(CHART_DIR / f"{save_name}.pdf", format='pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)

def get_minmax_score(series, maximize=True):
    if series.max() == series.min():
        return pd.Series(1.0, index=series.index)
    
    res = (series - series.min()) / (series.max() - series.min())
    return res if maximize else 1 - res

def generate_qualis_table(df_plot):
    df_plot['Total_Time_sec'] = df_plot['Fit_Time_sec'] + df_plot['Predict_Time_Test_sec']
    records = []
    
    for (perspective, model), group in df_plot.groupby(['Perspective', 'Model']):
        group = group.reset_index(drop=True)
        
        score_mcc = get_minmax_score(group['Test_MCC'], maximize=True)
        score_time = get_minmax_score(group['Total_Time_sec'], maximize=False)
        score_mem = get_minmax_score(group['Peak_Memory_MB'], maximize=False)
        
        records.append({
            'Perspective': perspective,
            'Model': model,
            'Max MCC': group.loc[group['Test_MCC'].idxmax(), 'Preprocessor'],
            'Min Time': group.loc[group['Total_Time_sec'].idxmin(), 'Preprocessor'],
            'Min Memory': group.loc[group['Peak_Memory_MB'].idxmin(), 'Preprocessor'],
            'Best Trade-off': group.loc[(score_mcc + score_time + score_mem).idxmax(), 'Preprocessor']
        })
        
    df_table = pd.DataFrame(records)
    df_table.to_csv(CHART_DIR / "preprocessors_summary.csv", index=False)
    
    latex_code = df_table.style.hide(axis="index").to_latex(
        hrules=True, column_format="llcccc", 
        caption="Best preprocessor per algorithm across different evaluation metrics.",
        label="tab:preprocessor_summary"
    )
    
    with open(CHART_DIR / "preprocessors_summary.tex", 'w') as f:
        f.write(latex_code)

def main():
    df = pd.read_csv(BASE_DIR / 'data/aggregated_results.csv')
    df = df[~df['Preprocessor'].str.contains('Raw')].copy()
    df['Preprocessor'] = df['Preprocessor'].str.replace(rf' \(Scaled\)| Scaled', '', regex=True)

    aggs = ['Test_MCC', 'Fit_Time_sec', 'Predict_Time_Test_sec', 'Peak_Memory_MB', 'Pipeline_Disk_Size_MB']
    df_plot = df.groupby(['Perspective', 'Model', 'Preprocessor'])[aggs].median().reset_index()

    for persp in ['Equity', 'Equality']:
        df_sub = df_plot[df_plot['Perspective'] == persp]
        if not df_sub.empty:
            plot_bubble_chart(df_sub, 'Fit_Time_sec', 'Test_MCC', 'Peak_Memory_MB', 'Model',
                              "Fit Time (seconds) - Log Scale", "Peak Memory (MB)", f"bubble_study_{persp.lower()}")
            plot_bubble_chart(df_sub, 'Predict_Time_Test_sec', 'Test_MCC', 'Pipeline_Disk_Size_MB', 'Model',
                              "Predict Time (seconds) - Log Scale", "Disk Size (MB)", f"bubble_production_{persp.lower()}")

    generate_qualis_table(df_plot)

if __name__ == "__main__":
    main()