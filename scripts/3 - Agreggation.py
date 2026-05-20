import os
import glob
import pandas as pd


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_dir = os.path.abspath(os.path.join(script_dir, '../data/2 - results'))
    output_path = os.path.abspath(os.path.join(script_dir, '../data/aggregated_results.csv'))
    
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    results = [pd.read_csv(file) for file in csv_files]
    merged_results = pd.concat(results, ignore_index=True)

    grouping_keys = ['Dataset', 'Model', 'Preprocessor', 'Perspective']

    memory_columns = ['Peak_Memory_MB', 'Pipeline_Disk_Size_MB']
    memory_metrics = merged_results[merged_results['Fold'] == 1][grouping_keys + memory_columns].copy()

    time_columns = ['Fit_Time_sec', 'Predict_Time_Test_sec']
    time_metrics = merged_results[merged_results['Fold'] > 1].groupby(grouping_keys)[time_columns].mean().reset_index()

    performance_columns = [
        'Converged', 'Train_ROC_AUC', 'Train_PR_AUC', 'Train_MCC',
        'Test_ROC_AUC', 'Test_PR_AUC', 'Test_MCC'
    ]
    performance_metrics = merged_results.groupby(grouping_keys)[performance_columns].mean().reset_index()

    aggregated_results = pd.merge(performance_metrics, time_metrics, on=grouping_keys, how='left')
    aggregated_results = pd.merge(aggregated_results, memory_metrics, on=grouping_keys, how='left')

    final_column_order = grouping_keys + memory_columns + time_columns + performance_columns
    aggregated_results = aggregated_results[final_column_order]

    aggregated_results.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()