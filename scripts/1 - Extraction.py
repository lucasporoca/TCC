import os
import random
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

random.seed(42)
np.random.seed(42)

def recreate_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)


def calculate_dataset_metadata(df):
    instances = len(df)
    features  = len(df.columns) - 1

    X = df.drop(columns=['target'])

    total_missing = int(X.isna().sum().sum())

    target_counts = df['target'].value_counts()

    return {
        'instances': instances,
        'features': features,
        'prop_missing': round(total_missing / (instances * features), 4) if features > 0 else 0,
        'imbalance_ratio': round(target_counts.max() / target_counts.min(), 4),
    }


def clean_dataset(df):
    sampled_df = df.copy()

    nunique_counts = sampled_df.nunique(dropna=True)
    valid_cols = [
        col for col in sampled_df.columns
        if nunique_counts[col] > 1 or col == 'target'
    ]

    return sampled_df[valid_cols]


def rename_dataset_columns(df, is_categorical_map):
    num_count = cat_count = 0
    rename_dict = {}

    for col in df.columns:
        if col == 'target':
            continue
        if is_categorical_map.get(col, False):
            rename_dict[col] = f"CAT_{cat_count:02d}"
            cat_count += 1
        else:
            rename_dict[col] = f"NUM_{num_count:02d}"
            num_count += 1

    return df.rename(columns=rename_dict)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    data_dir = os.path.join(project_dir, 'data')
    datasets_dir = os.path.join(data_dir, '1 - datasets')

    os.makedirs(data_dir, exist_ok=True)
    recreate_directory(datasets_dir)

    study_suite = openml.study.get_suite('OpenML-CC18')
    datasets_metadata_df = openml.datasets.list_datasets(
        data_id=study_suite.data, output_format='dataframe'
    )

    binary_mask = datasets_metadata_df['NumberOfClasses'] == 2
    binary_datasets = datasets_metadata_df[binary_mask].copy()

    metadata_records = []

    for _, row in tqdm(
        binary_datasets.iterrows(),
        total=len(binary_datasets),
        desc="Extracting Datasets"
    ):
        dataset_id   = row['did']
        dataset_name = row['name']

        dataset = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute,
        )

        is_categorical_map = dict(zip(X.columns, categorical_indicator))

        raw_df = X.copy()
        le = LabelEncoder()
        raw_df['target'] = le.fit_transform(y)

        meta_raw = calculate_dataset_metadata(raw_df)

        processed_df = clean_dataset(raw_df)
        processed_df = rename_dataset_columns(processed_df, is_categorical_map)

        meta_clean = calculate_dataset_metadata(processed_df)

        metadata_records.append({
            'dataset_id': dataset_id,
            'dataset_name': dataset_name,

            'instances_raw': meta_raw['instances'],
            'features_raw': meta_raw['features'],
            'imbalance_ratio_raw': meta_raw['imbalance_ratio'],

            'instances_clean': meta_clean['instances'],
            'features_clean': meta_clean['features'],
            'num_features_clean': len([c for c in processed_df.columns if c.startswith('NUM_')]),
            'cat_features_clean': len([c for c in processed_df.columns if c.startswith('CAT_')]),
            'prop_missing_clean': meta_clean['prop_missing'],
            'imbalance_ratio_clean': meta_clean['imbalance_ratio'],
            'minority_count_clean': int(processed_df['target'].value_counts().min()),
        })

        processed_df.to_csv(
            os.path.join(datasets_dir, f"{dataset_name}.csv"),
            index=False,
        )

    pd.DataFrame(metadata_records).to_csv(
        os.path.join(data_dir, 'metadata.csv'),
        index=False,
    )


if __name__ == "__main__":
    main()