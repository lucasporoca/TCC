import copy
import gc
import os
import pickle
import random
import time
import traceback
import tracemalloc
import warnings
from functools import partial

import numpy as np
import pandas as pd
from feature_engine.outliers import OutlierTrimmer, Winsorizer
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from xgboost import XGBClassifier

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler,
                                   PowerTransformer, QuantileTransformer,
                                   StandardScaler, TargetEncoder)
from sklearn.svm import SVC

sklearn.set_config(transform_output="pandas")
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class DropHighMissingFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.cols_to_keep_ = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.cols_to_keep_ = X_df.columns[X_df.isnull().mean() <= self.threshold].tolist()
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in self.cols_to_keep_:
            if col not in X_df.columns:
                X_df[col] = np.nan
        return X_df[self.cols_to_keep_]


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, tail='both', fold=1.5):
        self.tail = tail
        self.fold = fold
        self.capper_ = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
        valid_cols = []
        for c in X_df.columns:
            q1, q3 = X_df[c].quantile([0.25, 0.75])
            if q1 != q3:
                valid_cols.append(c)
        if valid_cols:
            self.capper_ = Winsorizer(
                capping_method='iqr', tail=self.tail, fold=self.fold, variables=valid_cols
            )
            self.capper_.fit(X_df)
        else:
            self.capper_ = None
        return self

    def transform(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
        if self.capper_ is not None:
            return self.capper_.transform(X_df)
        return X_df


class FiniteCapper(BaseEstimator, TransformerMixin):
    _FLOAT32_SAFE = float(np.finfo(np.float32).max) * 0.99

    def fit(self, X, y=None):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
        self.max_finite_ = np.zeros(X_arr.shape[1])
        self.min_finite_ = np.zeros(X_arr.shape[1])
        for i in range(X_arr.shape[1]):
            col_data = X_arr[:, i]
            finite_mask = np.isfinite(col_data)
            if finite_mask.any():
                self.max_finite_[i] = np.clip(np.max(col_data[finite_mask]), -self._FLOAT32_SAFE, self._FLOAT32_SAFE)
                self.min_finite_[i] = np.clip(np.min(col_data[finite_mask]), -self._FLOAT32_SAFE, self._FLOAT32_SAFE)
            else:
                self.max_finite_[i] = 0.0
                self.min_finite_[i] = 0.0
        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        X_arr = X.copy().to_numpy() if is_df else np.copy(X)
        for i in range(X_arr.shape[1]):
            X_arr[:, i] = np.where(np.isposinf(X_arr[:, i]), self.max_finite_[i], X_arr[:, i])
            X_arr[:, i] = np.where(np.isneginf(X_arr[:, i]), self.min_finite_[i], X_arr[:, i])
            X_arr[:, i] = np.nan_to_num(X_arr[:, i], nan=0.0, posinf=self.max_finite_[i], neginf=self.min_finite_[i])
        X_arr = np.clip(X_arr, -self._FLOAT32_SAFE, self._FLOAT32_SAFE)
        if is_df:
            return pd.DataFrame(X_arr, columns=X.columns, index=X.index)
        return X_arr


def outlier_trimmer(X, y, fold=1.5, min_samples=20):
    is_numpy = isinstance(X, np.ndarray)
    X_df = pd.DataFrame(X) if is_numpy else X
    if is_numpy and hasattr(y, 'index'):
        X_df.index = y.index
    valid_cols = []
    for c in X_df.columns:
        q1, q3 = X_df[c].quantile([0.25, 0.75])
        if q1 != q3:
            valid_cols.append(c)
    if not valid_cols:
        return X, y
    trimmer = OutlierTrimmer(capping_method='iqr', tail='both', fold=fold, variables=valid_cols)
    X_res_df = trimmer.fit_transform(X_df)
    if len(X_res_df) < min_samples:
        return X, y
    y_res = y.loc[X_res_df.index] if hasattr(y, 'loc') else y[X_res_df.index]
    if is_numpy:
        return X_res_df.values, y_res.values if isinstance(y_res, pd.Series) else y_res
    else:
        return X_res_df, y_res


def get_positive_scores(pipeline, X_data):
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_data)
        if probs.shape[1] == 1:
            return np.zeros(X_data.shape[0])
        return probs[:, 1]
    else:
        return pipeline.decision_function(X_data)


def build_pipeline(model_obj, prep_steps):
    model_cloned = clone(model_obj)
    
    num_pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    cat_pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value')),
        ('te', TargetEncoder(target_type='binary', cv=5, random_state=42))
    ])
    
    impute_and_encode = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, make_column_selector(pattern='^NUM_')),
            ('cat', cat_pipeline, make_column_selector(pattern='^CAT_'))
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    steps = [
        ('drop_nans', DropHighMissingFeatures(threshold=0.5)),
        ('impute_and_encode', impute_and_encode)
    ]

    if prep_steps:
        for name, step in prep_steps:
            steps.append((name, clone(step)))

    steps.append(('fix_infs', FiniteCapper()))
    steps.append(('classifier', model_cloned))

    return ImbPipeline(steps)


def run_single_fold(fold_idx, train_idx, test_idx, X, y, dataset_name,
                    m_name, model_obj, p_name, perspective, prep_steps, models_with_convergence):
    
    np.random.seed(42 + fold_idx)
    random.seed(42 + fold_idx)
    
    try:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        pipeline = build_pipeline(model_obj, prep_steps)

        model_converged = True if m_name in models_with_convergence else np.nan
        measure_resources = (fold_idx == 1)
        
        if measure_resources:
            tracemalloc.start()

        t0_fit = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(X_train, y_train)
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    if m_name in models_with_convergence:
                        model_converged = False
        fit_time = time.perf_counter() - t0_fit

        if measure_resources:
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            ram_used_mb = peak_mem / (1024 * 1024)
            disk_size = len(pickle.dumps(pipeline)) / (1024 * 1024)
        else:
            ram_used_mb = np.nan
            disk_size = np.nan

        y_train_scores = get_positive_scores(pipeline, X_train)
        y_train_pred = (y_train_scores >= 0.5).astype(int) if hasattr(pipeline, "predict_proba") else (y_train_scores >= 0).astype(int)

        tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_train_pred, labels=[0, 1]).ravel()
        roc_tr = roc_auc_score(y_train, y_train_scores)
        pr_auc_tr = average_precision_score(y_train, y_train_scores)
        mcc_tr = matthews_corrcoef(y_train, y_train_pred)

        t0_pred_test = time.perf_counter()
        y_test_scores = get_positive_scores(pipeline, X_test)
        y_test_pred = (y_test_scores >= 0.5).astype(int) if hasattr(pipeline, "predict_proba") else (y_test_scores >= 0).astype(int)
        pred_time_test = time.perf_counter() - t0_pred_test

        tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()
        roc_te = roc_auc_score(y_test, y_test_scores)
        pr_auc_te = average_precision_score(y_test, y_test_scores)
        mcc_te = matthews_corrcoef(y_test, y_test_pred)

        return {
            'Dataset': dataset_name, 'Model': m_name, 'Preprocessor': p_name, 'Perspective': perspective, 'Fold': fold_idx,
            'Converged': model_converged, 'Fit_Time_sec': fit_time, 'Predict_Time_Test_sec': pred_time_test,
            'Peak_Memory_MB': ram_used_mb, 'Pipeline_Disk_Size_MB': disk_size,
            'Train_TP': tp_tr, 'Train_TN': tn_tr, 'Train_FP': fp_tr, 'Train_FN': fn_tr,
            'Train_ROC_AUC': roc_tr, 'Train_PR_AUC': pr_auc_tr, 'Train_MCC': mcc_tr,
            'Test_TP': tp_te, 'Test_TN': tn_te, 'Test_FP': fp_te, 'Test_FN': fn_te,
            'Test_ROC_AUC': roc_te, 'Test_PR_AUC': pr_auc_te, 'Test_MCC': mcc_te,
            'Error_Log': None
        }

    except Exception:
        return {
            'Dataset': dataset_name, 'Model': m_name, 'Preprocessor': p_name, 'Perspective': perspective, 'Fold': fold_idx,
            'Converged': np.nan, 'Fit_Time_sec': np.nan, 'Predict_Time_Test_sec': np.nan,
            'Peak_Memory_MB': np.nan, 'Pipeline_Disk_Size_MB': np.nan,
            'Train_TP': np.nan, 'Train_TN': np.nan, 'Train_FP': np.nan, 'Train_FN': np.nan,
            'Train_ROC_AUC': np.nan, 'Train_PR_AUC': np.nan, 'Train_MCC': np.nan,
            'Test_TP': np.nan, 'Test_TN': np.nan, 'Test_FP': np.nan, 'Test_FN': np.nan,
            'Test_ROC_AUC': np.nan, 'Test_PR_AUC': np.nan, 'Test_MCC': np.nan,
            'Error_Log': traceback.format_exc()
        }


def run_experiment(file_path, output_dir, n_jobs=-1):
    dataset_name = os.path.basename(file_path).replace('.csv', '')
    output_path = os.path.join(output_dir, f"{dataset_name}_results.csv")

    if os.path.exists(output_path):
        tqdm.write(f"File {os.path.basename(output_path)} already exists. Skipping...")
        return False

    df = pd.read_csv(file_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    cat_cols = [c for c in df.columns if c.startswith('CAT_')]
    for col in cat_cols:
        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else np.nan)

    X, y_raw = df.drop(columns=['target']), df['target']
    minority_class = y_raw.value_counts().idxmin()
    y = (y_raw == minority_class).astype(int)
    y.index = X.index

    models = {
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=1),
        'KNN': KNeighborsClassifier(n_jobs=1),
        'SVM': SVC(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
        'MLP': MLPClassifier(random_state=42, max_iter=1000),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1)
    }
    models_with_convergence = ['LogisticRegression', 'SVM', 'MLP']

    preprocessors_dict = {
        'Baseline': [],
        'Standard Scaling': [('scaler', StandardScaler())],
        'MinMax Scaling': [('scaler', MinMaxScaler())],
        'IQR Capping': [('capper', OutlierCapper(tail='both', fold=1.5))],
        'IQR Removal': [('trimmer', FunctionSampler(func=outlier_trimmer, kw_args={'fold': 1.5}, validate=False))],
        'Yeo-Johnson': [('yj', PowerTransformer(method='yeo-johnson', standardize=False))],
        'Quantile Transform': [('qt', QuantileTransformer(output_distribution='normal', random_state=42))],
        'Uniform Binning': [('bin', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'))],
        'Quantile Binning': [('bin', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf'))],
        'PCA (Raw)': [('pca', PCA(n_components=0.95, random_state=42, whiten=False))],
        'PCA (Scaled)': [('scaler', StandardScaler()), ('pca', PCA(n_components=0.95, random_state=42, whiten=False))],
        'LDA': [('lda', LinearDiscriminantAnalysis())],
        'Random Undersampling': [('rus', RandomUnderSampler(random_state=42))],
        'SMOTE (Raw)': [('smote', SMOTE(random_state=42))],
        'SMOTE (Scaled)': [('scaler', StandardScaler()), ('smote', SMOTE(random_state=42))],
        'Select Percentile (ANOVA)': [('sel', SelectPercentile(score_func=f_classif, percentile=50))],
        'Select Percentile (MI)': [('sel', SelectPercentile(score_func=partial(mutual_info_classif, random_state=42), percentile=50))]
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    pbar = tqdm(total=len(models) * len(preprocessors_dict) * 2, desc=dataset_name, leave=False, position=1)

    def eval_pipeline(p_name, perspective, prep_steps, current_model_name, current_model_obj):
        gc.collect()
        folds = list(enumerate(cv.split(X, y), start=1))

        fold_isolated = Parallel(n_jobs=1, backend='sequential')(
            delayed(run_single_fold)(
                fold_idx, train_idx, test_idx, X, y, dataset_name,
                current_model_name, current_model_obj, p_name, perspective, prep_steps, models_with_convergence
            ) for fold_idx, (train_idx, test_idx) in folds[:1]
        )

        if len(folds) > 1:
            fold_rest = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(run_single_fold)(
                    fold_idx, train_idx, test_idx, X, y, dataset_name,
                    current_model_name, current_model_obj, p_name, perspective, prep_steps, models_with_convergence
                ) for fold_idx, (train_idx, test_idx) in folds[1:]
            )
        else:
            fold_rest = []

        return fold_isolated + fold_rest

    for m_name, model_obj in models.items():
        for base_p_name, base_prep_steps in preprocessors_dict.items():
            
            base_results = eval_pipeline(base_p_name, "Equality", base_prep_steps, m_name, model_obj)
            results.extend(base_results)
            pbar.update(1)
            
            if m_name in ['SVM', 'LogisticRegression', 'KNN', 'MLP']:
                custom_step = [('custom_scaler', StandardScaler())]
            elif m_name == 'GaussianNB':
                custom_step = [('custom_qt', QuantileTransformer(output_distribution='normal', random_state=42))]
            else:
                custom_step = []
                
            is_redundant = False
            if custom_step:
                custom_transformer_class = type(custom_step[0][1])
                is_redundant = any(isinstance(step_obj, custom_transformer_class) for _, step_obj in base_prep_steps)

            if not custom_step or is_redundant:
                custom_results = copy.deepcopy(base_results)
                for res in custom_results:
                    res['Perspective'] = "Equity"
                results.extend(custom_results)
                pbar.update(1)
                
            else:
                custom_prep_steps = base_prep_steps + custom_step
                custom_results = eval_pipeline(base_p_name, "Equity", custom_prep_steps, m_name, model_obj)
                results.extend(custom_results)
                pbar.update(1)

    pbar.close()
    pd.DataFrame(results).to_csv(output_path, index=False)
    return True

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, '../data/1 - datasets'))
    output_dir = os.path.abspath(os.path.join(script_dir, '../data/2 - results'))

    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    pbar_overall = tqdm(total=len(csv_files), desc="Overall Progress", position=0, leave=True)

    for file in csv_files:
        full_path = os.path.join(input_dir, file)
        exec_status = run_experiment(full_path, output_dir, n_jobs=-1)
        pbar_overall.update(1)
        if not exec_status:
            pbar_overall.refresh()

    pbar_overall.close()