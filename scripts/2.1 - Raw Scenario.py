import gc
import os
import pickle
import time
import traceback
import tracemalloc
import warnings

import msvcrt
import winsound

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from feature_engine.outliers import OutlierTrimmer, Winsorizer
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (KBinsDiscretizer, LabelEncoder,
                                   MinMaxScaler, OneHotEncoder,
                                   PowerTransformer, QuantileTransformer,
                                   StandardScaler)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from functools import partial

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def ask_to_continue_with_timeout(timeout_seconds):
    tqdm.write(f"\nDone. Press 'y' to continue, 'n' to stop, or wait {timeout_seconds}s...")
    start_time = time.time()
    last_beep_time = 0
    while True:
        current_time = time.time()
        if current_time - last_beep_time >= 1:
            winsound.Beep(1000, 300) 
            last_beep_time = current_time
        if msvcrt.kbhit():
            char = msvcrt.getch().decode('utf-8', 'ignore').lower()
            if char == 'y':
                tqdm.write("Continuing...")
                return True
            elif char == 'n':
                tqdm.write("Stopped.")
                return False
        if current_time - start_time > timeout_seconds:
            tqdm.write("Timeout. Continuing...")
            return True
        time.sleep(0.1)

def optimize_threshold_mcc(y_true, y_scores):
    thresholds = np.unique(np.percentile(y_scores, np.linspace(1, 99, 99)))
    
    if len(thresholds) == 0:
        return 0.5, matthews_corrcoef(y_true, (y_scores >= 0.5).astype(int))

    best_thresh = thresholds[len(thresholds) // 2]
    best_mcc = -1.0
    
    for t in thresholds:
        y_pred_t = (y_scores >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_t)
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
                    
    return best_thresh, best_mcc

def get_positive_scores(pipeline, X_data):
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_data)
        if probs.shape[1] == 1:
            return np.zeros(X_data.shape[0])
        return probs[:, 1]
    else:
        return pipeline.decision_function(X_data)

def outlier_trimmer(X, y, method, fold, cols):
    valid_cols = []
    for c in cols:
        if c not in X.columns:
            continue
            
        if method == 'iqr':
            q1, q3 = X[c].quantile([0.25, 0.75])
            if q1 != q3:
                valid_cols.append(c)
        elif method == 'quantiles':
            q_low, q_high = X[c].quantile([fold, 1 - fold])
            if q_low != q_high:
                valid_cols.append(c)
                
    if not valid_cols:
        return X, y

    trimmer = OutlierTrimmer(capping_method=method, tail='both', fold=fold, variables=valid_cols)
    X_res = trimmer.fit_transform(X)
    
    if len(X_res) <= len(X) * 0.80:
        return X, y
        
    return X_res, y.loc[X_res.index]

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, capping_method='iqr', tail='both', fold=1.5):
        self.capping_method = capping_method
        self.tail = tail
        self.fold = fold
        self.capper_ = None
        
    def fit(self, X, y=None):
        valid_cols = []
        for c in X.columns:
            if self.capping_method == 'iqr':
                q1, q3 = X[c].quantile([0.25, 0.75])
                if q1 != q3:
                    valid_cols.append(c)
            elif self.capping_method == 'quantiles':
                q_low, q_high = X[c].quantile([self.fold, 1 - self.fold])
                if q_low != q_high:
                    valid_cols.append(c)
                    
        if valid_cols:
            self.capper_ = Winsorizer(
                capping_method=self.capping_method, 
                tail=self.tail, 
                fold=self.fold, 
                variables=valid_cols
            )
            self.capper_.fit(X)
        else:
            self.capper_ = None
            
        return self

    def transform(self, X):
        if self.capper_ is not None:
            return self.capper_.transform(X)
        return X

def build_pipeline(model_obj, prep_obj, num_cols, cat_cols):
    model_cloned = clone(model_obj)
    prep_cloned = prep_obj if isinstance(prep_obj, str) else clone(prep_obj)

    sampler = None
    selector = None
    
    if isinstance(prep_cloned, (SMOTE, SMOTENC, RandomUnderSampler, FunctionSampler)):
        sampler = prep_cloned
        num_transformer = 'passthrough'
        
    elif isinstance(prep_cloned, SelectPercentile):
        selector = prep_cloned
        num_transformer = 'passthrough'
        
    else:
        num_transformer = prep_cloned 

    transformers = [('num', num_transformer, num_cols)]
    
    if cat_cols:
        cat_pipe = ImbPipeline([
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
        ])
        transformers.append(('cat', cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )

    steps = []
    
    if sampler:
        steps.append(('sampler', sampler))
        
    steps.append(('preprocessor', preprocessor))
    
    if selector:
        steps.append(('selector', selector))
        
    steps.append(('classifier', model_cloned))

    return ImbPipeline(steps)

def run_experiment(file_path, output_dir):
    dataset_name = os.path.basename(file_path).replace('.csv', '')
    output_path = os.path.join(output_dir, f"{dataset_name}_raw_results.csv")

    if os.path.exists(output_path):
        tqdm.write(f"File {os.path.basename(output_path)} already exists. Skipping...")
        return False

    df = pd.read_csv(file_path)

    X, y_raw = df.drop(columns=['target']), df['target']
    y = pd.Series(LabelEncoder().fit_transform(y_raw), index=X.index)
    
    num_cols = [c for c in X.columns if c.startswith('NUM')]
    cat_cols = [c for c in X.columns if c.startswith('CAT')]

    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    models = {
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=2000, n_jobs=1),
        'KNN': KNeighborsClassifier(n_jobs=1),
        'SVM': SVC(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
        'MLP': MLPClassifier(random_state=42, max_iter=2000, early_stopping=True),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=1)
    }

    if len(cat_cols) > 0:
        smote_technique = SMOTENC(categorical_features=cat_indices, random_state=42)
    else:
        smote_technique = SMOTE(random_state=42)

    preprocessors = {
        'Baseline': 'passthrough',
        
        'IQR Capping': OutlierCapper(capping_method='iqr', tail='both', fold=1.5),
        'IQR Removal': FunctionSampler(func=outlier_trimmer, kw_args={'method': 'iqr', 'fold': 1.5, 'cols': num_cols}, validate=False),
        
        'Standard Scaler': StandardScaler(),
        'Min-Max Scaler': MinMaxScaler(),
        
        'Yeo-Johnson': PowerTransformer(method='yeo-johnson'),
        'Quantile Transform': QuantileTransformer(output_distribution='normal', random_state=42),
        
        'Uniform Binning': KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'),
        'Quantile Binning': KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf'),
        
        'PCA': PCA(n_components=0.95, random_state=42),
        'Select Percentile (Mutual Info)': SelectPercentile(score_func=partial(mutual_info_classif, random_state=42), percentile=50),
        
        'Random Undersampling': RandomUnderSampler(random_state=42),
        'SMOTE / SMOTENC': smote_technique
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    
    pbar = tqdm(total=len(models) * len(preprocessors), desc=dataset_name, leave=False, position=1)

    for m_name, model_obj in models.items():
        for p_name, prep_obj in preprocessors.items():
            
            try:
                for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

                    gc.collect() 
                    pipeline = build_pipeline(model_obj, prep_obj, num_cols, cat_cols)

                    tracemalloc.start()
                    tracemalloc.clear_traces()
                    tracemalloc.reset_peak()
                    
                    t0_fit = time.process_time()
                    pipeline.fit(X_train, y_train)
                    fit_time = time.process_time() - t0_fit
                    
                    _, py_peak_mem = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    clf = pipeline.named_steps['classifier']
                    if hasattr(clf, 'n_iter_'):
                        val = clf.n_iter_
                        iters = val.max() if isinstance(val, np.ndarray) else val
                    else:
                        iters = np.nan
                        
                    disk_size = len(pickle.dumps(pipeline)) / (1024 * 1024)

                    y_train_scores = get_positive_scores(pipeline, X_train)
                    best_thresh, mcc_tr = optimize_threshold_mcc(y_train, y_train_scores)
                    
                    y_train_pred_opt = (y_train_scores >= best_thresh).astype(int)
                    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_train_pred_opt, labels=[0, 1]).ravel()
                    roc_tr = roc_auc_score(y_train, y_train_scores)

                    t0_pred_test = time.process_time()
                    y_test_scores = get_positive_scores(pipeline, X_test)
                    pred_time_test = time.process_time() - t0_pred_test
                    
                    y_test_pred_opt = (y_test_scores >= best_thresh).astype(int)
                    mcc_te = matthews_corrcoef(y_test, y_test_pred_opt)
                    
                    tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_test_pred_opt, labels=[0, 1]).ravel()
                    roc_te = roc_auc_score(y_test, y_test_scores)

                    row = {
                        'Dataset': dataset_name, 'Model': m_name, 'Preprocessor': p_name, 'Fold': fold_idx,
                        'Fit_Time_sec': fit_time, 'Predict_Time_Test_sec': pred_time_test,
                        'Peak_Memory_MB': py_peak_mem / (1024 * 1024), 
                        'Pipeline_Disk_Size_MB': disk_size, 'Iterations': iters, 
                        'Optimal_Threshold': best_thresh,
                        
                        'Train_TP': tp_tr, 'Train_TN': tn_tr, 'Train_FP': fp_tr, 'Train_FN': fn_tr, 
                        'Train_ROC_AUC': roc_tr, 'Train_MCC': mcc_tr,
                        
                        'Test_TP': tp_te, 'Test_TN': tn_te, 'Test_FP': fp_te, 'Test_FN': fn_te, 
                        'Test_ROC_AUC': roc_te, 'Test_MCC': mcc_te,
                        
                        'Error_Log': None
                    }
                    results.append(row)

            except Exception as e:
                err_row = {
                    'Dataset': dataset_name, 'Model': m_name, 'Preprocessor': p_name, 'Fold': np.nan,
                    'Fit_Time_sec': np.nan, 'Predict_Time_Test_sec': np.nan,
                    'Peak_Memory_MB': np.nan,'Pipeline_Disk_Size_MB': np.nan, 'Iterations': np.nan,
                    'Optimal_Threshold': np.nan,
                    
                    'Train_TP': np.nan, 'Train_TN': np.nan, 'Train_FP': np.nan, 'Train_FN': np.nan, 
                    'Train_ROC_AUC': np.nan, 'Train_MCC': np.nan,
                    
                    'Test_TP': np.nan, 'Test_TN': np.nan, 'Test_FP': np.nan, 'Test_FN': np.nan, 
                    'Test_ROC_AUC': np.nan, 'Test_MCC': np.nan,
                    
                    'Error_Log': traceback.format_exc()
                }
                results.append(err_row)
            finally:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                pbar.update(1)
                
    pbar.close()
    pd.DataFrame(results).to_csv(output_path, index=False)
    return True 

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../data/3 - interim'))
    OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../data/4 - results/1 - raw'))

    ASK_TO_CONTINUE = True
    TIMEOUT_SECONDS = 10

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]

    pbar_overall = tqdm(total=len(csv_files), desc="Overall", position=0, leave=True)

    for i, file in enumerate(csv_files):
        full_path = os.path.join(INPUT_DIR, file)
        
        exec_status = run_experiment(full_path, OUTPUT_DIR)
        
        pbar_overall.update(1)
        
        if not exec_status:
            pbar_overall.refresh()
            continue
        
        if ASK_TO_CONTINUE and i < len(csv_files) - 1:
            should_continue = ask_to_continue_with_timeout(TIMEOUT_SECONDS)
            
            if not should_continue:
                tqdm.write("\nAborted.")
                break

    pbar_overall.close()