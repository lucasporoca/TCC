import os
import gc

import time
import winsound
import msvcrt

import warnings
import tracemalloc
import traceback
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from sklearn import set_config
set_config(transform_output="pandas")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder, PowerTransformer, QuantileTransformer, 
                                   MinMaxScaler, StandardScaler, RobustScaler, 
                                   KBinsDiscretizer, LabelEncoder)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn import FunctionSampler
from feature_engine.outliers import Winsorizer, OutlierTrimmer

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

def build_pipeline(model_obj, prep_obj, num_cols, cat_cols):
    transformers = []
    sampler = None
    num_step = 'passthrough'

    if isinstance(prep_obj, (SMOTE, SMOTENC, RandomUnderSampler, FunctionSampler)):
        sampler = prep_obj
    else:
        if prep_obj != 'passthrough':
            num_step = prep_obj

    transformers.append(('num', num_step, num_cols))

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
    steps.append(('classifier', model_obj))

    return ImbPipeline(steps)


def trim_outliers(X, y, method, fold, cols):
    trimmer = OutlierTrimmer(capping_method=method, tail='both', fold=fold, variables=cols)
    X_res = trimmer.fit_transform(X)
    return X_res, y.loc[X_res.index]


def get_positive_scores(pipeline, X_data):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X_data)[:, 1]
    else:
        return pipeline.decision_function(X_data)


def run_experiment(file_path, output_dir):
    dataset_name = os.path.basename(file_path).replace('.csv', '')
    output_path = os.path.join(output_dir, f"{dataset_name}_results.csv")

    if os.path.exists(output_path):
        tqdm.write(f"File {os.path.basename(output_path)} already exists. Skipping...")
        return False

    df = pd.read_csv(file_path)
    X, y_raw = df.drop(columns=['target']), df['target']
    
    y = pd.Series(LabelEncoder().fit_transform(y_raw), index=X.index)
    
    num_cols = [c for c in X.columns if c.startswith('NUM')]
    cat_cols = [c for c in X.columns if c.startswith('CAT')]

    models = {
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=2000, n_jobs=1),
        'KNN': KNeighborsClassifier(n_jobs=1),
        'SVM': SVC(random_state=42, max_iter=2000), 
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1),
        'MLP': MLPClassifier(random_state=42, max_iter=2000, early_stopping=True)
    }

    if len(cat_cols) > 0:
        smote_technique = SMOTENC(categorical_features=cat_cols, random_state=42)
    else:
        smote_technique = SMOTE(random_state=42)

    preprocessors = {
        'Baseline': 'passthrough',
        'IQR Capping': Winsorizer(capping_method='iqr', tail='both', fold=1.5),
        'Quantile Capping': Winsorizer(capping_method='quantiles', tail='both', fold=0.05),
        'IQR Removal': FunctionSampler(func=trim_outliers, kw_args={'method': 'iqr', 'fold': 1.5, 'cols': num_cols}, validate=False),
        'Quantile Removal': FunctionSampler(func=trim_outliers, kw_args={'method': 'quantiles', 'fold': 0.05, 'cols': num_cols}, validate=False),
        'Yeo-Johnson': PowerTransformer(method='yeo-johnson'),
        'Quantile Transform': QuantileTransformer(output_distribution='normal', random_state=42),
        'Min-Max Scaler': MinMaxScaler(),
        'Standard Scaler': StandardScaler(),
        'Robust Scaler': RobustScaler(),
        'Uniform Binning': KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'),
        'Quantile Binning': KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf'),
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

                    t0_pred_train = time.process_time()
                    y_train_pred = pipeline.predict(X_train)
                    pred_time_train = time.process_time() - t0_pred_train
                    y_train_scores = get_positive_scores(pipeline, X_train)
                    
                    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_train_pred, labels=[0, 1]).ravel()
                    roc_tr = roc_auc_score(y_train, y_train_scores)
                    pr_tr = average_precision_score(y_train, y_train_scores)

                    t0_pred_test = time.process_time()
                    y_test_pred = pipeline.predict(X_test)
                    pred_time_test = time.process_time() - t0_pred_test
                    y_test_scores = get_positive_scores(pipeline, X_test)
                    
                    tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()
                    roc_te = roc_auc_score(y_test, y_test_scores)
                    pr_te = average_precision_score(y_test, y_test_scores)

                    row = {
                        'Dataset': dataset_name, 'Model': m_name, 'Preprocessor': p_name, 'Fold': fold_idx,
                        'Fit_Time_sec': fit_time, 'Predict_Time_Train_sec': pred_time_train, 'Predict_Time_Test_sec': pred_time_test,
                        'Peak_Memory_MB': py_peak_mem / (1024 * 1024), 
                        'Pipeline_Disk_Size_MB': disk_size, 'Iterations': iters, 
                        
                        'Train_TP': tp_tr, 'Train_TN': tn_tr, 'Train_FP': fp_tr, 'Train_FN': fn_tr, 
                        'Train_ROC_AUC': roc_tr, 'Train_PR_AUC': pr_tr,
                        
                        'Test_TP': tp_te, 'Test_TN': tn_te, 'Test_FP': fp_te, 'Test_FN': fn_te, 
                        'Test_ROC_AUC': roc_te, 'Test_PR_AUC': pr_te,
                        
                        'Error_Log': None
                    }
                    results.append(row)

            except Exception as e:
                err_row = {
                    'Dataset': dataset_name, 'Model': m_name, 'Preprocessor': p_name, 'Fold': np.nan,
                    'Fit_Time_sec': np.nan, 'Predict_Time_Train_sec': np.nan, 'Predict_Time_Test_sec': np.nan,
                    'Peak_Memory_MB': np.nan,'Pipeline_Disk_Size_MB': np.nan, 'Iterations': np.nan, 
                    
                    'Train_TP': np.nan, 'Train_TN': np.nan, 'Train_FP': np.nan, 'Train_FN': np.nan, 
                    'Train_ROC_AUC': np.nan, 'Train_PR_AUC': np.nan,
                    
                    'Test_TP': np.nan, 'Test_TN': np.nan, 'Test_FP': np.nan, 'Test_FN': np.nan, 
                    'Test_ROC_AUC': np.nan, 'Test_PR_AUC': np.nan,
                    
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../data/3 - interim'))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../data/4 - results'))

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