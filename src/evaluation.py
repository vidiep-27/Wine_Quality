"""Đánh giá nhất quán và lưu kết quả từng fold."""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from .pipelines import SCORING

def evaluate_cv(name, estimator, X, y, folds, n_jobs=2):
    res = cross_validate(estimator, X, y, cv=folds, scoring=SCORING,
                         return_train_score=True, n_jobs=n_jobs, error_score='raise')
    row = {'model': name}
    fold_rows = []
    for metric in SCORING:
        for split in ['train', 'test']:
            values = res[f'{split}_{metric}']
            prefix = 'val' if split == 'test' else 'train'
            row[f'{prefix}_{metric}_mean'] = float(np.mean(values))
            row[f'{prefix}_{metric}_std'] = float(np.std(values, ddof=1))
    row['fit_time_mean'] = float(np.mean(res['fit_time']))
    for i in range(len(folds)):
        fold_rows.append(dict(model=name, fold=i + 1, **{
            k: float(v[i]) for k, v in res.items()}))
    return row, pd.DataFrame(fold_rows)

def metrics(y_true, y_pred):
    return dict(accuracy=accuracy_score(y_true, y_pred),
                balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
                f1_macro=f1_score(y_true, y_pred, average='macro', zero_division=0),
                f1_weighted=f1_score(y_true, y_pred, average='weighted', zero_division=0))
