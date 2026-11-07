"""CV và tìm kiếm chỉ nhận train; test chỉ được nạp sau khi chốt mô hình."""
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from .pipelines import model_pipelines, search_spaces, SCORING, CLASS_NAMES
from .evaluation import evaluate_cv, metrics

def run_training(project_root, n_iter=8, n_jobs=2):
    root = Path(project_root); out = root/'output'; fig = out/'figures'; fig.mkdir(exist_ok=True)
    started = time.perf_counter()
    X = pd.read_csv(out/'X_train.csv')
    y = pd.read_csv(out/'y_train.csv')['label']
    models = model_pipelines()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    joblib.dump(folds, out/'cv_folds.joblib')
    summaries, details = [], []
    for name, pipe in models.items():
        print('CV ban đầu:', name, flush=True)
        row, detail = evaluate_cv(name, pipe, X, y, folds, n_jobs)
        summaries.append(row); details.append(detail)
    initial = pd.DataFrame(summaries)
    initial.to_csv(out/'initial_cv_results.csv', index=False)
    pd.concat(details).to_csv(out/'initial_fold_results.csv', index=False)
    initial.iloc[:3].to_csv(out/'baseline_results.csv', index=False)

    candidates = {name: clone(pipe) for name, pipe in models.items()}
    selection = []
    for row in summaries:
        selection.append(dict(model=row['model'], stage='initial',
                              f1_macro=row['val_f1_macro_mean'],
                              std=row['val_f1_macro_std']))
    searches, params = [], {}
    for name, space in search_spaces().items():
        print('Tìm siêu tham số:', name, flush=True)
        search = RandomizedSearchCV(
            models[name], space, n_iter=n_iter, scoring=SCORING, refit='f1_macro',
            cv=folds, random_state=42, n_jobs=n_jobs,
            return_train_score=True, error_score='raise')
        search.fit(X, y)
        cvres = pd.DataFrame(search.cv_results_)
        cvres.to_csv(out/f'search_{name.lower().replace(" ", "_")}.csv', index=False)
        best = search.best_index_
        vals = [float(cvres.loc[best, f'split{i}_test_f1_macro']) for i in range(5)]
        row = dict(model=name, stage='tuned', f1_macro=float(search.best_score_),
                   std=float(np.std(vals, ddof=1)),
                   train_f1_macro=float(cvres.loc[best, 'mean_train_f1_macro']),
                   best_params=search.best_params_)
        searches.append(row); params[name] = search.best_params_
        candidates[name+' tuned'] = search.best_estimator_
        selection.append(dict(model=name+' tuned', stage='tuned',
                              f1_macro=row['f1_macro'], std=row['std']))
        print('  F1 macro:', round(search.best_score_, 4), flush=True)
    pd.DataFrame(searches).to_csv(out/'tuned_cv_results.csv', index=False)
    (out/'best_params.json').write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding='utf-8')
    board = pd.DataFrame(selection).sort_values(
        ['f1_macro', 'std', 'model'], ascending=[False, True, True])
    board.to_csv(out/'selection_cv_results.csv', index=False)
    winner = board.iloc[0]['model']
    final = clone(candidates[winner])
    # Quyết định được ghi xuống đĩa trước khi nạp hoặc đánh giá test.
    decision = dict(model=winner, selection_metric='f1_macro',
                    selection_cv_mean=float(board.iloc[0]['f1_macro']),
                    selection_cv_std=float(board.iloc[0]['std']),
                    n_iter=n_iter, random_state=42)
    (out/'selection_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print('Đã chốt mô hình:', winner, flush=True)

    # OOF chỉ mang tính chẩn đoán hậu lựa chọn, không là đánh giá độc lập.
    oof = cross_val_predict(final, X, y, cv=folds, n_jobs=n_jobs, method='predict')
    train_ids = pd.read_csv(out/'train_metadata.csv')
    oof_data = train_ids.copy()
    oof_data['true_label'] = y.to_numpy()
    oof_data['pred_label'] = oof
    oof_data['correct'] = oof_data.true_label == oof_data.pred_label
    oof_data.to_csv(out/'train_oof_predictions.csv', index=False)
    group_rows = []
    for kind, frame in [('all', oof_data), *list(oof_data.groupby('wine_type'))]:
        group_rows.append(dict(group=kind, n=len(frame), **metrics(frame.true_label, frame.pred_label)))
    pd.DataFrame(group_rows).to_csv(out/'oof_group_metrics.csv', index=False)
    pd.DataFrame(confusion_matrix(y, oof, labels=[0,1,2])).to_csv(out/'oof_confusion_matrix.csv', index=False)
    # Hoán vị trên validation của từng fold; pipeline fit riêng từng lần.
    importance_rows = []
    for fold_no, (tr, va) in enumerate(folds, 1):
        fitted = clone(final).fit(X.iloc[tr], y.iloc[tr])
        imp = permutation_importance(fitted, X.iloc[va], y.iloc[va],
                    scoring='f1_macro', n_repeats=5, random_state=42, n_jobs=n_jobs)
        for j, feature in enumerate(X.columns):
            for repeat, value in enumerate(imp.importances[j]):
                importance_rows.append(dict(fold=fold_no, feature=feature, repeat=repeat+1, importance=value))
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(out/'permutation_importance_repeats.csv', index=False)
    importance_summary = importance.groupby('feature')['importance'].agg(['mean','std']).sort_values('mean', ascending=False)
    importance_summary.to_csv(out/'permutation_importance.csv')
    ax = importance_summary.sort_values('mean')['mean'].plot.barh(
        xerr=importance_summary.sort_values('mean')['std'], figsize=(8,5), color='#366d9c')
    ax.set_xlabel('Decrease in validation macro F1'); ax.set_title('Permutation importance on training CV folds')
    plt.tight_layout(); plt.savefig(fig/'permutation_importance.png', dpi=180); plt.close()

    # Chỉ tại đây mới nạp test; không có tham số được chọn bằng test.
    Xt = pd.read_csv(out/'X_test.csv')
    yt = pd.read_csv(out/'y_test.csv')['label']
    final.fit(X, y)
    pred = final.predict(Xt)
    test_metrics = metrics(yt, pred)
    (out/'test_metrics.json').write_text(json.dumps(test_metrics, indent=2), encoding='utf-8')
    report = pd.DataFrame(classification_report(yt, pred, labels=[0,1,2],
                    target_names=CLASS_NAMES, output_dict=True, zero_division=0)).T
    report.to_csv(out/'test_classification_report.csv')
    test_data = pd.read_csv(out/'test_metadata.csv')
    test_data['true_label'] = yt.to_numpy(); test_data['pred_label'] = pred
    test_data['correct'] = test_data.true_label == test_data.pred_label
    for col in Xt: test_data[col] = Xt[col]
    test_data.to_csv(out/'test_predictions.csv', index=False)
    test_data.loc[~test_data.correct].head(20).to_csv(out/'test_error_examples.csv', index=False)
    matrix = confusion_matrix(yt, pred, labels=[0,1,2])
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out/'test_confusion_matrix.csv')
    for norm, fname in [(None, 'test_confusion_matrix'), ('true','test_confusion_matrix_normalized')]:
        disp = ConfusionMatrixDisplay.from_predictions(yt, pred, labels=[0,1,2],
             display_labels=['Low','Medium','High'], normalize=norm,
             cmap='Blues', values_format='d' if norm is None else '.2f', colorbar=False)
        disp.ax_.set_title('Final test confusion matrix')
        plt.tight_layout(); plt.savefig(fig/f'{fname}.png', dpi=180); plt.close()
    figx, ax = plt.subplots(figsize=(9,5))
    ordered = initial.sort_values('val_f1_macro_mean')
    ax.barh(ordered.model, ordered.val_f1_macro_mean,
            xerr=ordered.val_f1_macro_std, color='#366d9c')
    ax.set_xlabel('Validation macro F1 (mean ± sample SD)'); ax.set_xlim(0,1)
    plt.tight_layout(); plt.savefig(fig/'initial_cv_comparison.png', dpi=180); plt.close()
    joblib.dump(final, out/'final_pipeline.joblib')
    model = final.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        pd.Series(model.feature_importances_, index=X.columns, name='importance').sort_values(
            ascending=False).to_csv(out/'impurity_importance.csv')
    env = dict(python=platform.python_version(), sklearn=sklearn.__version__,
               pandas=pd.__version__, numpy=np.__version__, platform=platform.platform(),
               cpu=platform.processor() or 'not reported', n_jobs=n_jobs,
               duration_seconds=time.perf_counter()-started)
    (out/'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    print('Test metrics:', test_metrics, flush=True)
    return initial, pd.DataFrame(searches), decision, test_metrics
