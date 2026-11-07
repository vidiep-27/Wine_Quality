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


GROUPS = {
    'vi': ['Dummy most_frequent', 'Dummy stratified', 'Logistic Regression', 'Decision Tree', 'Bagging'],
    'quan': ['Random Forest', 'Extra Trees'],
}

def data_signature(out):
    """Detect different training inputs or shared settings between members."""
    import hashlib
    digest = hashlib.sha256()
    for name in ['X_train.csv', 'y_train.csv', 'train_metadata.csv', 'config.json']:
        digest.update(name.encode())
        digest.update((out/name).read_bytes())
    from . import pipelines
    import inspect
    digest.update(inspect.getsource(pipelines).encode())
    digest.update(sklearn.__version__.encode())
    return digest.hexdigest()

def run_group(project_root, owner, n_iter=8, n_jobs=2):
    """Train only this member's models. Never read test files."""
    out = Path(project_root)/'output'
    dest = out/'members'/owner
    dest.mkdir(parents=True, exist_ok=True)
    names = GROUPS[owner]
    X = pd.read_csv(out/'X_train.csv')
    y = pd.read_csv(out/'y_train.csv')['label']
    from .pipelines import FEATURES
    assert list(X.columns) == FEATURES and X.shape[1] == 12
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    models = model_pipelines()
    summaries, details, searches, candidates = [], [], [], {}
    for name in names:
        print('CV:', owner, name, flush=True)
        row, detail = evaluate_cv(name, models[name], X, y, folds, n_jobs)
        summaries.append(row); details.append(detail)
        candidates[name] = clone(models[name])
        if name not in search_spaces():
            continue
        search = RandomizedSearchCV(models[name], search_spaces()[name],
            n_iter=n_iter, scoring=SCORING, refit='f1_macro', cv=folds,
            random_state=42, n_jobs=n_jobs, return_train_score=True, error_score='raise')
        search.fit(X, y)
        cvres = pd.DataFrame(search.cv_results_)
        cvres.to_csv(dest/f'search_{name.lower().replace(" ", "_")}.csv', index=False)
        best = search.best_index_
        vals = [float(cvres.loc[best, f'split{i}_test_f1_macro']) for i in range(5)]
        searches.append(dict(model=name, stage='tuned', f1_macro=float(search.best_score_),
            std=float(np.std(vals, ddof=1)),
            train_f1_macro=float(cvres.loc[best, 'mean_train_f1_macro']),
            best_params=search.best_params_))
        candidates[name+' tuned'] = search.best_estimator_
    initial = pd.DataFrame(summaries)
    tuned = pd.DataFrame(searches)
    initial.to_csv(dest/'initial_cv_results.csv', index=False)
    tuned.to_csv(dest/'tuned_cv_results.csv', index=False)
    pd.concat(details).to_csv(dest/'initial_fold_results.csv', index=False)
    joblib.dump(candidates, dest/'candidates.joblib')
    joblib.dump(folds, dest/'cv_folds.joblib')
    manifest = dict(owner=owner, models=names, n_iter=n_iter, random_state=42,
                    signature=data_signature(out), sklearn=sklearn.__version__)
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return initial, tuned

def aggregate_and_evaluate(project_root, n_jobs=2):
    """Vi merges both members' CV results, selects, then evaluates test."""
    root = Path(project_root); out = root/'output'; fig = out/'figures'
    fig.mkdir(exist_ok=True)
    if (out/'selection_decision.json').exists():
        raise RuntimeError('Đã chốt và đánh giá test. Đọc kết quả đã lưu; không chạy lại để chọn theo test.')
    started = time.perf_counter()
    X = pd.read_csv(out/'X_train.csv'); y = pd.read_csv(out/'y_train.csv')['label']
    signature = data_signature(out)
    initials, tuneds, details, candidates, manifests = [], [], [], {}, []
    folds = None
    for owner, names in GROUPS.items():
        dest = out/'members'/owner
        manifest = json.loads((dest/'manifest.json').read_text())
        assert manifest['signature'] == signature, 'Hai người phải dùng cùng dữ liệu, cấu hình và Pipeline.'
        assert manifest['models'] == names
        current_folds = joblib.load(dest/'cv_folds.joblib')
        if folds is None:
            folds = current_folds
        else:
            assert all(np.array_equal(a,c) and np.array_equal(b,d)
                       for (a,b),(c,d) in zip(folds,current_folds)), 'Khác fold CV'
        initials.append(pd.read_csv(dest/'initial_cv_results.csv'))
        tuneds.append(pd.read_csv(dest/'tuned_cv_results.csv'))
        details.append(pd.read_csv(dest/'initial_fold_results.csv'))
        candidates.update(joblib.load(dest/'candidates.joblib'))
        manifests.append(manifest)
    assert manifests[0]['n_iter'] == manifests[1]['n_iter'], 'Phải cùng ngân sách tìm kiếm'
    n_iter = manifests[0]['n_iter']
    initial = pd.concat(initials, ignore_index=True)
    tuned = pd.concat(tuneds, ignore_index=True)
    assert set(initial.model) == set(sum(GROUPS.values(), []))
    initial.to_csv(out/'initial_cv_results.csv', index=False)
    tuned.to_csv(out/'tuned_cv_results.csv', index=False)
    pd.concat(details).to_csv(out/'initial_fold_results.csv', index=False)
    initial[initial.model.isin(GROUPS['vi'][:3])].to_csv(out/'baseline_results.csv', index=False)
    selection = [dict(model=r.model, stage='initial', f1_macro=r.val_f1_macro_mean,
                      std=r.val_f1_macro_std) for r in initial.itertuples()]
    selection += [dict(model=r.model+' tuned', stage='tuned', f1_macro=r.f1_macro,
                       std=r.std) for r in tuned.itertuples()]
    board = pd.DataFrame(selection).sort_values(['f1_macro','std','model'], ascending=[False,True,True])
    board.to_csv(out/'selection_cv_results.csv', index=False)
    winner = board.iloc[0]['model']; final = clone(candidates[winner])
    decision = dict(model=winner, selection_metric='f1_macro',
        selection_cv_mean=float(board.iloc[0].f1_macro), selection_cv_std=float(board.iloc[0]['std']),
        n_iter=n_iter, random_state=42)
    (out/'selection_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
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
    return initial, tuned, decision, test_metrics
