"""Thí nghiệm cải tiến và ablation sau khi đã chốt mô hình.

run_improvement() chạy trên train với năm fold đã lưu. evaluate_improved()
chọn ứng viên theo CV, ghi quyết định rồi mới nạp test để đối chiếu.
A0 là mô hình chính đã chốt trong selection_decision.json.
"""
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, validation_curve
from sklearn.pipeline import Pipeline
from .pipelines import SCORING, RANDOM_STATE

# Chỉ số bổ sung cho lớp Cao (nhãn 2), lớp thiểu số và bị nhận diện kém nhất.
ABLATION_SCORING = dict(SCORING,
    recall_high=make_scorer(recall_score, labels=[2], average='macro', zero_division=0),
    precision_high=make_scorer(precision_score, labels=[2], average='macro', zero_division=0))


class SulfurRatio(BaseEstimator, TransformerMixin):
    """Thêm tỷ lệ SO2 tự do / SO2 tổng; biến đổi theo từng dòng, không học tham số."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['free_so2_ratio'] = X['free sulfur dioxide'] / X['total sulfur dioxide'].clip(lower=1)
        return X


class ClassWeightedDecision(BaseEstimator, ClassifierMixin):
    """Nhân xác suất lớp Cao với hệ số w rồi lấy argmax; w chọn bằng CV lồng bên trong dữ liệu fit.

    Xác suất out-of-fold của dữ liệu fit (inner CV) dùng để chọn w tối đa hóa F1 macro, nên
    phần validation của fold ngoài không bao giờ tham gia chọn w. Mô hình cuối khớp lại trên
    toàn bộ dữ liệu fit.
    """

    def __init__(self, estimator=None, grid=(1.0, 1.25, 1.5, 1.75, 2.0, 2.5), inner_splits=3,
                 high_class=2, random_state=RANDOM_STATE):
        self.estimator = estimator
        self.grid = grid
        self.inner_splits = inner_splits
        self.high_class = high_class
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        inner = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        proba = cross_val_predict(clone(self.estimator), X, y, cv=inner, method='predict_proba')
        self.classes_ = np.unique(y)
        col = int(np.flatnonzero(self.classes_ == self.high_class)[0])
        best_w, best_f1 = 1.0, -1.0
        for w in self.grid:
            scale = np.ones(len(self.classes_)); scale[col] = w
            f1 = f1_score(y, self.classes_[(proba * scale).argmax(1)], average='macro')
            if f1 > best_f1 + 1e-12:
                best_w, best_f1 = w, f1
        self.weight_ = best_w
        self.col_ = col
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.estimator_.predict_proba(X).copy()
        p[:, self.col_] *= self.weight_
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]




def extra_trees(**kw):
    # Đúng cấu hình Extra Trees đã chốt (100 cây, mặc định), chỉ đổi tham số được nêu.
    params = dict(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
    params.update(kw)
    return ExtraTreesClassifier(**params)


def build(ratio=False, select=False, balanced=False, leaf=1, decision=False):
    steps = []
    if ratio:
        steps.append(('ratio', SulfurRatio()))
    steps.append(('imputer', SimpleImputer(strategy='median').set_output(transform='pandas')))
    if select:
        # Ngưỡng trung vị độ quan trọng impurity, học lại trong từng fold.
        steps.append(('select', SelectFromModel(extra_trees(), threshold='median')))
    model = extra_trees(class_weight='balanced' if balanced else None, min_samples_leaf=leaf)
    if decision:
        model = ClassWeightedDecision(model)
    return Pipeline(steps + [('model', model)])


# Mỗi dòng gộp các kỹ thuật cùng nhóm vấn đề; chỉ khác mốc ở các bước được nêu.
BASELINE = 'A0'   # mã nội bộ của Extra Trees ban đầu trong các tệp kết quả
ABLATION = [
    # (mã, mô tả, nhóm vấn đề, cấu hình)
    ('A0', 'Extra Trees ban đầu (mô hình đã chốt)', 'Mốc', {}),
    ('A1', '+ tỷ lệ SO2 tự do/tổng và chọn đặc trưng', 'Đặc trưng', dict(ratio=True, select=True)),
    ('A2', '+ class_weight=balanced và min_samples_leaf=3', 'Quá khớp và mất cân bằng lớp',
     dict(balanced=True, leaf=3)),
    ('A3', '+ hệ số quyết định cho lớp Cao (chọn bằng CV lồng)', 'Lớp Cao bị bỏ sót', dict(decision=True)),
    ('A4', '+ kết hợp A2 và A3', 'Kiểm tra kết hợp', dict(balanced=True, leaf=3, decision=True)),
]
ALL_CONFIGS = ABLATION


def summarize(res):
    row = {}
    for m in ABLATION_SCORING:
        v = res[f'test_{m}']
        row[f'{m}_mean'] = float(np.mean(v))
        row[f'{m}_std'] = float(np.std(v, ddof=1))
    row['train_f1_macro_mean'] = float(np.mean(res['train_f1_macro']))
    return row


# Tiêu chí đọc chênh lệch so với A0 (dùng chung cho notebook và báo cáo).
MIN_FOLDS_CONSISTENT = 4   # số fold tối thiểu cùng chiều để coi là nhất quán


def paired_deltas(folds_df, metric='f1_macro', base='A0'):
    """Chênh lệch ghép theo fold so với mốc: trung bình, độ lệch chuẩn, min, max, số fold tốt hơn.

    Các cấu hình dùng chung 5 fold nên so sánh ghép theo fold chính xác hơn so với
    đặt hai độ lệch chuẩn cạnh nhau. Với chỉ 5 fold, đây là mô tả, không phải kiểm định.
    """
    wide = folds_df.pivot(index='fold', columns='code', values=metric)
    rows = []
    for code in wide.columns:
        d = wide[code] - wide[base]
        rows.append(dict(code=code,
                         delta_mean=float(d.mean()),
                         delta_std=float(d.std(ddof=1)) if code != base else 0.0,
                         delta_min=float(d.min()), delta_max=float(d.max()),
                         folds_better=int((d > 0).sum()) if code != base else 0,
                         folds_worse=int((d < 0).sum()) if code != base else 0))
    out = pd.DataFrame(rows)
    out['verdict'] = [verdict(r) for r in out.itertuples()]
    return out


def verdict(r):
    """Cải thiện / giảm nhất quán nếu >= 4/5 fold cùng chiều; ngược lại: không phân biệt được."""
    if r.code == 'A0':
        return 'mốc'
    if r.folds_better >= MIN_FOLDS_CONSISTENT and r.delta_mean > 0:
        return 'cải thiện nhất quán'
    if r.folds_worse >= MIN_FOLDS_CONSISTENT and r.delta_mean < 0:
        return 'giảm nhất quán'
    return 'không phân biệt được với A0'


def check_baseline(out, a0_f1):
    """Kiểm tra mốc ablation với điểm CV đã lưu trước khi chọn ứng viên."""
    path = out / 'selection_decision.json'
    ref = json.loads(path.read_text(encoding='utf-8'))['selection_cv_mean']
    if not np.isclose(a0_f1, ref, rtol=0, atol=1e-6):
        raise ValueError(f'A0 ({a0_f1:.6f}) lệch điểm CV đã chốt ({ref:.6f}). Kiểm tra dữ liệu, fold và Pipeline.')
    return ref


def run_improvement(project_root, n_jobs=2):
    root = Path(project_root); out = root/'output'; fig = out/'figures'; fig.mkdir(exist_ok=True)
    X = pd.read_csv(out/'X_train.csv')
    y = pd.read_csv(out/'y_train.csv')['label']
    assert len(X) == len(y) and set(y) == {0, 1, 2}, 'Dữ liệu train không đúng định dạng.'
    folds = joblib.load(out / 'cv_folds.joblib')
    if len(folds) != 5:
        raise ValueError('cv_folds.joblib phải chứa đúng năm fold đã dùng để chọn mô hình.')

    rows, fold_rows = [], []
    kept, weights = [], []
    for code, desc, group, cfg in ALL_CONFIGS:
        print('Ablation', code, desc, flush=True)
        keep_estimators = cfg.get('select', False) or cfg.get('decision', False)
        res = cross_validate(build(**cfg), X, y, cv=folds, scoring=ABLATION_SCORING,
                             return_train_score=True, return_estimator=keep_estimators,
                             n_jobs=n_jobs, error_score='raise')
        for fold, pipe in enumerate(res.pop('estimator', ()), start=1):
            if cfg.get('select'):
                names = pipe.named_steps['imputer'].get_feature_names_out()
                mask = pipe.named_steps['select'].get_support()
                kept.append(dict(fold=fold, n_kept=int(mask.sum()), features='; '.join(names[mask])))
            if cfg.get('decision'):
                weights.append(dict(code=code, fold=fold, high_weight=float(pipe.named_steps['model'].weight_)))
        rows.append(dict(code=code, change=desc, group=group, **summarize(res)))
        for i in range(len(folds)):
            fold_rows.append(dict(code=code, fold=i + 1,
                                  f1_macro=float(res['test_f1_macro'][i]),
                                  recall_high=float(res['test_recall_high'][i]),
                                  precision_high=float(res['test_precision_high'][i]),
                                  balanced_accuracy=float(res['test_balanced_accuracy'][i])))
    abl = pd.DataFrame(rows)
    base = abl.loc[abl.code == 'A0'].iloc[0]
    folds_df = pd.DataFrame(fold_rows)
    pf = paired_deltas(folds_df).set_index('code')
    abl['delta_f1_macro'] = abl.f1_macro_mean - base.f1_macro_mean
    abl['delta_recall_high'] = abl.recall_high_mean - base.recall_high_mean
    abl['delta_f1_std'] = abl.code.map(pf.delta_std)       # SD của chênh lệch ghép theo fold
    abl['delta_f1_min'] = abl.code.map(pf.delta_min)
    abl['delta_f1_max'] = abl.code.map(pf.delta_max)
    abl['folds_better'] = abl.code.map(pf.folds_better)
    abl['folds_worse'] = abl.code.map(pf.folds_worse)
    abl['verdict'] = abl.code.map(pf.verdict)
    abl['gap_f1_macro'] = abl.train_f1_macro_mean - abl.f1_macro_mean
    check_baseline(out, float(base.f1_macro_mean))
    abl.to_csv(out/'ablation_results.csv', index=False)
    folds_df.to_csv(out/'ablation_fold_results.csv', index=False)

    pd.DataFrame(kept).to_csv(out / 'ablation_selected_features.csv', index=False)
    pd.DataFrame(weights).to_csv(out / 'ablation_decision_weights.csv', index=False)

    # 2. Đường cong kiểm định của Extra Trees theo min_samples_leaf và max_depth.
    curves = []
    for param, values in [('model__min_samples_leaf', [1, 2, 3, 5, 8, 12, 20]),
                          ('model__max_depth', [4, 6, 8, 10, 15, 20, 30])]:
        print('Validation curve', param, flush=True)
        tr_s, va_s = validation_curve(build(), X, y, param_name=param, param_range=values,
                                      cv=folds, scoring='f1_macro', n_jobs=n_jobs)
        for v, a, b in zip(values, tr_s, va_s):
            curves.append(dict(param=param.replace('model__', ''), value=v,
                               train_mean=a.mean(), train_std=a.std(ddof=1),
                               val_mean=b.mean(), val_std=b.std(ddof=1)))
    curves = pd.DataFrame(curves)
    curves.to_csv(out/'validation_curve_extra_trees.csv', index=False)

    plot_figures(out, abl, curves)
    return abl, curves


def plot_figures(out, abl, curves):
    fig = out/'figures'
    # Hình 1: F1 macro CV trước và sau tìm kiếm siêu tham số của bốn mô hình cây.
    init = pd.read_csv(out/'initial_cv_results.csv').set_index('model')
    tuned = pd.read_csv(out/'tuned_cv_results.csv').set_index('model')
    names = ['Decision Tree', 'Bagging', 'Random Forest', 'Extra Trees']
    x = np.arange(len(names)); w = 0.38
    f, ax = plt.subplots(figsize=(8, 4.6))
    a = ax.bar(x - w/2, init.loc[names, 'val_f1_macro_mean'], w, yerr=init.loc[names, 'val_f1_macro_std'],
               capsize=4, color='#9db9d3', label='Initial configuration')
    b = ax.bar(x + w/2, tuned.loc[names, 'f1_macro'], w, yerr=tuned.loc[names, 'std'],
               capsize=4, color='#366d9c', label='Best of RandomizedSearchCV')
    ax.bar_label(a, fmt='%.4f', padding=3, fontsize=8); ax.bar_label(b, fmt='%.4f', padding=3, fontsize=8)
    ax.set_xticks(x, names)
    lo = min((init.loc[names, 'val_f1_macro_mean'] - init.loc[names, 'val_f1_macro_std']).min(),
             (tuned.loc[names, 'f1_macro'] - tuned.loc[names, 'std']).min())
    hi = max((init.loc[names, 'val_f1_macro_mean'] + init.loc[names, 'val_f1_macro_std']).max(),
             (tuned.loc[names, 'f1_macro'] + tuned.loc[names, 'std']).max())
    ax.set_ylim(lo - 0.04, hi + 0.04)
    ax.set_ylabel('Validation macro F1 (mean ± SD, 5 folds)')
    ax.set_title('Macro F1 before and after hyperparameter search')
    ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(fig/'tuning_before_after.png', dpi=180); plt.close()

    # Hình 2: ablation, F1 macro và recall lớp Cao so với mốc A0.
    f, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(abl) + 2.4), sharey=True)
    labels = abl.code.replace({'A0': 'Initial Extra Trees'})[::-1]
    for ax, col, title, color in [(axes[0], 'f1_macro', 'Validation macro F1', '#366d9c'),
                                  (axes[1], 'recall_high', 'Validation recall of class High', '#c0504d')]:
        vals = abl[f'{col}_mean'][::-1]; errs = abl[f'{col}_std'][::-1]
        ax.barh(labels, vals, xerr=errs, color=color, alpha=0.85, capsize=3)
        ax.axvline(abl.loc[abl.code == 'A0', f'{col}_mean'].iloc[0], color='black', ls='--', lw=1,
                   label='Initial Extra Trees (selected model)')
        hi = max(vals + errs)
        for yv, v in zip(labels, vals):
            ax.text(hi + 0.012, yv, f'{v:.4f}', va='center', ha='left', fontsize=8)
        ax.set_xlabel(f'{title} (mean ± SD, 5 folds)')
        lo = min(vals - errs) - 0.03
        ax.set_xlim(lo, hi + 0.075)
        ax.grid(axis='x', alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=8, frameon=False)
    axes[0].set_ylabel('Ablation configuration')
    f.suptitle('Ablation of Extra Trees on the training folds')
    plt.tight_layout(); plt.savefig(fig/'ablation_extra_trees.png', dpi=180); plt.close()

    # Hình 3: đường cong kiểm định (train và validation) của Extra Trees.
    f, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, param in zip(axes, ['min_samples_leaf', 'max_depth']):
        c = curves[curves.param == param]
        for col, lab, color in [('train', 'Training', '#c0504d'), ('val', 'Validation', '#366d9c')]:
            x = c['value'].to_numpy()
            mean = c[f'{col}_mean'].to_numpy()
            std = c[f'{col}_std'].to_numpy()
            ax.plot(x, mean, 'o-', color=color, label=lab)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel(param); ax.grid(alpha=0.3)
    axes[0].set_ylabel('Macro F1 (mean ± SD, 5 folds)')
    axes[0].legend()
    f.suptitle('Validation curves of Extra Trees')
    plt.tight_layout(); plt.savefig(fig/'validation_curve_extra_trees.png', dpi=180); plt.close()


def choose_improved(out):
    """Quy tắc chọn ứng viên cải tiến, chỉ dựa trên CV của train (đặt trước khi nạp test).

    Ứng viên là cấu hình khác mốc, cải thiện nhất quán (>= 4/5 fold tốt hơn A0 và chênh lệch
    trung bình dương), lấy F1 macro CV cao nhất. Trả về dòng của bảng ablation hoặc None.
    """
    out = Path(out)
    abl = pd.read_csv(out/'ablation_results.csv')
    baseline = abl.loc[abl.code == BASELINE, 'f1_macro_mean'].item()
    check_baseline(out, baseline)
    cand = abl[
        (abl.code != BASELINE)
        & (abl.delta_f1_macro > 0)
        & (abl.folds_better >= MIN_FOLDS_CONSISTENT)
    ]
    return None if cand.empty else cand.sort_values('f1_macro_mean', ascending=False).iloc[0]


def evaluate_improved(project_root):
    """Đánh giá đúng MỘT lần ứng viên cải tiến trên tập kiểm tra.

    Ứng viên được chọn bằng CV trên train (choose_improved) và quyết định được ghi ra trước
    khi nạp test. Kết quả là phép đo tham khảo, không dùng để chọn hay chỉnh mô hình; hàm từ
    chối chạy lại nếu quyết định đã tồn tại. Lưu ý: tập kiểm tra đã được dùng để đánh giá
    mô hình A0, nên đây không phải một tập hoàn toàn chưa từng thấy.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    from .evaluation import metrics
    from .pipelines import CLASS_NAMES
    out = Path(project_root)/'output'
    decision_path = out / 'improved_decision.json'
    if decision_path.exists():
        raise RuntimeError('Đã chốt quyết định cho lần đánh giá test. Kiểm tra các tệp kết quả đã lưu.')
    row = choose_improved(out)
    if row is None:
        raise RuntimeError('Không có cấu hình vòng 2 nào cải thiện nhất quán; không có gì để đánh giá.')
    cfg = dict((c, k) for c, _, _, k in ALL_CONFIGS)[row.code]
    decision = dict(code=row.code, change=row.change, config=cfg,
                    selection_cv_mean=float(row.f1_macro_mean), selection_cv_std=float(row.f1_macro_std),
                    rule='cấu hình khác mốc, cải thiện nhất quán (>=4/5 fold), F1 macro CV cao nhất')
    X = pd.read_csv(out / 'X_train.csv')
    y = pd.read_csv(out / 'y_train.csv')['label']
    test_features = out / 'X_test.csv'
    test_labels = out / 'y_test.csv'
    for path in (test_features, test_labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    model = build(**cfg).fit(X, y)

    # Chế độ 'x' chặn cả hai lần gọi đồng thời trước khi đọc tập kiểm tra.
    with decision_path.open('x', encoding='utf-8') as file:
        json.dump(decision, file, indent=2, ensure_ascii=False)
    Xt = pd.read_csv(test_features)
    yt = pd.read_csv(test_labels)['label']
    pred = model.predict(Xt)
    result = {k: float(v) for k, v in metrics(yt, pred).items()}
    (out/'improved_test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(classification_report(yt, pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                 output_dict=True, zero_division=0)).T.to_csv(out/'improved_test_classification_report.csv')
    pd.DataFrame(confusion_matrix(yt, pred, labels=[0, 1, 2]), index=CLASS_NAMES,
                 columns=CLASS_NAMES).to_csv(out/'improved_test_confusion_matrix.csv')
    joblib.dump(model, out/'improved_pipeline.joblib')
    return decision, result
