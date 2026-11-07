"""Các Pipeline dùng chung; mọi biến đổi có học được fit trong từng fold."""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier

RANDOM_STATE = 42
PHYSICO = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol']
FEATURES = PHYSICO + ['wine_type_white']
CLASS_NAMES = ['Thấp', 'Trung bình', 'Cao']
SCORING = dict(accuracy='accuracy', balanced_accuracy='balanced_accuracy',
               f1_macro='f1_macro', f1_weighted='f1_weighted')

def make_pipeline(model, scale=False):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scaler', StandardScaler()))
    return Pipeline(steps + [('model', model)])

def model_pipelines():
    return {
        'Dummy most_frequent': make_pipeline(DummyClassifier(strategy='most_frequent')),
        'Dummy stratified': make_pipeline(DummyClassifier(strategy='stratified', random_state=42)),
        'Logistic Regression': make_pipeline(LogisticRegression(max_iter=3000, random_state=42), scale=True),
        'Decision Tree': make_pipeline(DecisionTreeClassifier(random_state=42)),
        'Bagging': make_pipeline(BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42), n_estimators=100,
            random_state=42, n_jobs=1)),
        'Random Forest': make_pipeline(RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=1)),
        'Extra Trees': make_pipeline(ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=1)),
    }

def search_spaces():
    # Tên tham số đi qua bước "model" của Pipeline.
    return {
        'Decision Tree': {
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_leaf': [1, 2, 5],
            'model__class_weight': [None, 'balanced']},
        'Bagging': {
            'model__n_estimators': [100, 200],
            'model__max_samples': [0.7, 1.0],
            'model__estimator__max_depth': [None, 10],
            'model__estimator__min_samples_leaf': [1, 3]},
        'Random Forest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_leaf': [1, 3],
            'model__max_features': ['sqrt', 1.0],
            'model__class_weight': [None, 'balanced']},
        'Extra Trees': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_leaf': [1, 3],
            'model__max_features': ['sqrt', 1.0],
            'model__class_weight': [None, 'balanced']},
    }
