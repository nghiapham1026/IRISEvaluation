import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define models to evaluate, wrapped in pipelines for preprocessing
models = {
    "Naive Bayes": Pipeline([('scaler', StandardScaler()), ('classifier', GaussianNB())]),
    "Support Vector Machine": Pipeline([('scaler', StandardScaler()), ('classifier', SVC(probability=True))]),  # Enable probability for ROC AUC
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "K-Nearest Neighbors": Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier())])
}

# Define 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_weighted': make_scorer(f1_score, average='weighted', labels=np.unique(y)),
    'roc_auc_ovr': make_scorer(roc_auc_score, needs_proba=True, average='macro', multi_class='ovr')
}

# Run cross-validation and collect results
results = {}
for name, pipeline in models.items():
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    results[name] = {
        'Accuracy': np.mean(cv_results['test_accuracy']),
        'F1 Score (Weighted)': np.mean(cv_results['test_f1_weighted']),
        'ROC AUC (OvR)': np.mean(cv_results['test_roc_auc_ovr'])
    }

# Display results
for model, scores in results.items():
    print(f"Model: {model}")
    for score_name, score_value in scores.items():
        print(f"  {score_name}: {score_value:.4f}")
    print("-" * 30)
