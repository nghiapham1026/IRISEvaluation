import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
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
X, y = iris.data, iris.target  # Features and labels

# Set up 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),  # Overall correctness
    'f1_weighted': make_scorer(f1_score, average='weighted', labels=np.unique(y)),  # Weighted F1 score
    'roc_auc_ovr': make_scorer(roc_auc_score, needs_proba=True, average='macro', multi_class='ovr')  # One-vs-Rest ROC AUC
}

# Parameters for model tuning
param_grid = {
    "Support Vector Machine": {
        'classifier__C': [0.1, 1, 10, 100],  # Regularization parameter
        'classifier__gamma': [1, 0.1, 0.01, 0.001]  # Kernel coefficient
    },
    "K-Nearest Neighbors": {
        'classifier__n_neighbors': [3, 5, 7, 9],  # Number of neighbors
        'classifier__weights': ['uniform', 'distance']  # Weighting function
    }
}

# Initialize models within pipelines for preprocessing
models = {
    "Naive Bayes": Pipeline([('scaler', StandardScaler()), ('classifier', GaussianNB())]),
    "Support Vector Machine": Pipeline([('scaler', StandardScaler()), ('classifier', SVC(probability=True))]),  # For ROC AUC
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "K-Nearest Neighbors": Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier())])
}

# Perform model evaluation and hyperparameter tuning
results = {}
for name, pipeline in models.items():
    if name in param_grid:  # If hyperparameters are defined
        grid_search = GridSearchCV(pipeline, param_grid[name], cv=cv, scoring='accuracy', refit=True)
        grid_search.fit(X, y)  # Fit model
        best_model = grid_search.best_estimator_  # Best model found
        cv_results = cross_validate(best_model, X, y, cv=cv, scoring=scoring)  # Evaluate model
    else:  # For models without hyperparameter tuning
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    
    # Store results
    results[name] = {
        'Accuracy': np.mean(cv_results['test_accuracy']),
        'F1 Score (Weighted)': np.mean(cv_results['test_f1_weighted']),
        'ROC AUC (OvR)': np.mean(cv_results['test_roc_auc_ovr'])
    }

# Display the evaluation metrics for each model
for model, scores in results.items():
    print(f"Model: {model}")
    for score_name, score_value in scores.items():
        print(f"  {score_name}: {score_value:.4f}")
    print("-" * 30)