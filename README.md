# Nathan Pham - Programming Assignment #2

This project evaluates multiple machine learning models on the Iris dataset. `iris_evaluation.py` includes preprocessing with standardization, hyperparameter tuning for select models, and cross-validation. `iris_evaluation_normalization.py` is similar to `iris_evaluation.py`, only difference is that `iris_evaluation_normalization.py` preprocesses the data using normalization instead of standardization.

## Dependencies

- Python 3.x
- scikit-learn
- numpy
- xgboost

You can install these packages using pip:

```bash
pip install numpy scikit-learn xgboost
```

## Running the Scripts

To run the script, navigate to the root directory:

```bash
python iris_evaluation.py
```

```bash
python iris_evaluation_normalization.py
```

## Output

The script outputs the performance metrics (Accuracy, F1 Score (Weighted), and ROC AUC (OvR)) for each evaluated model:

- Naive Bayes
- Support Vector Machine
- Random Forest
- XGBoost
- K-Nearest Neighbors

After standardization and hyperparameter tuning steps, the final performance metrics are displayed. The metrics for each model after each step are recorded in `output.txt`. The metrics for models preprocessing through normalization are recorded in `normalization_output.txt`
