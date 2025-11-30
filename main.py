import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, make_scorer,
    average_precision_score,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

DATASET_DIR = 'datasets/'
DATASET_NAMES = ['cardio', 'musk', 'vowels']
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_SPLITS = 5

all_run_results = []


def inverted_roc_auc_scorer(y_true, y_score, **kwargs):
    return roc_auc_score(y_true, -y_score)

for dataset_name in DATASET_NAMES:
    print(f"\n{'=' * 60}")
    print(f"Rozpoczynam przetwarzanie zbioru: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    FILE_NAME = os.path.join(DATASET_DIR, f"{dataset_name}.mat")

    print(f"Ładowanie zbioru '{FILE_NAME}'...")
    try:
        data = loadmat(FILE_NAME)
        X = data['X']
        y = data['y'].ravel()
        print(f"Załadowano. Wymiary X: {X.shape}, Wymiary y: {y.shape}")
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{FILE_NAME}'. Pomijam ten zbiór.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipe = Pipeline([
        ('scaler', 'passthrough'),
        ('model', IsolationForest(random_state=RANDOM_STATE))
    ])

    param_grid = {
        'scaler': [StandardScaler(), RobustScaler(), 'passthrough'],
        'model__n_estimators': [50, 100, 200],
        'model__max_samples': [0.5, 'auto'],
        'model__contamination': [0.01, 0.05, 0.1]
    }

    kfold_cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring_metric = make_scorer(inverted_roc_auc_scorer, needs_threshold=True)

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=kfold_cv,
        verbose=1,
        n_jobs=-1
    )

    print(f"Uruchamiam GridSearchCV dla Isolation Forest na zbiorze '{dataset_name}'...")
    grid_search.fit(X_train, y_train)

    print("\n--- Najlepsze parametry znalezione na zbiorze walidacyjnym ---")
    print(grid_search.best_params_)
    print(f"\n--- Najlepszy wynik (ROC-AUC) na CV ({CV_SPLITS}-fold) ---")
    print(f"{grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    y_scores_test = best_model.decision_function(X_test)
    y_pred_test = best_model.predict(X_test)
    y_pred_binary = np.where(y_pred_test == -1, 1, 0)

    print("\n--- Pełna Ocena Modelu na Zbiorze Testowym ---")

    test_roc_auc = roc_auc_score(y_test, -y_scores_test)
    test_pr_auc = average_precision_score(y_test, -y_scores_test)
    print(f"ROC-AUC (Score):   {test_roc_auc:.4f}")
    print(f"PR-AUC (Score):    {test_pr_auc:.4f}")

    test_f1 = f1_score(y_test, y_pred_binary, pos_label=1)
    test_recall = recall_score(y_test, y_pred_binary, pos_label=1)
    test_balanced_acc = balanced_accuracy_score(y_test, y_pred_binary)
    test_mcc = matthews_corrcoef(y_test, y_pred_binary)

    best_contamination = grid_search.best_params_['model__contamination']
    print(f"\n--- Metryki Binarne (dla 'outlier' przy contamination={best_contamination:.2f}) ---")
    print(f"F1 (Outlier):      {test_f1:.4f}")
    print(f"Recall (Outlier):  {test_recall:.4f}")
    print(f"Balanced Accuracy: {test_balanced_acc:.4f}")
    print(f"MCC:               {test_mcc:.4f}")

    run_summary = {
        'Dataset': dataset_name,
        'Model': 'IsolationForest',
        'CV_ROC_AUC': grid_search.best_score_,
        'Test_ROC_AUC': test_roc_auc,
        'Test_PR_AUC': test_pr_auc,
        'Test_F1_out': test_f1,
        'Test_Recall_out': test_recall,
        'Test_Balanced_Acc': test_balanced_acc,
        'Test_MCC': test_mcc,
        'Best_Scaler': str(grid_search.best_params_['scaler']),
        'Best_Contamination': best_contamination
    }
    all_run_results.append(run_summary)

print(f"\n{'=' * 60}")
print("Zakończono wszystkie eksperymenty. Podsumowanie:")
print(f"{'=' * 60}")

results_df = pd.DataFrame(all_run_results)
print(results_df.set_index('Dataset').to_string())
