import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.neighbors import LocalOutlierFactor
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

# --- 1. Konfiguracja ---
from conf import *

# --- NOWOŚĆ: Utworzenie katalogu na wyniki LOF ---
RESULTS_DIR = 'results_lof'
os.makedirs(RESULTS_DIR, exist_ok=True)

all_run_results_lof = []


# --- 2. Funkcja scoringowa (bez zmian) ---
def inverted_roc_auc_scorer(y_true, y_score, **kwargs):
    # LOF z sklearn (novelty=True) również daje niższe wyniki anomaliom
    return roc_auc_score(y_true, -y_score)


# --- 3. Pętla eksperymentów (LOF) ---
for dataset_name in DATASET_NAMES:
    print(f"\n{'=' * 60}")
    print(f"Przetwarzanie zbioru: {dataset_name.upper()} (LOF)")
    print(f"{'=' * 60}")

    FILE_NAME = os.path.join(DATASET_DIR, f"{dataset_name}.mat")
    try:
        data = loadmat(FILE_NAME)
        X = data['X']
        y = data['y'].ravel()
        print(f"Załadowano. Wymiary X: {X.shape}, Wymiary y: {y.shape}")
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{FILE_NAME}'. Pomijam.")
        continue

    # --- Podział danych (bez zmian) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --- Pipeline i GridSearch (bez zmian) ---
    pipe = Pipeline([
        ('scaler', 'passthrough'),
        # Ważne: novelty=True jest wymagane dla GridSearchCV i predykcji
        ('model', LocalOutlierFactor(novelty=True))
    ])

    param_grid = {
        'scaler': [StandardScaler(), RobustScaler(), 'passthrough'],
        'model__n_neighbors': [10, 20, 35],
        # 'auto' jest przydatne, ale dla wykresów wpływu contamination
        # warto by było mieć stałe wartości, np. [0.01, 0.05, 0.1]
        # Zostawiam jednak Twoją logikę
        'model__contamination': ['auto', 0.01, 0.05]
    }

    kfold_cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring_metric = make_scorer(inverted_roc_auc_scorer, needs_threshold=True)

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=kfold_cv,
        verbose=1,
        n_jobs=-1,
        refit=True
    )

    print(f"Uruchamiam GridSearchCV dla LOF na zbiorze '{dataset_name}'...")
    grid_search.fit(X_train, y_train)

    # --- NOWOŚĆ: Zapis pełnych wyników CV do CSV ---
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_filename = os.path.join(RESULTS_DIR, f"{dataset_name}_lof_cv_results.csv")
    cv_results_df.to_csv(cv_filename, index=False)
    print(f"Zapisano pełne wyniki CV do: {cv_filename}")

    # --- Najlepsze wyniki (bez zmian) ---
    print("\n--- Najlepsze parametry (CV) ---")
    print(grid_search.best_params_)
    print(f"ROC-AUC (CV): {grid_search.best_score_:.4f}")

    # --- Ocena testowa (logika bez zmian) ---
    best_model = grid_search.best_estimator_
    y_scores_test = best_model.decision_function(X_test)
    y_pred_test = best_model.predict(X_test)
    y_pred_binary = np.where(y_pred_test == -1, 1, 0)

    # --- NOWOŚĆ: Zapis danych do krzywej PR ---
    pr_data_filename = os.path.join(RESULTS_DIR, f"{dataset_name}_lof_pr_data.npz")
    np.savez_compressed(pr_data_filename, y_test=y_test, y_scores=y_scores_test)
    print(f"Zapisano dane PR do: {pr_data_filename}")

    test_roc_auc = roc_auc_score(y_test, -y_scores_test)
    test_pr_auc = average_precision_score(y_test, -y_scores_test)
    test_f1 = f1_score(y_test, y_pred_binary, pos_label=1)
    test_recall = recall_score(y_test, y_pred_binary, pos_label=1)
    test_balanced = balanced_accuracy_score(y_test, y_pred_binary)
    test_mcc = matthews_corrcoef(y_test, y_pred_binary)

    best_cont = grid_search.best_params_['model__contamination']
    # Twoja logika do obsługi 'auto' - bardzo dobra
    best_cont_readable = None if best_cont == 'auto' else best_cont

    print("\n--- Testowe metryki ---")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"PR-AUC: {test_pr_auc:.4f}")
    print(f"F1: {test_f1:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Balanced Acc: {test_balanced:.4f}")
    print(f"MCC: {test_mcc:.4f}")

    # --- Zbieranie podsumowania (bez zmian) ---
    run_summary = {
        'Dataset': dataset_name,
        'Model': 'LocalOutlierFactor',
        'CV_ROC_AUC': grid_search.best_score_,
        'Test_ROC_AUC': test_roc_auc,
        'Test_PR_AUC': test_pr_auc,
        'Test_F1_out': test_f1,
        'Test_Recall_out': test_recall,
        'Test_Balanced_Acc': test_balanced,
        'Test_MCC': test_mcc,
        'Best_Scaler': str(grid_search.best_params_['scaler']),
        'Best_Contamination': best_cont_readable,
        'Best_n_neighbors': grid_search.best_params_['model__n_neighbors']
    }
    all_run_results_lof.append(run_summary)

# --- 4. Podsumowanie (logika bez zmian) ---
print(f"\n{'=' * 60}")
print("Zakończono eksperymenty LOF. Podsumowanie:")
print(f"{'=' * 60}")

results_df = pd.DataFrame(all_run_results_lof)
print(results_df.set_index('Dataset').to_string())

# --- NOWOŚĆ: Zapis tabeli podsumowującej do CSV ---
summary_filename = os.path.join(RESULTS_DIR, "lof_summary_results.csv")
results_df.to_csv(summary_filename, index=False)
print(f"\nZapisano tabelę podsumowującą do: {summary_filename}")