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

# --- 1. Konfiguracja ---
DATASET_DIR = 'datasets/'
# NOWOŚĆ: Lista zbiorów danych do przetworzenia
DATASET_NAMES = ['cardio', 'musk', 'vowels']
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_SPLITS = 5

all_run_results = []


# --- 2. Definicja funkcji scoringowej (bez zmian) ---
def inverted_roc_auc_scorer(y_true, y_score, **kwargs):
    return roc_auc_score(y_true, -y_score)


# --- 3. Główna pętla eksperymentu ---
for dataset_name in DATASET_NAMES:
    print(f"\n{'=' * 60}")
    print(f"Rozpoczynam przetwarzanie zbioru: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # --- 3.1. Ładowanie Danych ---
    FILE_NAME = os.path.join(DATASET_DIR, f"{dataset_name}.mat")

    print(f"Ładowanie zbioru '{FILE_NAME}'...")
    try:
        data = loadmat(FILE_NAME)
        X = data['X']
        y = data['y'].ravel()
        print(f"Załadowano. Wymiary X: {X.shape}, Wymiary y: {y.shape}")
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{FILE_NAME}'. Pomijam ten zbiór.")
        continue  # Przejdź do następnego zbioru w pętli

    # --- 3.2. Przygotowanie Danych (Podział) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # --- 3.3. Definicja Potoku i Siatki Parametrów ---
    # Siatka jest taka sama dla wszystkich - GridSearch znajdzie najlepsze
    # parametry dla każdego zbioru niezależnie.

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

    # --- 3.4. Konfiguracja Walidacji (GridSearchCV) ---
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

    # --- 3.5. Uruchomienie Eksperymentu ---
    print(f"Uruchamiam GridSearchCV dla Isolation Forest na zbiorze '{dataset_name}'...")
    grid_search.fit(X_train, y_train)

    # --- 3.6. Zbieranie Wyników (CV) ---
    print("\n--- Najlepsze parametry znalezione na zbiorze walidacyjnym ---")
    print(grid_search.best_params_)
    print(f"\n--- Najlepszy wynik (ROC-AUC) na CV ({CV_SPLITS}-fold) ---")
    print(f"{grid_search.best_score_:.4f}")

    # --- 3.7. Obliczanie Pełnego Zestawu Metryk (Test) ---
    best_model = grid_search.best_estimator_
    y_scores_test = best_model.decision_function(X_test)
    y_pred_test = best_model.predict(X_test)
    y_pred_binary = np.where(y_pred_test == -1, 1, 0)  # Mapowanie -1/1 -> 1/0

    print("\n--- Pełna Ocena Modelu na Zbiorze Testowym ---")

    # Metryki "Score"
    test_roc_auc = roc_auc_score(y_test, -y_scores_test)
    test_pr_auc = average_precision_score(y_test, -y_scores_test)
    print(f"ROC-AUC (Score):   {test_roc_auc:.4f}")
    print(f"PR-AUC (Score):    {test_pr_auc:.4f}")

    # Metryki binarne
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

    # --- 3.8. Zapisywanie wyników do listy ---
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

# --- 4. Podsumowanie Końcowe ---
print(f"\n{'=' * 60}")
print("Zakończono wszystkie eksperymenty. Podsumowanie:")
print(f"{'=' * 60}")

results_df = pd.DataFrame(all_run_results)
# Ustawienie 'Dataset' jako indeksu dla czytelności
print(results_df.set_index('Dataset').to_string())
