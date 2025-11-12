import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# --- 1. Konfiguracja (bez zmian) ---

RESULTS_DIRS = {
    'IF': 'results_if',
    'LOF': 'results_lof',
    'SVM': 'results_svm'
}
DATASET_NAMES = ['cardio', 'musk', 'vowels']
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
print(f"Rozpoczynam generowanie wykresów. Zostaną zapisane w folderze: {PLOT_DIR}")


# --- 2. Funkcje pomocnicze do ładowania danych ---
# (Te funkcje są bez zmian)

def load_all_summaries():
    """Wczytuje wszystkie pliki 'summary_results.csv' i łączy je w jeden DataFrame."""
    all_summaries = []
    for model_short, res_dir in RESULTS_DIRS.items():
        summary_file = os.path.join(res_dir, f"{model_short.lower()}_summary_results.csv")
        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file)
            all_summaries.append(df)
        else:
            print(f"OSTRZEŻENIE: Nie znaleziono pliku podsumowania: {summary_file}")

    if not all_summaries:
        print("BŁĄD: Nie wczytano żadnych plików podsumowań. Prerywam.")
        return None

    return pd.concat(all_summaries, ignore_index=True)


def load_all_cv_results():
    """Wczytuje wszystkie pliki 'cv_results.csv' i łączy je w jeden DataFrame."""
    all_cv_data = []
    for model_short, res_dir in RESULTS_DIRS.items():
        for dataset in DATASET_NAMES:
            cv_file = os.path.join(res_dir, f"{dataset}_{model_short.lower()}_cv_results.csv")
            if os.path.exists(cv_file):
                df = pd.read_csv(cv_file)
                df['Model'] = model_short
                df['Dataset'] = dataset
                all_cv_data.append(df)
            else:
                print(f"OSTRZEŻENIE: Nie znaleziono pliku CV: {cv_file}")

    if not all_cv_data:
        print("BŁĄD: Nie wczytano żadnych plików CV. Prerywam.")
        return None

    full_cv_df = pd.concat(all_cv_data, ignore_index=True)

    # Ujednolicenie kolumn parametrów (contamination i nu)
    if 'param_model__nu' in full_cv_df.columns:
        full_cv_df['param_contamination'] = full_cv_df['param_model__nu']
    else:
        full_cv_df['param_contamination'] = np.nan

    if 'param_model__contamination' in full_cv_df.columns:
        full_cv_df['param_contamination'] = full_cv_df['param_contamination'].fillna(
            full_cv_df['param_model__contamination'])

    return full_cv_df


def clean_scaler_name(scaler_str):
    """Zamienia pełną nazwę skalera na czytelną etykietę."""
    if 'StandardScaler' in scaler_str:
        return 'StandardScaler'
    if 'RobustScaler' in scaler_str:
        return 'RobustScaler'
    if 'passthrough' in scaler_str:
        return 'Brak'  # Zmienione z 'Brak (passthrough)'
    return 'Inny'


# --- 3. Funkcje generujące wykresy ---

def plot_model_comparison(summary_df):
    # ... (bez zmian) ...
    metrics_to_plot = ['Test_PR_AUC', 'Test_ROC_AUC', 'Test_F1_out', 'Test_Balanced_Acc']
    df_melted = summary_df.melt(
        id_vars=['Dataset', 'Model'],
        value_vars=metrics_to_plot,
        var_name='Metryka',
        value_name='Wynik'
    )
    model_map = {'IsolationForest': 'IF', 'LocalOutlierFactor': 'LOF', 'OneClassSVM': 'SVM'}
    df_melted['Model'] = df_melted['Model'].map(model_map)

    g = sns.catplot(
        data=df_melted,
        x='Dataset',
        y='Wynik',
        hue='Model',
        col='Metryka',
        kind='bar',
        height=4,
        aspect=0.8,
        sharey=False
    )
    g.fig.suptitle('Wykres 1: Porównanie końcowych metryk modeli', y=1.03)
    g.set_axis_labels("Zbiór danych", "Wynik")
    g.set_titles("{col_name}")
    plot_path = os.path.join(PLOT_DIR, '1_model_comparison.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres 1: {plot_path}")


def plot_pr_curves(summary_df):
    # ... (bez zmian) ...
    model_map = {'IsolationForest': 'IF', 'LocalOutlierFactor': 'LOF', 'OneClassSVM': 'SVM'}
    model_map_inv = {v: k for k, v in model_map.items()}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Wykres 2: Krzywe Precyzja-Przywołanie (PR) dla najlepszych modeli', fontsize=16)

    for ax, dataset in zip(axes, DATASET_NAMES):
        best_model_row = summary_df[summary_df['Dataset'] == dataset].nlargest(1, 'Test_PR_AUC').iloc[0]
        model_name = best_model_row['Model']
        model_short = model_map[model_name]
        pr_auc = best_model_row['Test_PR_AUC']
        pr_data_file = os.path.join(RESULTS_DIRS[model_short], f"{dataset}_{model_short.lower()}_pr_data.npz")

        if not os.path.exists(pr_data_file):
            print(f"OSTRZEŻENIE: Nie znaleziono pliku PR {pr_data_file} dla zwycięzcy.")
            continue

        data = np.load(pr_data_file)
        y_test = data['y_test']
        y_scores = -data['y_scores']
        precision, recall, _ = precision_recall_curve(y_test, y_scores)

        ax.plot(recall, precision, label=f'{model_name}\n(PR-AUC = {pr_auc:.4f})')
        ax.set_title(f"Zbiór: {dataset.capitalize()}")
        ax.set_xlabel('Recall (Czułość)')
        ax.set_ylabel('Precision (Precyzja)')
        ax.legend(loc='lower left')
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plot_path = os.path.join(PLOT_DIR, '2_pr_curves.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres 2: {plot_path}")


def plot_contamination_effect(full_cv_df):
    """WYKRES 3: Wpływ parametru contamination/nu."""

    # Przygotowanie danych
    df_plot = full_cv_df[['param_contamination', 'mean_test_score', 'Model', 'Dataset']].copy()

    # ZMIANA: Konwertujemy na string, aby 'auto' i 0.01 były traktowane jako kategorie
    df_plot['param_contamination'] = df_plot['param_contamination'].astype(str)
    df_plot = df_plot.dropna(subset=['param_contamination'])

    # ZMIANA: Ustawienie kolejności na osi X
    param_order = sorted(df_plot['param_contamination'].unique(),
                         key=lambda x: (x != 'auto', pd.to_numeric(x, errors='coerce')))

    g = sns.catplot(
        data=df_plot,
        x='param_contamination',
        y='mean_test_score',
        hue='Model',
        col='Dataset',
        kind='point',
        height=4,
        aspect=0.8,
        # ZMIANA: Wyłączenie "zakresów" (przedziałów ufności)
        errorbar=None,
        order=param_order
    )

    g.fig.suptitle('Wykres 3: Wpływ parametru Contamination na wynik ROC-AUC (CV)', y=1.03)
    g.set_axis_labels("Wartość parametru Contamination", "Uśredniony ROC-AUC (CV)")
    g.set_titles("{col_name}")

    plot_path = os.path.join(PLOT_DIR, '3_contamination_effect.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres 3: {plot_path}")


def plot_scaling_effect(full_cv_df):
    """WYKRES 4: Wpływ skalowania danych."""

    df_plot = full_cv_df[['param_scaler', 'mean_test_score', 'Model', 'Dataset']].copy()
    df_plot['Skaler'] = df_plot['param_scaler'].apply(clean_scaler_name)

    g = sns.catplot(
        data=df_plot,
        x='Skaler',
        y='mean_test_score',
        hue='Model',
        col='Dataset',
        kind='point',
        height=4,
        aspect=0.8,
        order=['Brak', 'StandardScaler', 'RobustScaler'],
        # ZMIANA: Wyłączenie "zakresów" (przedziałów ufności)
        errorbar=None
    )

    g.fig.suptitle('Wykres 4: Wpływ skalowania na wynik ROC-AUC (CV)', y=1.03)
    g.set_axis_labels("Metoda skalowania", "Uśredniony ROC-AUC (CV)")
    g.set_titles("{col_name}")

    plot_path = os.path.join(PLOT_DIR, '4_scaling_effect.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Zapisano wykres 4: {plot_path}")


# --- 4. Główna funkcja wykonawcza ---

def main():
    print("--- Krok 1: Ładowanie danych podsumowujących ---")
    summary_df = load_all_summaries()
    if summary_df is None:
        return

    print("--- Krok 2: Ładowanie pełnych danych CV ---")
    full_cv_df = load_all_cv_results()
    if full_cv_df is None:
        # Ten błąd może wystąpić, jeśli nie uruchomiłeś zmodyfikowanych skryptów
        print("BŁĄD: full_cv_df jest pusty. Upewnij się, że uruchomiłeś zmodyfikowane skrypty IF, LOF i SVM,")
        print("aby wygenerować pliki *_cv_results.csv w folderach results_*.")
        # Kontynuujemy, aby wygenerować wykresy 1 i 2, które mogą jeszcze działać

    print("\n--- Krok 3: Generowanie wykresów ---")

    if summary_df is not None:
        # Wykres 1: Porównanie modeli i metryk
        plot_model_comparison(summary_df)

        # Wykres 2: Krzywe PR
        plot_pr_curves(summary_df)
    else:
        print("Pominięto wykresy 1 i 2 z powodu braku danych podsumowujących.")

    if full_cv_df is not None:
        # Wykres 3: Wpływ contamination
        plot_contamination_effect(full_cv_df)

        # Wykres 4: Wpływ skalowania
        plot_scaling_effect(full_cv_df)
    else:
        print("Pominięto wykresy 3 i 4 z powodu braku danych CV.")

    print(f"\n--- Zakończono ---")
    print(f"Wszystkie wygenerowane wykresy znajdują się w folderze: {PLOT_DIR}")


if __name__ == '__main__':
    main()