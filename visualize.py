import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    accuracy_score, balanced_accuracy_score, top_k_accuracy_score
)
from sklearn.manifold import TSNE
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import k_hop_subgraph, degree
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

from src.model import GAT
from src.utils import get_device, apply_patches

# Konfiguracja globalna wykresów
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12  # <--- WIĘKSZA CZCIONKA GLOBALNIE
plt.rcParams['axes.labelsize'] = 14  # Większe etykiety osi
plt.rcParams['xtick.labelsize'] = 12  # Większe napisy na osi X
plt.rcParams['ytick.labelsize'] = 12  # Większe napisy na osi Y
sns.set_theme(style="whitegrid")

# Tworzenie folderu na wykresy
os.makedirs('plots', exist_ok=True)


# =============================================================================
# FUNKCJA POMOCNICZA DO NAZW KATEGORII
# =============================================================================

def load_category_mapping():
    """Wczytuje mapowanie ID -> Nazwa kategorii (np. 0 -> cs.AI)."""
    mapping_path = './data/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz'
    try:
        if not os.path.exists(mapping_path):
            # Próba alternatywnej ścieżki
            mapping_path = './data/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv'

        df = pd.read_csv(mapping_path)
        # Tworzymy słownik {0: 'cs.AI', ...} i czyścimy nazwy
        # Oryginalnie w pliku jest np. 'arxiv cs ai', zmieniamy na 'cs.AI'
        label_map = {}
        for idx, row in df.iterrows():
            raw_name = row['arxiv category']
            clean_name = raw_name.replace('arxiv ', '').replace(' ', '.')  # arxiv cs ai -> cs.ai
            label_map[row['label idx']] = clean_name

        print(f"Wczytano mapowanie dla {len(label_map)} kategorii.")
        return label_map
    except Exception as e:
        print(f"Nie udało się wczytać nazw kategorii: {e}")
        return {i: str(i) for i in range(40)}


# =============================================================================
# WYKRESY
# =============================================================================

def plot_learning_curves():
    """[1/10] Krzywe uczenia."""
    print("=" * 60)
    print("1. KRZYWE UCZENIA")
    print("=" * 60)

    if not os.path.exists('training_history.csv'):
        print("Brak pliku training_history.csv!")
        return

    df = pd.read_csv('training_history.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss
    sns.lineplot(data=df, x='epoch', y='loss', ax=ax1, color='#E74C3C', linewidth=2.5)
    ax1.fill_between(df['epoch'], df['loss'], alpha=0.2, color='#E74C3C')
    ax1.set_title('Funkcja Straty (Training Loss)', fontweight='bold')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # Accuracy
    sns.lineplot(data=df, x='epoch', y='train_acc', ax=ax2, label='Trening',
                 color='#2ECC71', linewidth=2.5)
    sns.lineplot(data=df, x='epoch', y='val_acc', ax=ax2, label='Walidacja',
                 color='#3498DB', linewidth=2.5)
    ax2.set_title('Dokładność (Accuracy)', fontweight='bold')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/learning_curves.png')  # Zapis do folderu plots
    print("Zapisano: plots/learning_curves.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, label_map):
    """[2/10] Macierz pomyłek z nazwami kategorii (Wersja HD)."""
    print("\n" + "=" * 60)
    print("2. MACIERZ POMYŁEK (HD)")
    print("=" * 60)

    cm = confusion_matrix(y_true, y_pred)
    labels = [label_map[i] for i in range(len(label_map))]

    # Zmiana: Znacznie większy rozmiar figury
    plt.figure(figsize=(24, 20))
    from matplotlib.colors import LogNorm

    # Heatmapa
    sns.heatmap(cm, cmap='Blues', norm=LogNorm(),
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Liczba (skala log)', 'shrink': 0.8}, # Shrink zmniejsza pasek legendy
                linewidths=0.5, linecolor='lightgray') # Dodatkowo linie siatki dla czytelności

    # Zmiana: Ogromne czcionki
    plt.title('Macierz Pomyłek (Confusion Matrix)', fontsize=28, fontweight='bold', pad=20)
    plt.xlabel('Predykcja modelu', fontsize=22, labelpad=15)
    plt.ylabel('Prawdziwa etykieta', fontsize=22, labelpad=15)

    # Zmiana: Czytelne etykiety osi
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)

    # Dostosowanie paska kolorów (colorbar) - dostęp przez ostatnią oś
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Liczba próbek (log)', fontsize=18)

    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', bbox_inches='tight') # bbox_inches='tight' ucina białe marginesy
    print("   ✅ Zapisano: plots/confusion_matrix.png")

    return cm


def plot_class_accuracy(cm, label_map):
    """[3/10] Dokładność per klasa z nazwami (Wersja HD)."""
    print("\n" + "=" * 60)
    print("3. DOKŁADNOŚĆ PER KLASA (HD)")
    print("=" * 60)

    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = cm.diagonal() / cm.sum(axis=1)
        class_acc = np.nan_to_num(class_acc)

    labels = [label_map[i] for i in range(len(class_acc))]

    # Zmiana: Szeroki wykres
    plt.figure(figsize=(24, 12))
    colors = ['#2ECC71' if a > 0.7 else '#F39C12' if a > 0.5 else '#E74C3C' for a in class_acc]

    # Słupki
    bars = plt.bar(range(len(class_acc)), class_acc, color=colors, edgecolor='black', alpha=0.8, width=0.7)

    # Linia średniej
    mean_val = np.mean(class_acc)
    plt.axhline(y=mean_val, color='red', linestyle='--', linewidth=3,
                label=f'Średnia: {mean_val:.3f}')

    # Opisy
    plt.title('Dokładność dla poszczególnych kategorii tematycznych', fontsize=28, fontweight='bold', pad=20)
    plt.ylabel('Dokładność (Accuracy)', fontsize=22)
    plt.xlabel('Kategoria arXiv', fontsize=22)

    # Zmiana: Duże etykiety na osi X
    plt.xticks(range(len(class_acc)), labels, rotation=90, fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlim(-1, len(class_acc))
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    plt.legend(fontsize=18, loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/class_accuracy.png', bbox_inches='tight')
    print("   ✅ Zapisano: plots/class_accuracy.png")
    plt.close()


def plot_tsne(embeddings, y_true, y_pred):
    """[4/10] Generowanie 3 osobnych wykresów t-SNE (Błędy jako kropki)."""
    print("\n" + "=" * 60)
    print("4. GENEROWANIE WYKRESÓW T-SNE (OSOBNE PLIKI)")
    print("=" * 60)

    # 1. Próbkowanie danych (t-SNE jest wolne, więc bierzemy próbkę np. 3000 punktów)
    n_samples = min(3000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)

    emb_sample = embeddings[indices]
    y_sample = y_true[indices]
    pred_sample = y_pred[indices]
    correct_sample = (y_sample == pred_sample)

    print(f"   -> Obliczanie t-SNE dla {n_samples} próbek (to może chwilę potrwać)...")

    # Obliczenie współrzędnych t-SNE (robimy to raz, używamy do wszystkich wykresów)
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    z = tsne.fit_transform(emb_sample)

    # Definicja wspólnej palety kolorów
    cmap = 'tab20'

    # =========================================================
    # WYKRES 1: RZECZYWISTE ETYKIETY (Ground Truth)
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    scatter1 = ax1.scatter(z[:, 0], z[:, 1], c=y_sample, cmap=cmap, s=20, alpha=0.7)

    ax1.set_title('t-SNE: Rzeczywisty podział na klasy (Ground Truth)', fontsize=16, fontweight='bold')
    ax1.axis('off')

    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('ID Klasy', fontsize=12)

    plt.tight_layout()
    plt.savefig('plots/tsne_1_rzeczywiste.png', bbox_inches='tight')
    print("   ✅ Zapisano: plots/tsne_1_rzeczywiste.png")
    plt.close()

    # =========================================================
    # WYKRES 2: PREDYKCJE MODELU (Jak model widzi świat)
    # =========================================================
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    scatter2 = ax2.scatter(z[:, 0], z[:, 1], c=pred_sample, cmap=cmap, s=20, alpha=0.7)

    ax2.set_title('t-SNE: Klasy przewidziane przez model', fontsize=16, fontweight='bold')
    ax2.axis('off')

    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('ID Klasy (Predykcja)', fontsize=12)

    plt.tight_layout()
    plt.savefig('plots/tsne_2_predykcje.png', bbox_inches='tight')
    print("   ✅ Zapisano: plots/tsne_2_predykcje.png")
    plt.close()

    # =========================================================
    # WYKRES 3: POPRAWNE VS BŁĘDNE (Analiza pomyłek)
    # =========================================================
    fig3, ax3 = plt.subplots(figsize=(12, 10))

    # Najpierw rysujemy poprawne (zielone) jako tło
    ax3.scatter(z[correct_sample, 0], z[correct_sample, 1],
                c='#2ECC71', s=20, alpha=0.4, label='Poprawne')

    # Na wierzchu rysujemy błędne (czerwone)
    # ZMIANA: Usunięto marker='x', teraz są to domyślne kropki
    ax3.scatter(z[~correct_sample, 0], z[~correct_sample, 1],
                c='#E74C3C', s=25, alpha=0.9, label='Błędne')

    ax3.set_title('t-SNE: Mapa błędów (Poprawne vs Błędne)', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12, markerscale=2)
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('plots/tsne_3_bledy.png', bbox_inches='tight')
    print("   ✅ Zapisano: plots/tsne_3_bledy.png")
    plt.close()


def plot_attention_heatmap(model, data, device):
    """[5/10] Wizualizacja atencji."""
    print("\n" + "=" * 60)
    print("5. HEATMAPA ATENCJI GAT")
    print("=" * 60)

    data_cpu = data.cpu()
    degrees = data_cpu.edge_index[0].bincount()
    candidates = ((degrees > 12) & (degrees < 20)).nonzero(as_tuple=True)[0]
    node_idx = candidates[0].item() if len(candidates) > 0 else 0

    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx, 1, data_cpu.edge_index, relabel_nodes=True
    )

    x_sub = data_cpu.x[subset].to(device)
    edge_index_sub = edge_index_sub.to(device)

    with torch.no_grad():
        _, (att_edge, att_val) = model.conv1(x_sub, edge_index_sub, return_attention_weights=True)

    att_val = att_val.mean(dim=1).cpu().numpy()
    src, dst = att_edge.cpu().numpy()

    G = nx.Graph()
    center_id = mapping.item()
    for s, d, w in zip(src, dst, att_val):
        if d == center_id and s != center_id:
            G.add_edge(s, d, weight=w)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [(w / max_w) * 5 + 0.5 for w in weights]

    nx.draw_networkx_nodes(G, pos, nodelist=[center_id], node_color='#E74C3C',
                           node_size=800, label='Analizowany artykuł')
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if n != center_id],
                           node_color='#3498DB', node_size=400, label='Sąsiedzi')
    edges = nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.Reds,
                                   width=widths, edge_vmin=0, edge_vmax=max_w)

    plt.colorbar(edges, label='Waga Atencji')
    plt.legend(loc='upper left')
    plt.title(f'Mechanizm Atencji (Węzeł {node_idx})', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/attention_heatmap.png')
    print("Zapisano: plots/attention_heatmap.png")
    plt.close()


def plot_top_confused_pairs(cm, label_map):
    """[6/10] Top 15 mylonych par z nazwami (Wersja HD)."""
    print("\n" + "=" * 60)
    print("6. TOP MYLONE PARY KLAS (HD)")
    print("=" * 60)

    # Przygotowanie danych (bez zmian logicznych)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    flat_indices = np.argsort(cm_no_diag.ravel())[::-1][:15]
    pairs = []

    for idx in flat_indices:
        true_class = idx // cm.shape[1]
        pred_class = idx % cm.shape[1]
        count = cm_no_diag[true_class, pred_class]
        if count > 0:
            pairs.append((true_class, pred_class, count))

    df_pairs = pd.DataFrame(pairs, columns=['Prawdziwa', 'Predykcja', 'Liczba'])
    # Dodajemy strzałkę, żeby było ładniej
    df_pairs['Para'] = df_pairs.apply(
        lambda x: f"{label_map[int(x['Prawdziwa'])]}  ⮕  {label_map[int(x['Predykcja'])]}", axis=1
    )

    # Zmiana: Wykres szeroki (dla tekstu) i wysoki (dla słupków)
    plt.figure(figsize=(18, 12))

    colors = sns.color_palette("Reds_r", len(df_pairs))
    bars = plt.barh(df_pairs['Para'], df_pairs['Liczba'], color=colors, edgecolor='black', height=0.7)

    plt.xlabel('Liczba błędnych klasyfikacji', fontsize=20, labelpad=15)
    plt.ylabel('Para klas (Prawdziwa ⮕ Predykcja)', fontsize=20, labelpad=15)
    plt.title('Top 15 Najczęściej Mylonych Par Klas', fontsize=26, fontweight='bold', pad=20)

    # Odwrócenie osi (najczęstsze na górze)
    plt.gca().invert_yaxis()

    # Zmiana: Duże czcionki na osiach
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)  # To najważniejsze - nazwy par będą duże

    # Wartości liczbowe obok słupków
    for bar, val in zip(bars, df_pairs['Liczba']):
        plt.text(bar.get_width() + (max(df_pairs['Liczba']) * 0.01),
                 bar.get_y() + bar.get_height() / 2,
                 f'{int(val)}', va='center', fontsize=16, fontweight='bold', color='black')

    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('plots/top_confused_pairs.png', bbox_inches='tight')
    print("   ✅ Zapisano: plots/top_confused_pairs.png")
    plt.close()


def plot_summary_dashboard(y_true, y_pred, probs, label_map):
    """[10/10] Generowanie 4 osobnych wykresów podsumowujących."""
    print("\n" + "=" * 60)
    print("10. GENEROWANIE WYKRESÓW PODSUMOWUJĄCYCH (OSOBNE PLIKI)")
    print("=" * 60)

    # --- OBLICZENIA WSPÓLNE ---
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    top3_acc = top_k_accuracy_score(y_true, probs, k=3)
    top5_acc = top_k_accuracy_score(y_true, probs, k=5)
    correct = y_true == y_pred

    # Przygotowanie danych do rozkładu
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    labels = [label_map[i] for i in range(40)]
    x = np.arange(40)

    # Obliczenie entropii
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

    # ==========================================
    # WYKRES 1: KLUCZOWE METRYKI
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    metrics = ['Accuracy', 'Balanced Acc', 'Top-3 Acc', 'Top-5 Acc']
    values = [acc, balanced_acc, top3_acc, top5_acc]
    colors = ['#2ECC71' if v > 0.7 else '#F39C12' if v > 0.5 else '#E74C3C' for v in values]

    bars = ax1.barh(metrics, values, color=colors, edgecolor='black', height=0.6)
    ax1.set_xlim(0, 1.05)  # Lekki margines dla tekstu
    ax1.set_title('Kluczowe Metryki', fontsize=16, fontweight='bold')

    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/1_metryki.png', bbox_inches='tight')
    print("Zapisano: plots/1_metryki.png")
    plt.close()

    # ==========================================
    # WYKRES 2: ROZKŁAD KLAS (LOG)
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(16, 8))  # Szeroki wykres dla czytelności etykiet

    ax2.bar(x - 0.2, true_counts.reindex(range(40), fill_value=0), 0.4, label='Prawdziwe', alpha=0.7, color='#3498DB')
    ax2.bar(x + 0.2, pred_counts.reindex(range(40), fill_value=0), 0.4, label='Predykcje', alpha=0.7, color='#E74C3C')

    ax2.set_ylabel('Liczba próbek (skala logarytmiczna)')
    ax2.set_title('Rozkład klas i predykcji', fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=10)  # Większa czcionka
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/2_rozklad_klas.png', bbox_inches='tight')
    print("Zapisano: plots/2_rozklad_klas.png")
    plt.close()

    # ==========================================
    # WYKRES 3: POPRAWNOŚĆ (PIE CHART)
    # ==========================================
    fig3, ax3 = plt.subplots(figsize=(8, 8))  # Kwadratowy

    correct_count = correct.sum()
    wrong_count = len(correct) - correct_count

    ax3.pie([correct_count, wrong_count],
            labels=[f'Poprawne\n({correct_count:,})', f'Błędne\n({wrong_count:,})'],
            colors=['#2ECC71', '#E74C3C'],
            autopct='%1.1f%%', explode=(0.03, 0), shadow=True,
            textprops={'fontsize': 14, 'fontweight': 'bold'})

    ax3.set_title(f'Poprawność predykcji', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/3_poprawnosc_kolowy.png', bbox_inches='tight')
    print("Zapisano: plots/3_poprawnosc_kolowy.png")
    plt.close()

    # ==========================================
    # WYKRES 4: ENTROPIA (PEWNOŚĆ SIEBIE)
    # ==========================================
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    ax4.hist(entropy[correct], bins=50, alpha=0.7, label='Poprawne (Pewne)', color='#2ECC71', density=True)
    ax4.hist(entropy[~correct], bins=50, alpha=0.7, label='Błędne (Niepewne)', color='#E74C3C', density=True)

    ax4.set_xlabel('Entropia predykcji (0 = duża pewność, wysoka = niepewność)')
    ax4.set_ylabel('Gęstość')
    ax4.set_title('Rozkład pewności siebie modelu (Entropia)', fontsize=16, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/4_entropia.png', bbox_inches='tight')
    print("Zapisano: plots/4_entropia.png")
    plt.close()


def plot_combined_distribution_accuracy(y_true, y_pred, label_map):
    """
    [BONUS] Połączony wykres: Rozkład klas (góra) + Dokładność (dół).
    Słupki są idealnie w jednej linii pionowej.
    """
    print("\n" + "=" * 60)
    print("BONUS. GENEROWANIE POŁĄCZONEGO WYKRESU (ROZKŁAD + ACC)")
    print("=" * 60)

    # --- PRZYGOTOWANIE DANYCH ---
    # 1. Rozkład (Góra)
    true_counts = pd.Series(y_true).value_counts().sort_index()

    # 2. Dokładność (Dół)
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = cm.diagonal() / cm.sum(axis=1)
        class_acc = np.nan_to_num(class_acc)

    # Etykiety
    labels = [label_map[i] for i in range(40)]
    x = np.arange(40)

    # --- TWORZENIE FIGURY ---
    # figsize=(24, 16) - Wysoki i szeroki
    # sharex=True - KLUCZOWE: Wspólna oś X dla obu wykresów
    # gridspec_kw - Ustalamy proporcje wysokości (np. góra 1: dół 1.5 lub 1:1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 18), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1.2]})

    # ==========================================
    # WYKRES GÓRNY: ROZKŁAD LICZNOŚCI
    # ==========================================
    # Rysujemy tylko prawdziwe liczności (niebieskie), bez predykcji
    ax1.bar(x, true_counts.reindex(range(40), fill_value=0),
            color='#3498DB', alpha=0.8, edgecolor='black', width=0.6)

    ax1.set_yscale('log')  # Skala logarytmiczna
    ax1.set_ylabel('Liczba próbek (log)', fontsize=18, labelpad=15)
    ax1.set_title('Korelacja: Liczność klasy vs Skuteczność modelu', fontsize=26, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Dodajemy legendę tylko dla góry
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='#3498DB', label='Liczba artykułów w zbiorze')
    ax1.legend(handles=[blue_patch], fontsize=16, loc='upper right')

    # ==========================================
    # WYKRES DOLNY: DOKŁADNOŚĆ (ACCURACY)
    # ==========================================
    colors = ['#2ECC71' if a > 0.7 else '#F39C12' if a > 0.5 else '#E74C3C' for a in class_acc]

    ax2.bar(x, class_acc, color=colors, edgecolor='black', alpha=0.9, width=0.6)

    # Linia średniej
    mean_val = np.mean(class_acc)
    ax2.axhline(y=mean_val, color='red', linestyle='--', linewidth=3, label=f'Średnia: {mean_val:.3f}')

    ax2.set_ylabel('Dokładność (Accuracy)', fontsize=18, labelpad=15)
    ax2.set_xlabel('Kategoria arXiv', fontsize=18, labelpad=15)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=16, loc='upper right')

    # ==========================================
    # WSPÓLNA OŚ X (Labels)
    # ==========================================
    # Ustawiamy etykiety tylko na dolnym wykresie (dzięki sharex góra ich nie ma)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=14)

    # Zmniejszamy odstęp między wykresami, żeby wyglądały jak całość
    plt.subplots_adjust(hspace=0.05)

    plt.savefig('plots/combined_distribution_accuracy.png', bbox_inches='tight')
    print("   ✅ Zapisano: plots/combined_distribution_accuracy.png")
    plt.close()


def plot_class_f1_score(y_true, y_pred, label_map):
    """[BONUS] Wynik F1 dla każdej klasy (lepsze dla niezbalansowanych danych)."""
    print("\n" + "=" * 60)
    print("BONUS. F1-SCORE PER KLASA (HD)")
    print("=" * 60)

    # Obliczamy F1 dla każdej klasy osobno (average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    labels = [label_map[i] for i in range(len(f1_per_class))]

    plt.figure(figsize=(24, 12))
    colors = ['#2ECC71' if a > 0.7 else '#F39C12' if a > 0.5 else '#E74C3C' for a in f1_per_class]

    bars = plt.bar(range(len(f1_per_class)), f1_per_class, color=colors, edgecolor='black', alpha=0.8, width=0.7)

    # Linia średniej (Macro F1)
    mean_val = np.mean(f1_per_class)
    plt.axhline(y=mean_val, color='blue', linestyle='--', linewidth=3,
                label=f'Macro F1 (Średnia): {mean_val:.3f}')

    plt.title('Wynik F1-Score dla poszczególnych kategorii (Harmoniczna Precyzji i Czułości)', fontsize=28,
              fontweight='bold', pad=20)
    plt.ylabel('F1-Score', fontsize=22)
    plt.xlabel('Kategoria arXiv', fontsize=22)

    plt.xticks(range(len(f1_per_class)), labels, rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(-1, len(f1_per_class))
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.legend(fontsize=18, loc='upper right')

    plt.tight_layout()
    plt.savefig('plots/class_f1_score.png', bbox_inches='tight')
    print("   ✅ Zapisano: plots/class_f1_score.png")
    plt.close()
# =============================================================================
# MAIN
# =============================================================================

def load_model_and_data():
    apply_patches()
    device = get_device()
    dataset = PygNodePropPredDataset('ogbn-arxiv', './data', transform=T.ToUndirected())
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    model = GAT(data.num_features, 96, dataset.num_classes, heads=8, dropout=0.3).to(device)

    model_path = 'models/gat_model.pth'
    if not os.path.exists(model_path):
        model_path = 'models/best_gat_model.pth'

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, data, dataset, split_idx, device


def run_inference(model, data, split_idx, device):
    test_loader = NeighborLoader(
        data, num_neighbors=[20, 15, 10], input_nodes=split_idx['test'],
        batch_size=1024, num_workers=0, shuffle=False
    )
    y_true, y_pred, logits = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            y_true.append(batch.y[:batch.batch_size].cpu())
            y_pred.append(out.argmax(dim=-1).cpu())
            logits.append(out.cpu())
    return (torch.cat(y_true).numpy().flatten(),
            torch.cat(y_pred).numpy().flatten(),
            torch.softmax(torch.cat(logits), dim=1).numpy(),
            torch.cat(logits).numpy())


def main():
    print("\n" + "=" * 70)
    print("GENEROWANIE WIZUALIZACJI (FOLDER PLOTS/)")
    print("=" * 70)

    label_map = load_category_mapping()
    plot_learning_curves()

    print("\nŁadowanie modelu i danych...")
    model, data, dataset, split_idx, device = load_model_and_data()

    print("\nInferencja...")
    y_true, y_pred, probs, embeddings = run_inference(model, data, split_idx, device)

    # Generowanie wykresów
    cm = plot_confusion_matrix(y_true, y_pred, label_map)
    plot_class_accuracy(cm, label_map)
    plot_tsne(embeddings, y_true, y_pred)
    plot_attention_heatmap(model, data, device)
    plot_top_confused_pairs(cm, label_map)
    plot_summary_dashboard(y_true, y_pred, probs, label_map)
    plot_combined_distribution_accuracy(y_true, y_pred, label_map)
    plot_class_f1_score(y_true, y_pred, label_map)
    print("\n" + "=" * 70)
    print("ZAKOŃCZONO! Sprawdź folder 'plots/'")
    print("=" * 70)


if __name__ == "__main__":
    main()