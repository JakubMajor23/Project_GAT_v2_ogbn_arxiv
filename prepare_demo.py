import torch
import pandas as pd
import os
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import k_hop_subgraph, degree
from torch_geometric.data import Data

# ==============================================================================
# FIX DLA PYTORCH 2.6+
# ==============================================================================
_original_torch_load = torch.load


def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = safe_torch_load


# ==============================================================================

def prepare_samples():
    data_root_dir = 'data'

    print(f">>> 1. Szukanie danych w katalogu '{data_root_dir}'...")
    try:
        # Wczytujemy dataset i symetryzujemy
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=data_root_dir, transform=T.ToUndirected())
        data = dataset[0]
        print(">>> SUKCES: Wczytano graf.")
    except Exception as e:
        print(f"BŁĄD: {e}")
        return

    # ---------------------------------------------------------
    # 2. POBRANIE PODZIAŁU NA ZBIORY (SPLITS)
    # ---------------------------------------------------------
    print(">>> 2. Pobieranie indeksów zbioru TESTOWEGO...")
    split_idx = dataset.get_idx_split()
    test_idx = split_idx['test']  # To jest tensor zawierający tylko ID z testu (rok 2019+)

    # Dodatkowo pobieramy lata publikacji dla weryfikacji wizualnej
    # (node_year jest częścią obiektu data w ogbn-arxiv, ale czasami jest oddzielnie)
    try:
        node_years = data.node_year.squeeze()
    except:
        node_years = None
        print("UWAGA: Nie udało się wczytać lat publikacji (node_year).")

    # ---------------------------------------------------------
    # 3. ŁADOWANIE MAPOWANIA
    # ---------------------------------------------------------
    mapping_dir = os.path.join(data_root_dir, 'ogbn_arxiv', 'mapping')
    mapping_file = 'labelidx2arxivcategory.csv.gz'
    mapping_path = os.path.join(mapping_dir, mapping_file)

    if not os.path.exists(mapping_path):
        mapping_path = os.path.join(mapping_dir, 'labelidx2arxivcategory.csv')

    if os.path.exists(mapping_path):
        label_map = pd.read_csv(mapping_path)
        idx_to_class = dict(zip(label_map['label idx'], label_map['arxiv category']))
    else:
        idx_to_class = {i: str(i) for i in range(40)}

    # ---------------------------------------------------------
    # 4. SZUKANIE CIEKAWYCH WĘZŁÓW TYLKO W ZBIORZE TESTOWYM
    # ---------------------------------------------------------
    print("\n>>> 3. Szukanie ciekawych węzłów w zbiorze TESTOWYM...")

    node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)

    candidates = []
    seen_classes = set()

    # Mieszamy indeksy TESTOWE
    shuffled_test_indices = test_idx[torch.randperm(test_idx.size(0))]

    # Sprawdzamy pierwsze 5000 wylosowanych węzłów TESTOWYCH
    for idx in shuffled_test_indices[:5000]:
        idx = idx.item()
        d = node_degrees[idx].item()
        c = data.y[idx].item()

        # Warunki: min 10 sąsiadów, max 30, unikalna klasa
        if d >= 10 and d <= 30 and c not in seen_classes:
            candidates.append(idx)
            seen_classes.add(c)

        if len(candidates) >= 3:
            break

    print(f">>> Wybrano węzły TESTOWE: {candidates}")

    saved_samples = []

    for center_id in candidates:
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=center_id,
            num_hops=1,
            edge_index=data.edge_index,
            relabel_nodes=True
        )

        sample_data = Data(
            x=data.x[subset],
            edge_index=sub_edge_index,
            y=data.y[subset],
            center_idx_mapping=mapping[0].item(),
            original_id=center_id,
            num_nodes=len(subset)
        )

        # Dodajemy rok publikacji do obiektu, żeby run_demo mogło go wyświetlić (opcjonalnie)
        if node_years is not None:
            sample_data.year = node_years[center_id].item()

        saved_samples.append(sample_data)

        # Info dla Ciebie
        cls_name = idx_to_class.get(data.y[center_id].item(), "Unknown")
        year_info = f"Rok: {sample_data.year}" if hasattr(sample_data, 'year') else ""
        print(f"   - ID: {center_id:<6} | {year_info} | Klasa: {cls_name:<15} | Sąsiadów: {len(subset) - 1}")

    # ---------------------------------------------------------
    # 5. ZAPIS
    # ---------------------------------------------------------
    package = {
        'samples': saved_samples,
        'class_mapping': idx_to_class
    }

    torch.save(package, 'demo_samples.pt')
    print(f"\n>>> SUKCES! Plik 'demo_samples.pt' zawiera węzły ze zbioru TESTOWEGO.")


if __name__ == "__main__":
    prepare_samples()