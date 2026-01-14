import os
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# Importy z własnych modułów projektu
from src.model import GAT
from src.engine import evaluate
from src.utils import apply_patches, get_device


def run_evaluation():
    """
    Główna funkcja ewaluacyjna.
    Ładuje najlepszy zapisany model i sprawdza jego dokładność (Accuracy) na zbiorze testowym.
    """

    # 1. Konfiguracja środowiska
    apply_patches()
    device = get_device()
    print(f"--- Ewaluacja modelu na {device} ---")

    dataset = PygNodePropPredDataset('ogbn-arxiv', './data', transform=T.ToUndirected())
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    test_loader = NeighborLoader(
        data, num_neighbors=[20, 15, 10],
        input_nodes=split_idx['test'],
        batch_size=2048, num_workers=0
    )

    model = GAT(data.num_features, 96, dataset.num_classes,8,0.3).to(device)

    model_path = 'models/gat_model.pth'
    if not os.path.exists(model_path):
        print("Nie znaleziono modelu! Uruchom najpierw train.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model wczytany.")

    evaluator = Evaluator('ogbn-arxiv')
    acc = evaluate(model, test_loader, evaluator, device)

    print(f"\nWynik (Accuracy) na zbiorze testowym: {acc:.4f}")


if __name__ == "__main__":
    run_evaluation()