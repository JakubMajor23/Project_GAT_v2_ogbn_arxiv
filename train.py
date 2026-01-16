import os
import time
import torch
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


from src.model import GAT
from src.engine import train_epoch, evaluate
from src.utils import set_seed, count_parameters, apply_patches, get_device

torch.set_float32_matmul_precision('medium')


def main():
    apply_patches()
    set_seed(42)
    device = get_device()

    print(f"\n{'=' * 60}")
    print(f"GAT (Standard) - OGBN-ARXIV TRAINING")
    print(f"{'=' * 60}")

    # 1. Dane
    dataset = PygNodePropPredDataset('ogbn-arxiv', './data', transform=T.ToUndirected())
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    train_loader = NeighborLoader(
        data, num_neighbors=[20, 15, 10], input_nodes=split_idx['train'],
        batch_size=1024, shuffle=True, num_workers=2, persistent_workers=True
    )
    val_loader = NeighborLoader(
        data, num_neighbors=[20, 15, 10], input_nodes=split_idx['valid'],
        batch_size=2048, num_workers=2, persistent_workers=True
    )
    test_loader = NeighborLoader(
        data, num_neighbors=[20, 15, 10], input_nodes=split_idx['test'],
        batch_size=2048, num_workers=2, persistent_workers=True
    )

    # 2. Model
    model = GAT(
        in_channels=data.num_features,
        hidden_channels=96,
        out_channels=dataset.num_classes,
        heads=8,
        dropout=0.4
    ).to(device)

    print(f"Model structure: GATv2")
    print(f"Liczba parametrów: {count_parameters(model):,}")
    print(f"Device: {device}")

    # 3. Optymalizacja
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    MAX_EPOCHS = 300
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.003, epochs=MAX_EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.15, anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler('cuda')
    evaluator = Evaluator('ogbn-arxiv')

    # 4. Pętla treningowa
    MODEL_FILENAME = 'gatv2_model.pth'

    patience = 50
    best_val = 0
    no_improve = 0

    history = []

    print(f"Start training...\n")
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        val_acc = evaluate(model, val_loader, evaluator, device)


        elapsed_seconds = time.time() - start_time
        time_str = f"{elapsed_seconds / 60:.1f}m"

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), MODEL_FILENAME)
            no_improve = 0
            action_str = "SAVE"
        else:
            no_improve += 1
            action_str = f"Stop: {no_improve}/{patience}"

        print(f"Ep {epoch:03d} | {time_str:>6} | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | {action_str}")

        history.append({
            'Epoch': epoch,
            'Time': time_str,
            'Loss': round(loss, 4),
            'Train_Acc': round(train_acc, 4),
            'Val_Acc': round(val_acc, 4),
            'Action': action_str
        })

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n{'=' * 60}")

    pd.DataFrame(history).to_csv('logs.csv', index=False)

    if os.path.exists(MODEL_FILENAME):
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
        print(f"Wczytano najlepszy model z {MODEL_FILENAME}")

    final_test = evaluate(model, test_loader, evaluator, device)
    print(f"Final Test Acc: {final_test:.4f}")


if __name__ == "__main__":
    main()