import torch
import torch.nn.functional as F


def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    """
    Funkcja wykonująca jedną pełną epokę treningową.
    Oblicza gradienty i aktualizuje wagi modelu.
    """
    model.train()  # Przełączenie w tryb treningowy (aktywuje Dropout i BatchNorm)
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch in loader:
        batch = batch.to(device)

        # Resetowanie gradientów (set_to_none=True jest szybsze niż =0)
        optimizer.zero_grad(set_to_none=True)

        # --- MIXED PRECISION (AMP) ---
        # Używamy automatycznej precyzji mieszanej (FP16 + FP32).
        # Przyspiesza to trening na GPU (rdzenie Tensor Cores) i zmniejsza zużycie VRAM.
        with torch.amp.autocast('cuda'):
            # Wykonanie modelu (Forward Pass)
            # WAŻNE: NeighborLoader zwraca węzły docelowe + ich sąsiadów.
            # Nas interesuje predykcja TYLKO dla węzłów docelowych (pierwsze 'batch_size' wierszy).
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].squeeze()

            # Obliczenie funkcji kosztu z Label Smoothing (0.1)
            # Pomaga to zapobiegać nadmiernej pewności siebie modelu (overconfidence).
            loss = F.cross_entropy(out, y, label_smoothing=0.1)

        # --- BACKPROPAGATION Z AMP ---
        # Skalujemy stratę, aby uniknąć zanikania gradientów (underflow) przy precyzji FP16.
        scaler.scale(loss).backward()

        # Od-skalowanie gradientów przed aktualizacją wag
        scaler.unscale_(optimizer)

        # Gradient Clipping: Przycinamy normę gradientu do 1.0.
        # Zapobiega to "wybuchającym gradientom", co jest częste w sieciach GNN/GAT.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Aktualizacja wag modelu
        scaler.step(optimizer)
        scaler.update()

        # Aktualizacja Learning Rate
        # OneCycleLR wymaga aktualizacji co krok (batch), a nie co epokę.
        scheduler.step()

        # Agregacja statystyk (do logowania)
        total_loss += loss.item() * batch.batch_size
        total_correct += (out.argmax(dim=-1) == y).sum().item()
        total_samples += batch.batch_size

    # Zwracamy średnią stratę i dokładność z całej epoki
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()  # Wyłącza obliczanie gradientów (oszczędność pamięci i czasu podczas testów)
def evaluate(model, loader, evaluator, device):
    """
    Funkcja ewaluacyjna.
    Przechodzi przez cały zbiór danych (walidacyjny lub testowy) bez uczenia.
    """
    model.eval()  # Przełączenie w tryb ewaluacji (wyłącza Dropout)
    y_true, y_pred = [], []

    for batch in loader:
        batch = batch.to(device)

        # Nawet przy ewaluacji używamy AMP dla przyspieszenia inferencji
        with torch.amp.autocast('cuda'):
            # Podobnie jak w treningu, bierzemy tylko węzły docelowe z batcha
            out = model(batch.x, batch.edge_index)[:batch.batch_size]

        # Przenosimy wyniki na CPU, aby nie zapchać pamięci GPU (VRAM)
        # przy zbieraniu wyników z całego, dużego grafu.
        y_true.append(batch.y[:batch.batch_size].cpu())
        y_pred.append(out.argmax(dim=-1, keepdim=True).cpu())

    # Sklejamy wyniki ze wszystkich batchy i obliczamy metrykę OGB (Accuracy)
    return evaluator.eval({
        'y_true': torch.cat(y_true),
        'y_pred': torch.cat(y_pred)
    })['acc']