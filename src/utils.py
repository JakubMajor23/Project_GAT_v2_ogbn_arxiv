import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Gwarantuje powtarzalność wyników."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Ustawiono seed: {seed}")

def count_parameters(model):
    """Liczy liczbę trenowalnych parametrów."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_patches():
    """
        Rozwiązuje problem kompatybilności wczytywania danych w nowszych wersjach PyTorch (>=2.4).

        Nowe wersje PyTorch domyślnie ustawiają flagę `weights_only=True` w funkcji `torch.load`
        ze względów bezpieczeństwa, co uniemożliwia wczytanie starszych plików datasetu OGB
        (zawierających pełne obiekty Pythona, a nie tylko wagi).

        Ta funkcja nadpisuje `torch.load`, wymuszając `weights_only=False`, co pozwala
        na poprawne załadowanie grafu 'ogbn-arxiv' bez błędów.
        """
    _original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_load(*args, **kwargs)
    torch.load = patched_load

def get_device():
    """
    Sprawdza dostępność akceleratora GPU (CUDA). Jeśli jest dostępny, zwraca obiekt
    urządzenia 'cuda', co pozwala na znaczne przyspieszenie obliczeń macierzowych.
    W przeciwnym razie (brak karty NVIDIA lub sterowników) zwraca 'cpu'.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')