import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm, Linear
import sys

# ==============================================================================
# 1. NAPRAWA KOMPATYBILNOŚCI (FIX PYTORCH 2.6+)
# ==============================================================================
# Zapewnia, że skrypt otworzy plik .pt nawet na nowszych wersjach biblioteki
try:
    _original_torch_load = torch.load


    def safe_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)


    torch.load = safe_torch_load
except:
    pass


# ==============================================================================
# 2. TWOJA KLASA MODELU (Dokładnie ta, którą podałeś)
# ==============================================================================
class GAT(torch.nn.Module):
    """
        Implementacja sieci GATv2 (Graph Attention Network v2) z wykorzystaniem
        połączeń rezydualnych (Residual Connections) i normalizacji warstwowej.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        self.dropout = dropout
        self.hid = hidden_channels * heads

        # --- PROJEKCJA WEJŚCIOWA (Input Projection) ---
        self.lin_in = Linear(in_channels, self.hid)

        # --- WARSTWA 1: GATv2 (Wejściowa) ---
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.norm1 = LayerNorm(self.hid)

        # --- WARSTWA 2: GATv2 (Ukryta) ---
        self.conv2 = GATv2Conv(self.hid, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.norm2 = LayerNorm(self.hid)

        # --- WARSTWA 3: WYJŚCIOWA (Klasyfikator) ---
        self.conv3 = GATv2Conv(self.hid, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # --- BLOK 1 ---
        # 1. Projekcja wejścia do wymiaru ukrytego (dla skip connection)
        identity = self.lin_in(x)

        # 2. Agregacja atencyjna (Message Passing)
        out = self.conv1(x, edge_index)

        # 3. Post-processing: Normalizacja -> Aktywacja -> Dropout
        out = self.norm1(out)
        out = F.gelu(out)  # GELU działa lepiej niż ReLU w głębokich sieciach
        out = F.dropout(out, p=self.dropout, training=self.training)

        # 4. Połączenie rezydualne (Dodanie wejścia do wyjścia)
        x = out + identity

        # --- BLOK 2 ---
        # W tym bloku 'x' ma już odpowiedni wymiar, więc identity to po prostu 'x'
        identity = x

        out = self.conv2(x, edge_index)
        out = self.norm2(out)
        out = F.gelu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        x = out + identity

        # --- BLOK WYJŚCIOWY ---
        x = self.conv3(x, edge_index)
        return x


# ==============================================================================
# 3. MAPOWANIE KATEGORII (Hardcoded - dla ładnego wyświetlania)
# ==============================================================================
# Nazwy kategorii arXiv (zgodne z ogbn-arxiv)
ARXIV_MAP = {
    0: 'cs.AI (Artificial Intelligence)', 1: 'cs.AR (Hardware)', 2: 'cs.CC (Complexity)',
    3: 'cs.CE (Computational Eng.)', 4: 'cs.CG (Geometry)', 5: 'cs.CL (Computation and Language)',
    6: 'cs.CR (Cryptography)', 7: 'cs.CV (Computer Vision)', 8: 'cs.CY (Computers and Society)',
    9: 'cs.DB (Databases)', 10: 'cs.DC (Distributed/Parallel)', 11: 'cs.DL (Digital Libraries)',
    12: 'cs.DM (Discrete Math)', 13: 'cs.DS (Data Structures)', 14: 'cs.ET (Emerging Tech)',
    15: 'cs.FL (Formal Languages)', 16: 'cs.GL (General Lit.)', 17: 'cs.GR (Graphics)',
    18: 'cs.GT (Game Theory)', 19: 'cs.HC (Human-Computer)', 20: 'cs.IR (Info Retrieval)',
    21: 'cs.IT (Info Theory)', 22: 'cs.LG (Machine Learning)', 23: 'cs.LO (Logic)',
    24: 'cs.MA (Multiagent Sys)', 25: 'cs.MM (Multimedia)', 26: 'cs.MS (Mathematical Soft)',
    27: 'cs.NA (Numerical Analysis)', 28: 'cs.NE (Neural & Evol)', 29: 'cs.NI (Networking)',
    30: 'cs.OH (Other)', 31: 'cs.OS (Operating Sys)', 32: 'cs.PF (Performance)',
    33: 'cs.PL (Prog. Languages)', 34: 'cs.RO (Robotics)', 35: 'cs.SC (Symbolic Comp)',
    36: 'cs.SD (Sound)', 37: 'cs.SE (Software Eng.)', 38: 'cs.SI (Social & Info)',
    39: 'cs.SY (Systems & Control)'
}


# ==============================================================================
# 4. GŁÓWNA PĘTLA DEMONSTRACYJNA
# ==============================================================================
def main():
    print("\n" + "=" * 80)
    print("  DEMONSTRACJA LIVE: KLASYFIKACJA WĘZŁÓW W SIECI CYTOWAŃ (ARCHITEKTURA GATv2)")
    print("=" * 80)

    # 1. Ładowanie próbki
    print(">>> 1. Wczytywanie wycinka grafu (demo_samples.pt)...")
    try:
        package = torch.load('demo_samples.pt')
        samples = package['samples']
        # Używamy naszego wbudowanego mapowania, bo jest czytelniejsze
        idx_to_class = ARXIV_MAP
    except FileNotFoundError:
        print("\n[BŁĄD]: Nie znaleziono pliku 'demo_samples.pt'.")
        print("Uruchom najpierw skrypt 'prepare_demo.py', aby przygotować dane!")
        return

    # 2. Konfiguracja i ładowanie modelu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> 2. Inicjalizacja modelu GATv2 na urządzeniu: {device}")

    # PARAMETRY MODELU - MUSZĄ PASOWAĆ DO TRENINGU
    # Zgodnie z Twoim sprawozdaniem: in=128, out=40, heads=8, hidden_dim=768.
    # Skoro w kodzie self.hid = hidden_channels * heads, to:
    # hidden_channels (na głowicę) = 768 / 8 = 96.
    model = GAT(in_channels=128, hidden_channels=96, out_channels=40, heads=8, dropout=0.3)

    model_path = 'models/gat_model.pth'
    try:
        # strict=False pozwala pominąć drobne różnice w nazwach, ale lepiej żeby pasowało idealnie
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f">>> 3. Pomyślnie załadowano wagi z: {model_path}")
    except FileNotFoundError:
        print(f"\n[BŁĄD]: Nie znaleziono modelu w '{model_path}'. Sprawdź folder 'models/'.")
        return
    except Exception as e:
        print(f"\n[BŁĄD KRYTYCZNY MODELU]: {e}")
        print("Upewnij się, że parametry w init (96, 8 heads) zgadzają się z tymi z treningu.")
        return

    model.to(device)
    model.eval()  # Wyłączenie dropoutu i batchnorm/layernorm w tryb testowy

    # 3. Prezentacja wyników
    print("\n" + "=" * 80)
    print("WYNIKI PREDYKCJI DLA WYBRANYCH ARTYKUŁÓW NAUKOWYCH")
    print("=" * 80)

    for i, data in enumerate(samples):
        data = data.to(device)

        # --- INFERENCJA (URUCHOMIENIE MODELU) ---
        with torch.no_grad():
            # Model otrzymuje cechy węzłów i strukturę połączeń (cytowania)
            out = model(data.x, data.edge_index)

            # Obliczamy prawdopodobieństwa
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)

        # --- WYCIĄGANIE DANYCH DLA WĘZŁA CENTRALNEGO ---
        center_idx = data.center_idx_mapping

        true_y = data.y[center_idx].item()
        pred_y = preds[center_idx].item()
        confidence = probs[center_idx][pred_y].item() * 100

        true_name = idx_to_class.get(true_y, f"Class {true_y}")
        pred_name = idx_to_class.get(pred_y, f"Class {pred_y}")

        # Kolorowanie wyniku
        if true_y == pred_y:
            result_color = "\033[92m"  # Zielony
            status = "POPRAWNIE"
        else:
            result_color = "\033[91m"  # Czerwony
            status = "BŁĄD"
        reset = "\033[0m"

        # --- WYPISANIE ---
        print(f"PRZYKŁAD #{i + 1} | ID Artykułu: {data.original_id}")
        print("-" * 60)
        print(f"Liczba analizowanych cytowań (sąsiadów): {data.num_nodes - 1}")
        print(f"Prawdziwa kategoria:     \033[94m{true_name}{reset}")
        print(f"Predykcja modelu GATv2:  {result_color}{pred_name}{reset}")
        print(f"Pewność modelu:          {confidence:.2f}%")
        print(f"Status klasyfikacji:     [{result_color}{status}{reset}]")
        print("=" * 80 + "\n")

    input("Naciśnij ENTER, aby zakończyć prezentację...")


if __name__ == "__main__":
    main()