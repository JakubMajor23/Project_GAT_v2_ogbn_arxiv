import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm, Linear


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
        # Zwracamy surowe logity (bez Softmax), ponieważ CrossEntropyLoss
        # w PyTorch oblicza Softmax wewnętrznie.
        x = self.conv3(x, edge_index)
        return x