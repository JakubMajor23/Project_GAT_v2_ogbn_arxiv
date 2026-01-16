# Dokumentacja Projektu GAT v2 (OGBN-Arxiv)

## Wymagania
- **Python 3.11** 
- System: Windows


### 1. Instalacja zależności

Na samym początku zalecane jest utworzenie i aktywacja wirtualnego środowiska (`venv`):

```bash
# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\activate
```

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

pip install -r requirements.txt
```

### 2. Pobranie danych

Skrypt automatycznie pobierze i przygotuje dane w folderze `data/`.

```bash
python download_data.py
```

### 3. Użycie projektu

Trzy główne tryby działania:

#### A. Uruchomienie infer
Jeśli chcesz zobaczyć jak model działa w praktyce, uruchom skrypt demo. Pozwoli on na przetestowanie klasyfikacji na losowych próbkach z datasetu i zweryfikowanie ich z API OpenAlex/ArXiv.

```bash
python infer.py
```

#### B. Sprawdzenie skuteczności (Evaluate)
Aby sprawdzić dokładność (Accuracy) zapisanego modelu na zbiorze testowym, uruchom:

```bash
python evaluate.py
```

#### C. Trenowanie modelu (Train)
Jeżeli chcesz wytrenować model od zera uruchom:

```bash
python train.py
```
