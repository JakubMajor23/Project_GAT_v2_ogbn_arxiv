### 1. Instalacja zależności

Na samym początku zalecane jest utworzenie i aktywacja wirtualnego środowiska (`venv`):

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Wymagania
- Python 3.13.7 (Kod testowany wyłącznie na tej wersji)

### 2. Pobranie danych

Skrypt automatycznie pobierze i przygotuje dane w folderze `data/`.

```bash
python download_data.py
```

### 3. Użycie projektu

Trzy główne tryby działania:

#### A. Uruchomienie DEMO
Jeśli chcesz zobaczyć jak model działa w praktyce, uruchom skrypt demo. Pozwoli on na przetestowanie klasyfikacji na losowych próbkach z datasetu i zweryfikowanie ich z API OpenAlex/ArXiv.

```bash
python demo.py
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
