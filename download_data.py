import os
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from src.utils import apply_patches

ROOT_DIR = './data'

def download_data():
    apply_patches()
    print("\n=== KROK 1: Pobieranie danych OGBN-Arxiv ===")
    PygNodePropPredDataset(name='ogbn-arxiv', root=ROOT_DIR, transform=T.ToUndirected())
    print("-> Dane zostały pobrane/zweryfikowane.")

def generate_csv_samples():
    print("\n=== KROK 2: Generowanie Próbek CSV ===")
    raw_dir = os.path.join(ROOT_DIR, 'ogbn_arxiv', 'raw')
    mapping_dir = os.path.join(ROOT_DIR, 'ogbn_arxiv', 'mapping')
    output_dir = os.path.join(ROOT_DIR, 'csv_samples')
    os.makedirs(output_dir, exist_ok=True)

    raw_files = ['edge.csv.gz', 'node-feat.csv.gz', 'node-label.csv.gz', 'node_year.csv.gz']

    for filename in raw_files:
        full_path = os.path.join(raw_dir, filename)
        
        if not os.path.exists(full_path):
            print(f"BŁĄD: Nie znaleziono pliku źródłowego: {filename}")
            continue

        print(f"-> Przetwarzanie: {filename}...")
        try:
            df = pd.read_csv(full_path, compression='gzip', nrows=10, header=None)
            output_name = filename.replace('.gz', '')
            df.to_csv(os.path.join(output_dir, output_name), index=False, header=False)
        except Exception as e:
            print(f"   Błąd podczas przetwarzania {filename}: {e}")

    mapping_file = 'labelidx2arxivcategeory.csv.gz'
    mapping_path = os.path.join(mapping_dir, mapping_file)

    print(f"-> Przetwarzanie: {mapping_file}...")
    
    try:
        df_mapping = pd.read_csv(mapping_path, compression='gzip')
        df_mapping.to_csv(os.path.join(output_dir, 'categories_mapping.csv'), index=False)

        print("-> Przetwarzanie: categories_mapping.csv...")

        labels_sample_path = os.path.join(output_dir, 'node-label.csv')
        
        if os.path.exists(labels_sample_path):
            df_labels = pd.read_csv(labels_sample_path, header=None, names=['label_idx'])
            df_human = df_labels.merge(df_mapping, left_on='label_idx', right_on='label idx', how='left')
            df_human = df_human[['label_idx', 'arxiv category']]
            df_human.columns = ['ID_Kategorii', 'Nazwa_Kategorii']
            df_human.insert(0, 'ID_Węzła', range(len(df_human)))
            
            human_path = os.path.join(output_dir, 'human_readable_labels.csv')
            df_human.to_csv(human_path, index=False)
        else:
            print("   UWAGA: Nie znaleziono 'node-label.csv' w folderze próbek. Pomijam human_readable.")

    except Exception as e:
        print(f"   Błąd przy przetwarzaniu kategorii: {e}")

if __name__ == "__main__":
    download_data()
    generate_csv_samples()
    print("\n=== ZAKOŃCZONO ===")