import os
import torch
import pandas as pd
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import k_hop_subgraph, degree
from torch_geometric.data import Data
from urllib.parse import quote
import requests

from src.model import GAT
from src.utils import apply_patches, get_device

def load_taxonomy():

    mapping_path = 'data/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz'

    if mapping_path:
        df = pd.read_csv(mapping_path)
        taxonomy = dict(zip(df['label idx'], df['arxiv category']))
        return taxonomy

    else:
        print("Nie znaleziono pliku mapowania kategorii.")

    return None


def get_paper_details(mag_id):
    if mag_id is None:
        return {
            "title": "Brak danych (Offline)",
            "authors": "", "topic": "", "concepts_str": "",
            "url": f"https://openalex.org/W{mag_id}" if mag_id else None
        }

    url = f"https://api.openalex.org/works?filter=ids.mag:{mag_id}"

    try:
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])

            if not results:
                return {
                    "title": "Nie znaleziono w API",
                    "authors": "", "topic": "", "concepts_str": "",
                    "url": f"https://openalex.org/W{mag_id}"
                }

            work = results[0]

            title = work.get('title', 'Nieznany tytuł')
            authorship = work.get('authorships', [])
            authors = [a['author']['display_name'] for a in authorship[:3]]
            if len(authorship) > 3: authors.append("et al.")
            author_str = ", ".join(authors) if authors else "Brak danych"

            primary_topic = work.get('primary_topic', {})
            topic_name = primary_topic.get('display_name', '')

            concepts_raw = work.get('concepts', [])
            concepts_list = []
            for c in concepts_raw:
                if c['level'] > 0 and c['display_name'] not in ["Computer science", "Engineering"]:
                    concepts_list.append(c['display_name'])
            concepts_str = ", ".join(concepts_list[:4])

            ids = work.get('ids', {})


            if 'arxiv' in ids:
                raw_arxiv = ids['arxiv']
                clean_id = raw_arxiv.replace("https://arxiv.org/abs/", "").replace("https://arxiv.org/pdf/",
                                                                                   "").replace(".pdf", "")
                landing_url = f"https://arxiv.org/abs/{clean_id}"

            elif 'doi' in ids and '10.48550/arXiv.' in ids['doi']:
                clean_id = ids['doi'].split("arXiv.")[-1]
                landing_url = f"https://arxiv.org/abs/{clean_id}"

            elif title and title != "Nieznany tytuł":
                safe_title = quote(title)
                landing_url = f"https://arxiv.org/search/?query={safe_title}&searchtype=title"

            else:
                landing_url = ids.get('openalex', f"https://openalex.org/W{mag_id}")

            return {
                "title": title,
                "authors": author_str,
                "topic": topic_name,
                "concepts_str": concepts_str,
                "concepts_list": concepts_list,
                "url": landing_url
            }
    except:
        return {
            "title": "Błąd połączenia",
            "authors": "", "topic": "", "concepts_str": "",
            "url": f"https://openalex.org/W{mag_id}"
        }


def get_demo_samples(num_needed, root_dir='data', mag_mapping=None):
    """
    Wczytuje dataset lokalnie, losuje num_needed próbek i zwraca listę Data objects.
    """
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root_dir, transform=T.ToUndirected())
    data = dataset[0]

    test_idx = dataset.get_idx_split()['test']
    shuffled = test_idx[torch.randperm(test_idx.size(0))]
    node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)

    samples = []

    for idx in shuffled:
        idx = idx.item()
        d = node_degrees[idx].item()

        if 10 <= d <= 50:
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(idx, 1, data.edge_index, relabel_nodes=True)

            s_data = Data(
                x=data.x[subset], edge_index=sub_edge_index, y=data.y[subset],
                center_idx_mapping=mapping[0].item(), original_id=idx, num_nodes=len(subset)
            )

            if mag_mapping is not None:
                s_data.mag_id = mag_mapping[idx]
            else:
                s_data.mag_id = None

            samples.append(s_data)

        if len(samples) >= num_needed:
            break

    return samples


def main():
    apply_patches()
    print("\n" + "=" * 80)
    print("  DEMONSTRACJA LIVE  ")
    print("=" * 80)

    idx_to_class = load_taxonomy()

    mapping_path = 'data/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz'

    df_map = pd.read_csv(mapping_path)
    mag_mapping = df_map['paper id'].values

    while True:
        inp = input("Ile próbek: ").strip()
        if not inp: num_samples = 3; break
        num_samples = int(inp)
        if num_samples > 0: break

    print("=" * 80)
    samples = get_demo_samples(num_needed=num_samples, mag_mapping=mag_mapping)
    device = get_device()
    model = GAT(in_channels=128, hidden_channels=96, out_channels=40, heads=8, dropout=0.4)

    mp = 'models/gatv2_model.pth'

    model.load_state_dict(torch.load(mp, map_location=device))
    model.to(device)
    model.eval()


    for i, data in enumerate(samples):
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)

        center = data.center_idx_mapping
        true_y = data.y[center].item()
        pred_y = preds[center].item()
        conf = F.softmax(out, dim=1)[center][pred_y].item() * 100

        true_name = f"Class {true_y}"
        pred_name = f"Class {pred_y}"
        if idx_to_class:
            true_name = idx_to_class.get(true_y, true_name)
            pred_name = idx_to_class.get(pred_y, pred_name)

        mag_id = getattr(data, 'mag_id', None)
        info = None
        if mag_id:
            print(f"   [Szukanie linku arXiv dla MAG ID: {mag_id}...] ", end="\r")
            info = get_paper_details(mag_id)
            print(" " * 60, end="\r")

        # Kolory
        c_ok = "\033[92m"
        c_err = "\033[91m"
        c_warn = "\033[93m"
        c_blue = "\033[94m"
        c_rst = "\033[0m"

        is_strict_correct = (true_y == pred_y)
        is_concept_correct = False

        if info and not is_strict_correct:
            pred_label_raw = pred_name.lower().split(" ")[0]
            check_str = pred_label_raw.replace("cs.", "")
            for concept in info.get('concepts_list', []):
                if check_str in concept.lower() or concept.lower() in pred_name.lower():
                    is_concept_correct = True
                    break

        if is_strict_correct:
            status_line = f"[{c_ok}POPRAWNIE{c_rst}]"
            pred_color = c_ok
        elif is_concept_correct:
            status_line = f"[{c_warn}NIEZGODNOŚĆ Z DATASETEM, ALE POTWIERDZONE W OPENALEX{c_rst}]"
            pred_color = c_warn
        else:
            status_line = f"[{c_err}BŁĄD{c_rst}]"
            pred_color = c_err

        print(f"PRZYKŁAD #{i + 1}")

        if info:
            print(f"TYTUŁ:           \033[1m{info['title']}\033[0m")
            print(f"AUTORZY:         {info['authors']}")
            if info['topic']:
                print(f"TEMATYKA:        {c_blue}{info['topic']}{c_rst}")

        print(f"MAG ID:          {mag_id if mag_id else 'N/A'}")

        final_url = info['url'] if info and info['url'] else f"https://openalex.org/W{mag_id}" if mag_id else "Brak ID"
        print(f"LINK: \033[4;94m{final_url}\033[0m")
        print(f"Prawdziwa kategoria:  {c_blue}{true_name}{c_rst}")
        print(f"Predykcja:       {pred_color}{pred_name}{c_rst}")
        print(f"Pewność:         {conf:.2f}%")
        print(f"Status:          {status_line}")
        print("=" * 80)


if __name__ == "__main__":
    main()