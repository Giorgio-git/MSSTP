import json

path = "/Users/giorgiobianchini/Desktop/UNIVERSITA/4° ANNO/I semestre/Ricerca operativa/Progetto_studio_1/MSSTP_studio_1.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cerchiamo la cella con le importazioni
for cell in nb['cells']:
    if cell.get('id') == 'cell-imports':
        source = cell['source']
        for i, line in enumerate(source):
            if 'resp = requests.get(url, timeout=30)' in line:
                source[i] = line.replace(
                    'resp = requests.get(url, timeout=30)',
                    "resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)"
                )
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
