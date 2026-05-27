import json

path = "/Users/giorgiobianchini/Desktop/UNIVERSITA/4° ANNO/I semestre/Ricerca operativa/Progetto/MSSTP.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cerchiamo la cella di codice che contiene il download (dovrebbe essere l'ultima)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            if 'urllib.request.urlretrieve(url, gz_path)' in line:
                # Sostituiamo urllib con una richiesta requests più robusta
                indent = line[:line.find('urllib')]
                new_source.append(f"{indent}# Uso requests con User-Agent per evitare il 403\n")
                new_source.append(f"{indent}headers = {{'User-Agent': 'Mozilla/5.0'}}\n")
                new_source.append(f"{indent}resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers))\n")
                new_source.append(f"{indent}with open(gz_path, 'wb') as f_gz:\n")
                new_source.append(f"{indent}    f_gz.write(resp.read())\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
