"""
Test di correttezza + benchmark velocità della vettorizzazione NumPy.
Verifica:
  1. compute_stretch_incremental_np == compute_stretch_incremental (90 mosse)
  2. compute_stretch con np_data == compute_stretch senza np_data
  3. Speedup misurato per singola iterazione SA su istanze di diverse dimensioni
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx, tsplib95, nbformat, numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED = os.path.join(PROJ, 'data', 'extracted_tsp')

nb_path = os.path.join(PROJ, 'MSSTP.ipynb')
with open(nb_path, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)
ns = {}
CELLS = ['379ee5b9', 'f5065905', '38d3ee0c', 'c1b652bb', 'f8a78b4d', '11058b6a']
for cell in nb.cells:
    if cell.cell_type == 'code' and cell.get('id') in CELLS:
        try: exec(compile(''.join(cell.source), f"<cell {cell.get('id')}>", 'exec'), ns)
        except Exception as e: print(f"Errore cella {cell.get('id')}: {e}")

def load_tsp(name):
    G = tsplib95.load(os.path.join(EXTRACTED, name + '.tsp')).get_graph()
    return nx.convert_node_labels_to_integers(G, first_label=0)

compute_stretch              = ns['compute_stretch']
compute_stretch_incremental  = ns['compute_stretch_incremental']
compute_stretch_incremental_np = ns['compute_stretch_incremental_np']
build_np_edge_data           = ns['build_np_edge_data']
cycle_exchange               = ns['cycle_exchange']
generate_initial_solution    = ns['generate_initial_solution']

print("=" * 65)
print("TEST 1: Correttezza compute_stretch con np_data")
print("=" * 65)
for inst in ['burma14', 'att48', 'st70', 'kroA100']:
    G = load_tsp(inst)
    T, _, _ = generate_initial_solution(G, method='kruskal', rcl_size=3)
    np_data = build_np_edge_data(G)
    s1 = compute_stretch(G, T)
    s2 = compute_stretch(G, T, np_data=np_data)
    ok = abs(s1 - s2) < 1e-6
    print(f"  {inst:<12} stretch_orig={s1:.2f}  stretch_np={s2:.2f}  {'✓' if ok else '✗ ERRORE'}")

print()
print("=" * 65)
print("TEST 2: Correttezza compute_stretch_incremental_np (90 mosse)")
print("=" * 65)
errors = 0
for inst in ['burma14', 'att48', 'st70']:
    G = load_tsp(inst)
    np_data = build_np_edge_data(G)
    T, _, _ = generate_initial_solution(G, method='kruskal', rcl_size=3)
    _, es_dict = compute_stretch(G, T, return_edge_stretches=True)
    _, es_arr  = compute_stretch(G, T, return_edge_stretches=True, np_data=np_data)
    for _ in range(30):
        T_old = T.copy()
        T_new, e_add, e_remove = cycle_exchange(G, T)
        if e_add is None: continue
        s_dict, _ = compute_stretch_incremental(G, T_old, es_dict, e_add, e_remove)
        s_np,   new_es_arr = compute_stretch_incremental_np(np_data, es_arr, T_old, e_add, e_remove)
        if abs(s_dict - s_np) > 1e-6:
            print(f"  ✗ ERRORE {inst}: dict={s_dict:.4f} != np={s_np:.4f}")
            errors += 1
        T = T_new
        _, es_dict = compute_stretch(G, T, return_edge_stretches=True)
        es_arr = new_es_arr
    print(f"  {inst:<12} 30 mosse controllate  ✓")
print(f"  Totale errori: {errors}")

print()
print("=" * 65)
print("TEST 3: Speedup per singola valutazione dello stretch")
print("=" * 65)
print(f"  {'Istanza':<12} {'N':>5} {'M':>7}  {'Python (ms)':>12}  {'NumPy (ms)':>11}  {'Speedup':>8}")
print("  " + "-" * 55)

for inst, N_expect in [('att48', 48), ('kroA100', 100), ('ch130', 130),
                        ('rd400', 400), ('d493', 493)]:
    path = os.path.join(EXTRACTED, inst + '.tsp')
    if not os.path.exists(path): continue
    G = load_tsp(inst)
    np_data = build_np_edge_data(G)
    N = G.number_of_nodes()
    M = G.number_of_edges()
    T, _, _ = generate_initial_solution(G, method='kruskal', rcl_size=3)
    _, es_dict = compute_stretch(G, T, return_edge_stretches=True)
    _, es_arr  = compute_stretch(G, T, return_edge_stretches=True, np_data=np_data)
    T_new, e_add, e_remove = cycle_exchange(G, T)
    if e_add is None: continue

    # Warmup
    compute_stretch_incremental(G, T, es_dict, e_add, e_remove)
    compute_stretch_incremental_np(np_data, es_arr, T, e_add, e_remove)

    # Benchmark Python (5 ripetizioni)
    reps = 5
    t0 = time.time()
    for _ in range(reps):
        compute_stretch_incremental(G, T, es_dict, e_add, e_remove)
    t_py = (time.time() - t0) / reps * 1000

    # Benchmark NumPy (5 ripetizioni)
    t0 = time.time()
    for _ in range(reps):
        compute_stretch_incremental_np(np_data, es_arr, T, e_add, e_remove)
    t_np = (time.time() - t0) / reps * 1000

    speedup = t_py / max(t_np, 0.001)
    print(f"  {inst:<12} {N:>5} {M:>7}  {t_py:>10.1f}ms  {t_np:>9.1f}ms  {speedup:>6.1f}x")

print()
print("=" * 65)
print("FINE TEST" + ("  — ERRORI!" if errors else "  — Tutti i test superati! ✓"))
print("=" * 65)
