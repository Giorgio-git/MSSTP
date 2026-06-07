"""
Test VNS — esegui con: venv/bin/python scratch/test_vns.py

Esegue 4 test:
  A) Correttezza topologica two_swap_random (è sempre un albero valido?)
  B) two_swap_random trova vicini migliori dal plateau? (rd400, rat575)
  C) Correttezza vns_kick per tutti e 3 i livelli di vicinato
  D) Benchmark SA vs SA-VNS su 5 istanze con timeout 120s
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx, tsplib95, nbformat, numpy as np

PROJ  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTR  = os.path.join(PROJ, 'data', 'extracted_tsp')
NB    = os.path.join(PROJ, 'MSSTP.ipynb')
CELLS = ['379ee5b9','f5065905','38d3ee0c','c1b652bb','f8a78b4d','11058b6a']

print("Caricamento notebook...")
with open(NB, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)
ns = {}
for cell in nb.cells:
    if cell.cell_type == 'code' and cell.get('id') in CELLS:
        try:
            exec(compile(''.join(cell.source), f"<{cell.get('id')}>", 'exec'), ns)
        except Exception as e:
            print(f"  Errore cella {cell.get('id')}: {e}")

def load_tsp(name):
    G = tsplib95.load(os.path.join(EXTR, name + '.tsp')).get_graph()
    return nx.convert_node_labels_to_integers(G, first_label=0)

gen    = ns['generate_initial_solution']
cs     = ns['compute_stretch']
np_ed  = ns['build_np_edge_data']
two_sw = ns['two_swap_random']
vkick  = ns['vns_kick']
sa     = ns['simulated_annealing']
sa_vns = ns['simulated_annealing_vns']


# ── TEST A: Correttezza topologica ─────────────────────────────────────────────
print("\n" + "="*65)
print("TEST A: Correttezza topologica two_swap_random (200 mosse × 3 istanze)")
print("="*65)
total_ok, total_fail = 0, 0
for inst in ['burma14', 'att48', 'st70', 'kroA100']:
    G = load_tsp(inst)
    np_data = np_ed(G)
    T, _, _ = gen(G, method='kruskal', rcl_size=3)
    fails = 0
    for _ in range(200):
        T_new, s = two_sw(G, T, np_data, n_samples=10)
        if T_new is None: continue
        if not nx.is_tree(T_new):
            fails += 1
        if T_new.number_of_edges() != G.number_of_nodes() - 1:
            fails += 1
    status = '✓' if fails == 0 else f'✗ ({fails} errori)'
    print(f"  {inst:<12}: {status}")
    total_fail += fails
print(f"  Totale errori topologici: {total_fail}")


# ── TEST B: two_swap_random sfugge ai plateau? ─────────────────────────────────
print("\n" + "="*65)
print("TEST B: two_swap_random trova soluzioni migliori dal plateau?")
print("="*65)
for inst in ['rd400', 'rat575', 'kroA100']:
    path = os.path.join(EXTR, inst + '.tsp')
    if not os.path.exists(path): continue
    G = load_tsp(inst)
    np_data = np_ed(G)
    best_T, best_st = None, float('inf')
    for m in ['kruskal','prim','dijkstra']:
        T, st, _ = gen(G, method=m, rcl_size=3)
        if st < best_st: best_st, best_T = st, T

    # Campiona 100 mosse 2-swap dall'ottimo locale
    results = []
    for _ in range(100):
        T_new, s = two_sw(G, best_T, np_data, n_samples=20)
        if T_new is not None: results.append(s)

    if not results:
        print(f"  {inst}: nessuna mossa 2-swap valida trovata")
        continue
    arr = np.array(results)
    pct_better = (arr < best_st).mean() * 100
    pct_equal  = (np.abs(arr - best_st) < 1e-6).mean() * 100
    pct_worse  = (arr > best_st).mean() * 100
    print(f"  {inst} (ottimo 1-swap={best_st:.0f}) | 2-swap: "
          f"{pct_better:.1f}% migliori, {pct_equal:.1f}% uguali, {pct_worse:.1f}% peggiori")
    print(f"    min={arr.min():.2f}  median={np.median(arr):.2f}  max={arr.max():.2f}")


# ── TEST C: vns_kick tutti i livelli ──────────────────────────────────────────
print("\n" + "="*65)
print("TEST C: vns_kick — correttezza per k=1,2,3")
print("="*65)
G = load_tsp('att48')
np_data = np_ed(G)
T, st, _ = gen(G, method='kruskal', rcl_size=3)
_, es_arr = cs(G, T, return_edge_stretches=True, np_data=np_data)
for k in [1, 2, 3]:
    T_new, s, es_new = vkick(G, T, np_data, es_arr, k, n_samples=10)
    is_tree = nx.is_tree(T_new)
    print(f"  k={k}: stretch={s:.2f}, is_tree={is_tree}  {'✓' if is_tree else '✗'}")


# ── TEST D: Benchmark SA vs SA-VNS ────────────────────────────────────────────
print("\n" + "="*65)
print("TEST D: SA vs SA-VNS (timeout 120s per istanza)")
print("="*65)
print(f"  {'Istanza':<12} {'N':>5} | {'SA stretch':>12} | {'VNS stretch':>12} | {'Delta':>7}")
print("  " + "-"*57)

for inst in ['kroA100', 'ch130', 'rd400', 'rat575']:
    path = os.path.join(EXTR, inst + '.tsp')
    if not os.path.exists(path): continue
    G = load_tsp(inst)
    N = G.number_of_nodes()
    best_T, best_st = None, float('inf')
    for m in ['kruskal','prim','dijkstra']:
        T_init, st, _ = gen(G, method=m, rcl_size=3)
        if st < best_st: best_st, best_T = st, T_init

    # SA baseline
    _, sa_stretch, _ = sa(G, best_T.copy(), max_seconds=120)
    # SA-VNS
    _, vns_stretch, _ = sa_vns(G, best_T.copy(), max_seconds=120,
                                stagnation_limit=50, n_2swap_samples=20)

    delta = sa_stretch - vns_stretch
    sign  = '+' if delta > 0 else ''
    print(f"  {inst:<12} {N:>5} | {sa_stretch:>12.2f} | {vns_stretch:>12.2f} | {sign}{delta:.2f}")

print("\n" + "="*65)
print("FINE TEST VNS")
print("="*65)
