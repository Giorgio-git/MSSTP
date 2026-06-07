"""
Patch #4 — VNS: Variable Neighborhood Search
Aggiunge tre funzioni:
  1. two_swap_random()          → cella c1b652bb (dopo cycle_exchange)
  2. vns_kick()                 → cella c1b652bb
  3. simulated_annealing_vns()  → cella 11058b6a (nuova funzione distinta)
"""
import json, sys, os

NB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'MSSTP.ipynb')

def slines(code):
    rows = code.split('\n')
    return [r + '\n' if i < len(rows) - 1 else r for i, r in enumerate(rows) if r != '' or i < len(rows) - 1]

# ─── Codice da appendere alla cella c1b652bb ──────────────────────────────────
TWO_SWAP_AND_KICK = '''

def two_swap_random(G, T, np_data, n_samples=20):
    """
    Mossa di diversificazione 2-swap: rimuove 2 archi casuali dall\'albero,
    ottiene 3 componenti connesse (A, B, C) e le riconnette scegliendo
    l\'arco di peso minimo per ciascun collegamento necessario.

    Per ogni campione esplora le 3 topologie di riconnessione "a catena":
      [A─B─C], [A─C─B], [B─A─C]  (scelta del nodo centrale)
    e tiene la migliore in termini di stretch calcolato con SciPy (vettoriale).

    L\'arco di peso minimo è preferito rispetto a quello che minimizza lo stretch
    direttamente: trattandosi di una mossa di diversificazione, vogliamo uscire
    dal bacino corrente senza ottimizzare troppo (evitare l\'intensificazione prematura).

    Parametri
    ---------
    G         : grafo originale (completo, pesato)
    T         : albero corrente (NetworkX Graph)
    np_data   : dict da build_np_edge_data(G)
    n_samples : numero di coppie di archi da campionare

    Restituisce
    -----------
    (T_best, best_stretch) oppure (None, inf) se nessuna mossa valida
    """
    tree_edges = list(T.edges())
    if len(tree_edges) < 2:
        return None, float(\'inf\')

    best_T, best_stretch = None, float(\'inf\')

    for _ in range(n_samples):
        # Step 1: scegli 2 archi distinti dell\'albero
        e1, e2 = random.sample(tree_edges, 2)

        # Step 2: rimuovi entrambi → 3 componenti
        T_split = T.copy()
        T_split.remove_edge(*e1)
        T_split.remove_edge(*e2)
        components = list(nx.connected_components(T_split))

        if len(components) != 3:
            continue  # archi adiacenti → stessa componente → skip

        A, B, C = components[0], components[1], components[2]

        # Step 3: prova le 3 topologie "a catena" (scelta del nodo centrale)
        topologie = [
            [(A, B), (B, C)],   # B è il nodo centrale
            [(A, C), (C, B)],   # C è il nodo centrale
            [(B, A), (A, C)],   # A è il nodo centrale
        ]

        for topo in topologie:
            nuovi_archi = []
            valida = True

            for (comp_from, comp_to) in topo:
                # Arco di peso minimo tra le due componenti (non-tree)
                best_edge, min_w = None, float(\'inf\')
                for u in comp_from:
                    for v in comp_to:
                        if G.has_edge(u, v):
                            w = G[u][v].get(\'weight\', 0)
                            if w < min_w:
                                min_w, best_edge = w, (u, v, w)

                if best_edge is None:
                    valida = False
                    break
                nuovi_archi.append(best_edge)

            if not valida:
                continue

            # Costruisce il candidato e verifica connettività (sanity check)
            T_candidate = T_split.copy()
            for u, v, w in nuovi_archi:
                T_candidate.add_edge(u, v, weight=w)

            if not nx.is_connected(T_candidate):
                continue

            # Valuta stretch con SciPy (vettoriale, O(N log N + M))
            s = compute_stretch(G, T_candidate, np_data=np_data)
            if s < best_stretch:
                best_stretch, best_T = s, T_candidate

    return best_T, best_stretch


def vns_kick(G, T, np_data, edge_stretches_arr, k_neighborhood, n_samples=20):
    """
    Perturbazione VNS: attiva il vicinato k_neighborhood.

    k=1 → cycle_exchange (1-swap) + calcolo incrementale NumPy
    k=2 → two_swap_random campionato (n_samples mosse)
    k=3 → hybrid_local_search (intensificazione completa)

    Restituisce (T_new, new_stretch, new_edge_stretches_arr)
    """
    if k_neighborhood == 1:
        T_old = T
        T_new, e_add, e_remove = cycle_exchange(G, T)
        if e_add is not None and e_remove is not None:
            new_stretch, new_es = compute_stretch_incremental_np(
                np_data, edge_stretches_arr, T_old, e_add, e_remove)
        else:
            new_stretch, new_es = compute_stretch(
                G, T_new, return_edge_stretches=True, np_data=np_data)
        return T_new, new_stretch, new_es

    elif k_neighborhood == 2:
        T_new, new_stretch = two_swap_random(G, T, np_data, n_samples)
        if T_new is None:
            # fallback a 1-swap se il 2-swap non trova mosse valide
            return vns_kick(G, T, np_data, edge_stretches_arr, 1, n_samples)
        _, new_es = compute_stretch(G, T_new, return_edge_stretches=True, np_data=np_data)
        return T_new, new_stretch, new_es

    else:  # k_neighborhood == 3
        cur_stretch = compute_stretch(G, T, np_data=np_data)
        T_new, new_stretch = hybrid_local_search(G, T, cur_stretch)
        _, new_es = compute_stretch(G, T_new, return_edge_stretches=True, np_data=np_data)
        return T_new, new_stretch, new_es
'''

# ─── simulated_annealing_vns da aggiungere in cella 11058b6a ─────────────────
SA_VNS = '''

def simulated_annealing_vns(G, T_init,
                             T_high=None, T_low=1.0, alpha=None,
                             max_iters_per_temp=None,
                             th_min=0, th_max=200,
                             penalty=50, grace_period=3,
                             max_seconds=None,
                             stagnation_limit=50,
                             n_2swap_samples=20):
    """
    Simulated Annealing con Variable Neighborhood Search (SA-VNS).

    Estende simulated_annealing() aggiungendo una gerarchia di vicinati:
      N₁ = 1-swap (cycle_exchange)       — veloce, O(M) per valutazione
      N₂ = 2-swap campionato             — 20-50x più espressivo di N₁
      N₃ = hybrid_local_search           — intensificazione completa

    Logica VNS:
      - Il SA opera normalmente in N₁
      - Se per stagnation_limit passi di temperatura consecutivi non migliora
        il record globale, scala a N₂ (poi N₃)
      - Dopo ogni miglioramento del record globale torna sempre a N₁

    Parametri aggiuntivi rispetto a simulated_annealing():
      stagnation_limit : passi di temperatura senza miglioramento prima di scalare
      n_2swap_samples  : campioni per ogni chiamata a two_swap_random
    """
    N = G.number_of_nodes()
    np_data = build_np_edge_data(G)

    # ── Parametri adattativi ──────────────────────────────────────────────────
    if T_high is None:
        T_high = max(100.0, float(N) * 5.0)
    if alpha is None:
        alpha = 1.0 - 0.5 / max(N, 2)
    if max_iters_per_temp is None:
        max_iters_per_temp = min(max(20, N // 2), 100)

    # ── Inizializzazione ──────────────────────────────────────────────────────
    current_solution = T_init.copy()
    current_stretch, edge_stretches = compute_stretch(
        G, current_solution, return_edge_stretches=True, np_data=np_data)

    best_solution = current_solution.copy()
    best_stretch  = current_stretch

    Temp          = T_high
    stretch_history = [best_stretch]
    ls_failures   = 0
    k             = 1   # vicinato attivo (1, 2 o 3)
    stagnation_count = 0

    ln_high = math.log(max(T_high, 1e-9))
    ln_low  = math.log(max(T_low,  1e-9))

    print(f"--- Inizio SA-VNS (stagnation_limit={stagnation_limit}, n_2swap={n_2swap_samples}) ---")
    print(f"Stretch Iniziale: {round(best_stretch, 2)}")
    print(f"Parametri: T_high={round(T_high,1)}, alpha={round(alpha,4)}, "
          f"iters_per_temp={max_iters_per_temp}, N={N}")

    iter_count = 0
    start_time = time.time()

    while Temp > T_low:
        if Temp < 0:
            break
        if max_seconds is not None and (time.time() - start_time) > max_seconds:
            print(f"  [TIMEOUT] SA-VNS interrotto dopo {round(time.time()-start_time,1)}s "
                  f"— stretch: {round(best_stretch,2)}")
            break

        # ── Soglia dinamica per LS (invariata) ───────────────────────────────
        ln_temp = math.log(max(Temp, 1e-9))
        ratio   = (ln_high - ln_temp) / (ln_high - ln_low + 1e-9)
        th_base = th_min + ratio * (th_max - th_min)
        if ls_failures > grace_period:
            th_eff = max(0, th_base - penalty * (ls_failures - grace_period))
        else:
            th_eff = th_base

        best_improved_this_temp = False

        # ── Iterazioni a temperatura costante ────────────────────────────────
        for i in range(max_iters_per_temp):
            if max_seconds is not None and i % 5 == 4 and (time.time() - start_time) > max_seconds:
                print(f"  [TIMEOUT] SA-VNS interrotto dopo {round(time.time()-start_time,1)}s "
                      f"— stretch: {round(best_stretch,2)}")
                Temp = -1
                break
            iter_count += 1

            # Perturbazione VNS: usa il vicinato attivo k
            T_new, new_stretch, new_edge_stretches = vns_kick(
                G, current_solution, np_data, edge_stretches, k, n_2swap_samples)

            delta    = new_stretch - current_stretch
            accepted = False

            if delta < 0:
                current_solution = T_new
                current_stretch  = new_stretch
                edge_stretches   = new_edge_stretches
                accepted = True
            else:
                try:
                    prob = math.exp(-delta / Temp)
                except OverflowError:
                    prob = 0.0
                if random.random() < prob:
                    current_solution = T_new
                    current_stretch  = new_stretch
                    edge_stretches   = new_edge_stretches
                    accepted = True

            if accepted:
                if current_stretch < best_stretch:
                    best_solution = current_solution.copy()
                    best_stretch  = current_stretch
                    best_improved_this_temp = True
                    if k > 1:
                        print(f"  [VNS k={k}→1] Miglioramento: {round(best_stretch, 2)}")
                    k = 1   # ← torna sempre a N₁ dopo un miglioramento del record

                # Trigger LS solo per k=1 (come nel SA originale)
                if k == 1 and current_stretch <= best_stretch + th_eff:
                    if \'hybrid_local_search\' in globals():
                        ls_T, ls_stretch = hybrid_local_search(G, current_solution, current_stretch)
                        current_solution = ls_T.copy()
                        current_stretch  = ls_stretch
                        _, edge_stretches = compute_stretch(
                            G, current_solution, return_edge_stretches=True, np_data=np_data)

                        if ls_stretch < best_stretch:
                            best_solution = ls_T.copy()
                            best_stretch  = ls_stretch
                            best_improved_this_temp = True
                            ls_failures   = 0
                        else:
                            ls_failures += 1
                            if ls_failures > grace_period:
                                th_eff = max(0, th_base - penalty * (ls_failures - grace_period))

        stretch_history.append(best_stretch)

        # ── Stagnation detection: scala il vicinato ───────────────────────────
        if best_improved_this_temp:
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= stagnation_limit and k < 3:
            k += 1
            stagnation_count = 0
            print(f"  [VNS] Stagnazione ({stagnation_limit} passi) → vicinato N{k} "
                  f"(stretch={round(best_stretch, 2)})")

        Temp *= alpha

    print(f"--- Fine SA-VNS ---")
    print(f"Stretch Migliore: {round(best_stretch, 2)} dopo {iter_count} iterazioni.")
    return best_solution, best_stretch, stretch_history
'''

# ─── Applicazione delle patch ─────────────────────────────────────────────────
print(f"Caricamento: {NB}")
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

ok = {'c1b652bb': False, '11058b6a': False}

for cell in nb['cells']:
    cid = cell.get('id', '')

    if cid == 'c1b652bb':
        src = ''.join(cell['source'])
        if 'two_swap_random' in src:
            print("  ⚠  c1b652bb: two_swap_random già presente, skipped")
            ok['c1b652bb'] = True
        else:
            src = src.rstrip() + '\n' + TWO_SWAP_AND_KICK
            cell['source'] = slines(src)
            cell['outputs'] = []
            cell['execution_count'] = None
            ok['c1b652bb'] = True
            print("  ✓ c1b652bb: aggiunte two_swap_random() e vns_kick()")

    elif cid == '11058b6a':
        src = ''.join(cell['source'])
        if 'simulated_annealing_vns' in src:
            print("  ⚠  11058b6a: simulated_annealing_vns già presente, skipped")
            ok['11058b6a'] = True
        else:
            src = src.rstrip() + '\n' + SA_VNS
            cell['source'] = slines(src)
            cell['outputs'] = []
            cell['execution_count'] = None
            ok['11058b6a'] = True
            print("  ✓ 11058b6a: aggiunta simulated_annealing_vns()")

if not all(ok.values()):
    print(f"\n✗ Patch FALLITA per celle: {[k for k,v in ok.items() if not v]}")
    sys.exit(1)

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

with open(NB, 'r', encoding='utf-8') as f:
    json.load(f)   # validazione JSON

print("\n✓ Notebook salvato e JSON validato!")
