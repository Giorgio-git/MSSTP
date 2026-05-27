import pulp
import tsplib95
import numpy as np
import time
import os
from datetime import datetime

def solve_msstp_exact(file_tsp, limit_nodes=None, time_limit_sec=None, output_file=None):
    print(f"\n--- AVVIO RISOLUZIONE ESATTA: {file_tsp} ---")
    
    # 1. Caricamento Dati
    if not os.path.exists(file_tsp):
        raise FileNotFoundError(f"File {file_tsp} non trovato.")
        
    problem = tsplib95.load(file_tsp)   # usiamo la libreria tsplib95 per caricare l'istanza TSP e ottenere i nodi e i costi
    nodes = list(problem.get_nodes())
    
    # Se vogliamo testare su un sotto-grafo per fare prima
    if limit_nodes and limit_nodes < len(nodes):
        nodes = nodes[:limit_nodes]
        print(f"ATTENZIONE: Analizzo solo i primi {limit_nodes} nodi per questioni di tempo.")
        
    N = len(nodes)
    print(f"Nodi attivi: {N}")

    # Creazione matrice dei costi
    cost = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                cost[i][j] = problem.get_weight(nodes[i], nodes[j])

    # 2. Inizializzazione Problema PuLP
    prob = pulp.LpProblem("Minimum_Stretch_Spanning_Tree", pulp.LpMinimize)

    # 3. Variabili Decisionali
    print("Generazione variabili in corso...")
    
    # x_ij = 1 se l'arco (i,j) fa parte dell'albero (Grafo non orientato, i < j)
    x_keys = [(i, j) for i in range(N) for j in range(N) if i < j]
    x = pulp.LpVariable.dicts("x", x_keys, cat='Binary')

    # y_stij = flusso da s a t che passa sull'arco orientato i->j
    # Per ottimizzare, le dichiariamo Continue tra 0 e 1.
    y_keys = [(s, t, i, j) for s in range(N) for t in range(N) if s < t 
                           for i in range(N) for j in range(N) if i != j]
    y = pulp.LpVariable.dicts("y", y_keys, lowBound=0, upBound=1, cat='Continuous')

    # z = scarto massimo da minimizzare
    z = pulp.LpVariable("z", lowBound=0, cat='Continuous')

    # 4. Funzione Obiettivo
    prob += z, "Minimizza_Scarto_Massimo"

    # 5. Vincoli
    print("Generazione vincoli in corso...")

    # A. L'albero deve avere N-1 archi
    prob += pulp.lpSum(x[(i, j)] for i, j in x_keys) == N - 1, "Tree_Edges_Count"

    # B. Conservazione del flusso (Trova i cammini minimi nell'albero)
    for s in range(N):
        for t in range(N):
            if s < t:
                for k in range(N):
                    # Flusso che esce da k - Flusso che entra in k
                    out_flow = pulp.lpSum(y[(s, t, k, j)] for j in range(N) if j != k)
                    in_flow  = pulp.lpSum(y[(s, t, i, k)] for i in range(N) if i != k)
                    
                    if k == s:
                        prob += (out_flow - in_flow == 1), f"Flow_Cons_{s}_{t}_node_{k}_is_source"
                    elif k == t:
                        prob += (out_flow - in_flow == -1), f"Flow_Cons_{s}_{t}_node_{k}_is_sink"
                    else:
                        prob += (out_flow - in_flow == 0), f"Flow_Cons_{s}_{t}_node_{k}_is_transit"

    # C. Capacità degli archi (il flusso passa solo se l'arco è nell'albero)
    for s in range(N):
        for t in range(N):
            if s < t:
                for i, j in x_keys:
                    # Il flusso può andare da i a j OPPURE da j a i
                    prob += y[(s, t, i, j)] + y[(s, t, j, i)] <= x[(i, j)], f"Capacity_{s}_{t}_edge_{i}_{j}"

    # D. Calcolo dello scarto (Linearizzazione di Max)
    for s in range(N):
        for t in range(N):
            if s < t:
                # Distanza sull'albero tra s e t (somma dei costi degli archi usati dal flusso y_st)
                distanza_albero = pulp.lpSum(cost[i][j] * y[(s, t, i, j)] for i in range(N) for j in range(N) if i != j)
                scarto = distanza_albero - cost[s][t]
                prob += z >= scarto, f"Stretch_Constraint_{s}_{t}"

    # 6. Risoluzione
    print(f"Modello pronto. Avvio solver CBC (Time Limit: {time_limit_sec if time_limit_sec is not None else 'NESSUN LIMITE'}s)...")
    start_time = time.time()
    
    # Configuriamo il solver CBC
    if time_limit_sec is not None:
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_sec, msg=True)
    else:
        solver = pulp.PULP_CBC_CMD(msg=True)
    status = prob.solve(solver)
    
    end_time = time.time()

    # 7. Output Risultati
    print("\n" + "="*40)
    print(f"Stato Ottimizzazione: {pulp.LpStatus[status]}")
    
    # Prepara il contenuto del report
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"REPORT RISOLUZIONE ESATTA MSSTP")
    report_lines.append(f"Data e Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*60)
    report_lines.append(f"Istanza: {os.path.basename(file_tsp)}")
    report_lines.append(f"Numero Nodi: {N}")
    report_lines.append(f"Stato Ottimizzazione: {pulp.LpStatus[status]}")
    
    if status == pulp.LpStatusOptimal or (status == pulp.LpStatusNotSolved and pulp.value(z) is not None):
        upper_bound = pulp.value(z)
        lower_bound = prob.objective.constant if hasattr(prob, 'objective') else None
        
        report_lines.append(f"\nRISULTATI:")
        report_lines.append(f"  Scarto Massimo (Stretch): {upper_bound}")
        if lower_bound is not None and lower_bound > 0:
            report_lines.append(f"  Lower bound (rilassata): {lower_bound:.2f}")
            gap = ((upper_bound - lower_bound) / upper_bound * 100) if upper_bound > 0 else 0
            report_lines.append(f"  Gap: {gap:.2f}%")
        report_lines.append(f"  Tempo di calcolo: {end_time - start_time:.2f} secondi")
        
        report_lines.append(f"\nARCHI DELL'ALBERO OTTIMALE ({N-1} archi):")
        archi_lista = []
        for i, j in x_keys:
            if pulp.value(x[(i, j)]) is not None and pulp.value(x[(i, j)]) > 0.5:
                arco_str = f"  Arco ({nodes[i]:3d}, {nodes[j]:3d}) - Costo: {cost[i][j]:8.1f}"
                report_lines.append(arco_str)
                archi_lista.append(arco_str)
        
        report_lines.append("\n" + "="*60)
        
        # Stampa a console
        print(f"SCARTO MASSIMO: {upper_bound}")
        if lower_bound is not None and lower_bound > 0:
            print(f"Lower bound (rilassata): {lower_bound:.2f}")
            gap = ((upper_bound - lower_bound) / upper_bound * 100) if upper_bound > 0 else 0
            print(f"Gap: {gap:.2f}%")
        print(f"Tempo di calcolo: {end_time - start_time:.2f} secondi")
        print(f"\nArchi dell'albero ottimale:")
        for arco in archi_lista:
            print(arco)
    else:
        report_lines.append("ESITO: Il solver non è riuscito a trovare una soluzione ammissibile nel tempo limite.")
        print("Il solver non è riuscito a trovare una soluzione ammissibile nel tempo limite.")
    
    # 8. Salvataggio risultati su file (se specificato)
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\n✓ Risultati salvati in: {output_file}")
        
    return upper_bound if status == pulp.LpStatusOptimal or (status == pulp.LpStatusNotSolved and pulp.value(z) is not None) else None

# ==========================================
# ESECUZIONE
# ==========================================
if __name__ == "__main__":
    # FASE 1: Test su istanze piccole PER GENERARE SOLUZIONI OTTIMALI
    # (nessun limite di tempo per trovare la soluzione esatta)
    print("\n" + "="*60)
    print("FASE 1: GENERAZIONE SOLUZIONI OTTIMALI (Nessun limite di tempo)")
    print("="*60)
    
    # Test su gr17.tsp
    print("\n[1/2] Risoluzione gr17.tsp in corso...")
    solve_msstp_exact("data/extracted_tsp/gr17.tsp", 
                     time_limit_sec=None, 
                     output_file="risultati_gr17_ottimo.txt")
    
    print("\n" + "#"*60 + "\n")
    
    # Test su ulysses16.tsp
    print("[2/2] Risoluzione ulysses16.tsp in corso...")
    solve_msstp_exact("data/extracted_tsp/ulysses16.tsp", 
                     time_limit_sec=None, 
                     output_file="risultati_ulysses16_ottimo.txt")
    
    print("\n" + "="*60)
    print("FASE 1 COMPLETATA")
    print("="*60)
    print("\nFile risultati generati:")
    print("  - risultati_gr17_ottimo.txt")
    print("  - risultati_ulysses16_ottimo.txt")
    
    # FASE 2: Test su istanze più grandi CON limite di tempo (per il futuro)
    # Decommentare le righe seguenti quando necessario testare istanze grandi:
    # print("\n" + "="*60)
    # print("FASE 2: TEST SU ISTANZE GRANDI (Con limite di tempo)")
    # print("="*60)
    # solve_msstp_exact("data/extracted_tsp/att48.tsp", time_limit_sec=600, output_file="risultati_att48_600s.txt")
    # solve_msstp_exact("data/extracted_tsp/berlin52.tsp", time_limit_sec=600, output_file="risultati_berlin52_600s.txt")