import numpy as np
import random
import math
import matplotlib.pyplot as plt
from typing import List, Tuple

from configuracao import GeneticAlgorithmConfig


# ===========================
# ========= GA TSP ==========
# ===========================
class GeneticAlgorithmTSP:
    def __init__(self, coordenadas, familias, visitas_familia, familia_no, config: GeneticAlgorithmConfig):
        """
        coordenadas: List[Tuple[float,float]] índice = nó
        familias: List[List[int]] lista de listas de nós (cada sublista é uma família)
        visitas_familia: List[int] quantidade a visitar em cada família (mesmo comprimento que familias)
        familia_no: dict {no: family_idx+1} (como no seu código original)
        """
        self.coordenadas = coordenadas
        self.familias = familias
        self.visitas_familia = visitas_familia
        self.familia_no = familia_no
        self.config = config

        # parâmetros
        self.tamanho_populacao = config.tamanho_populacao
        self.taxa_mutacao_base = config.taxa_mutacao
        self.taxa_crossover = config.taxa_crossover
        self.elite_size = config.elite_size
        self.max_geracoes = config.max_geracoes

        # histórico
        self.historico_fitness = []

        # seed opcional
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        # matriz de distâncias pré-computada
        self._dist = self._precompute_dist()

        # estrutura auxiliar: mapeia família -> set de nós (rápido p/ checagens)
        self._familia_sets = [set(f) for f in self.familias]

    # ======================================================
    # ------------------ Utilitários -----------------------
    # ======================================================
    def _precompute_dist(self):
        n = len(self.coordenadas)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            xi, yi = self.coordenadas[i]
            for j in range(i+1, n):
                xj, yj = self.coordenadas[j]
                d = math.hypot(xi - xj, yi - yj)
                D[i, j] = d
                D[j, i] = d
        return D

    def distancia(self, i, j):
        return self._dist[i, j]

    def _contagem_por_familia(self, individuo) -> List[int]:
        cont_fam = [0] * len(self.familias)
        for no in individuo[1:-1]:
            fidx = self.familia_no.get(no, 0)
            if fidx > 0:
                cont_fam[fidx - 1] += 1
        return cont_fam

    def calcular_fitness(self, individuo):
        return sum(self._dist[individuo[i], individuo[i+1]] for i in range(len(individuo) - 1))

    # ======================================================
    # ----------- Geração / Reparo de indivíduos -----------
    # ======================================================
    def _amostrar_nos_por_familia(self):
        """Sorteia nós respeitando visitas_familia por família."""
        individuo = [0]
        for l, familia in enumerate(self.familias):
            k = self.visitas_familia[l]
            # se a família tem exatamente k nós, pega todos; caso contrário, amostra
            if k >= len(familia):
                nos_sel = list(familia)
            else:
                nos_sel = random.sample(familia, k)
            individuo.extend(nos_sel)
        random.shuffle(individuo[1:])
        return [0] + individuo[1:] + [0]

    def gerar_individuo_aleatorio(self):
        return self._amostrar_nos_por_familia()

    def _insercao_mais_barata(self, rota: List[int], novo_no: int) -> List[int]:
        """Insere novo_no na melhor posição (rota contém 0...0)."""
        melhor_custo = float('inf')
        melhor_pos = 1
        for i in range(len(rota) - 1):
            a, b = rota[i], rota[i+1]
            delta = self._dist[a, novo_no] + self._dist[novo_no, b] - self._dist[a, b]
            if delta < melhor_custo:
                melhor_custo = delta
                melhor_pos = i + 1
        rota_corrigida = rota[:melhor_pos] + [novo_no] + rota[melhor_pos:]
        return rota_corrigida

    def _reparo_por_familia(self, individuo: List[int]) -> List[int]:
        """
        Repara um indivíduo para que exatamente visitas_familia[l] nós de cada família apareçam.
        Remove excedentes escolhendo remoções de menor impacto de custo e repõe faltantes por inserção barata.
        """
        for _ in range(self.config.repeticoes_reparo_max):
            cont = self._contagem_por_familia(individuo)
            ok = True
            # 1) remover excedentes
            for fidx, (count, need) in enumerate(zip(cont, self.visitas_familia)):
                if count > need:
                    ok = False
                    # candidatos para remoção: posições desses nós
                    posicoes = [i for i, no in enumerate(individuo[1:-1], start=1) if no in self._familia_sets[fidx]]
                    # compute melhor remoção (menor aumento de custo negativo = maior redução)
                    melhor_delta = float('inf')
                    melhor_pos = None
                    for p in posicoes:
                        a, x, b = individuo[p-1], individuo[p], individuo[p+1]
                        delta = (self._dist[a, b] - (self._dist[a, x] + self._dist[x, b]))
                        # delta < 0 significa que remover x AUMENTA custo; queremos menor custo final => maximizar delta
                        # mas como comparamos custos após remoção, tratamos como "custo efetivo" = -delta
                        custo_efetivo = -delta
                        if custo_efetivo < melhor_delta:
                            melhor_delta = custo_efetivo
                            melhor_pos = p
                    if melhor_pos is not None:
                        individuo = individuo[:melhor_pos] + individuo[melhor_pos+1:]
            if ok:
                # 2) adicionar faltantes
                cont = self._contagem_por_familia(individuo)  # recalc
                for fidx, (count, need) in enumerate(zip(cont, self.visitas_familia)):
                    if count < need:
                        ok = False
                        # nós disponíveis desta família que ainda não estão na rota
                        atuais = set(individuo[1:-1])
                        candidatos = [n for n in self.familias[fidx] if n not in atuais]
                        if not candidatos:
                            # se não houver candidato (edge case), reamostra total
                            return self.gerar_individuo_aleatorio()
                        # insere pelo critério de inserção mais barata
                        melhor_add = None
                        melhor_delta = float('inf')
                        melhor_pos = None
                        for cand in candidatos:
                            # avaliar inserção mais barata
                            # variação de custo ao inserir em cada aresta:
                            for i in range(len(individuo) - 1):
                                a, b = individuo[i], individuo[i+1]
                                delta = self._dist[a, cand] + self._dist[cand, b] - self._dist[a, b]
                                if delta < melhor_delta:
                                    melhor_delta = delta
                                    melhor_add = cand
                                    melhor_pos = i + 1
                        if melhor_add is not None:
                            individuo = individuo[:melhor_pos] + [melhor_add] + individuo[melhor_pos:]
            if ok:
                break
        # se ainda estiver inválido por algum motivo, recorre à geração aleatória (fallback raro)
        return individuo if self.validar_individuo(individuo) else self.gerar_individuo_aleatorio()

    def validar_individuo(self, individuo):
        if individuo[0] != 0 or individuo[-1] != 0:
            return False
        cont_fam = self._contagem_por_familia(individuo)
        return all(cont_fam[i] == self.visitas_familia[i] for i in range(len(cont_fam)))

    # ======================================================
    # ---------------- Inicialização -----------------------
    # ======================================================
    def _inicializar_boa(self) -> List[int]:
        """
        Seed “boa”: best-of-k amostras + uma passagem de inserção gulosa para reduzir custo.
        """
        k = self.config.bestofk_init
        candidatos = [self._amostrar_nos_por_familia() for _ in range(k)]
        # ajuste por inserção gulosa (nearest insertion) sobre a parte interna
        def nearest_insertion(rota):
            internos = rota[1:-1]
            if len(internos) <= 2:
                return rota
            # começa com nó mais próximo do depósito
            start = min(internos, key=lambda n: self._dist[0, n])
            tour = [0, start, 0]
            restantes = set(internos)
            restantes.remove(start)
            while restantes:
                # escolhe nó mais próximo do tour
                melhor_no = None
                melhor_dist = float('inf')
                for n in restantes:
                    d = min(self._dist[n, t] for t in tour if t != 0)
                    if d < melhor_dist:
                        melhor_dist = d
                        melhor_no = n
                # insere no mais barato lugar
                tour = self._insercao_mais_barata(tour, melhor_no)
                restantes.remove(melhor_no)
            return tour

        candidatos = [nearest_insertion(ind) for ind in candidatos]
        candidatos.sort(key=self.calcular_fitness)
        return candidatos[0]

    def inicializar_populacao(self):
        pop = []
        n_boa = int(self.tamanho_populacao * self.config.fracao_inicial_boas)
        for _ in range(n_boa):
            pop.append(self._inicializar_boa())
        while len(pop) < self.tamanho_populacao:
            pop.append(self.gerar_individuo_aleatorio())
        return pop

    # ======================================================
    # ------------------- Seleção --------------------------
    # ======================================================
    def selecao_torneio(self, populacao, fitness_scores, k=3):
        participantes = random.sample(list(zip(populacao, fitness_scores)), k)
        participantes.sort(key=lambda x: x[1])
        return participantes[0][0].copy()

    # ======================================================
    # ------------------- Crossover ------------------------
    # ======================================================
    def crossover_ox(self, pai1, pai2):
        if random.random() > self.taxa_crossover:
            return pai1.copy(), pai2.copy()

        seq1, seq2 = pai1[1:-1], pai2[1:-1]
        size = len(seq1)
        a, b = sorted(random.sample(range(size), 2))
        filho1, filho2 = [None] * size, [None] * size
        filho1[a:b], filho2[a:b] = seq1[a:b], seq2[a:b]

        def preencher(filho, ref):
            p = b
            for cidade in ref[b:] + ref[:b]:
                if cidade not in filho:
                    filho[p % size] = cidade
                    p += 1

        preencher(filho1, seq2)
        preencher(filho2, seq1)
        filho1 = [0] + filho1 + [0]
        filho2 = [0] + filho2 + [0]
        # reparo mais inteligente (mantém diversidade e validade)
        filho1 = self._reparo_por_familia(filho1)
        filho2 = self._reparo_por_familia(filho2)
        return filho1, filho2

    # ======================================================
    # -------------------- Mutação -------------------------
    # ======================================================
    def mutacao_inversao(self, individuo, taxa_mutacao):
        if random.random() > taxa_mutacao:
            return individuo
        i, j = sorted(random.sample(range(1, len(individuo) - 1), 2))
        individuo[i:j] = reversed(individuo[i:j])
        return individuo

    def mutacao_swap(self, individuo, taxa_mutacao):
        if random.random() > taxa_mutacao:
            return individuo
        i, j = sorted(random.sample(range(1, len(individuo) - 1), 2))
        individuo[i], individuo[j] = individuo[j], individuo[i]
        return individuo

    def mutacao_resample_familia(self, individuo, taxa_mutacao):
        """
        Troca um nó por outro da mesma família e insere na melhor posição.
        Mantém a contagem correta por família e permite explorar combinações de nós.
        """
        if random.random() > taxa_mutacao * self.config.prob_resample_family:
            return individuo

        internos = individuo[1:-1]
        # escolhe uma posição e identifica a família
        p = random.randint(1, len(individuo) - 2)
        no_antigo = individuo[p]
        fidx = self.familia_no.get(no_antigo, 0) - 1
        if fidx < 0:
            return individuo

        atuais = set(internos)
        candidatos = [n for n in self.familias[fidx] if n not in atuais or n == no_antigo]
        if len(candidatos) <= 1:
            return individuo

        # remove o nó antigo
        sem = individuo[:p] + individuo[p+1:]

        # escolhe melhor candidato para inserir
        melhor_no, melhor_delta, melhor_pos = None, float('inf'), None
        for cand in candidatos:
            if cand == no_antigo:
                continue
            for i in range(len(sem) - 1):
                a, b = sem[i], sem[i+1]
                delta = self._dist[a, cand] + self._dist[cand, b] - self._dist[a, b]
                if delta < melhor_delta:
                    melhor_delta = delta
                    melhor_no = cand
                    melhor_pos = i + 1
        if melhor_no is not None:
            individuo = sem[:melhor_pos] + [melhor_no] + sem[melhor_pos:]
        return individuo

    # ======================================================
    # ------------------- Busca Local ----------------------
    # ======================================================
    def _two_opt_once(self, rota: List[int]) -> Tuple[List[int], bool]:
        """Executa uma melhora 2-opt (uma passada) na parte interna da rota."""
        best_gain = 0.0
        best_i, best_k = None, None
        # ignorar 0 nas pontas
        for i in range(1, len(rota) - 3):
            a, b = rota[i-1], rota[i]
            for k in range(i+1, len(rota) - 1):
                c, d = rota[k], rota[k+1]
                # ganho (positivo) se trocarmos as arestas (a-b, c-d) por (a-c, b-d)
                delta = (self._dist[a, c] + self._dist[b, d]) - (self._dist[a, b] + self._dist[c, d])
                if delta < best_gain:  # delta negativo melhora custo
                    best_gain = delta
                    best_i, best_k = i, k
        if best_i is not None:
            nova = rota[:best_i] + list(reversed(rota[best_i:best_k+1])) + rota[best_k+1:]
            return nova, True
        return rota, False

    def two_opt(self, rota: List[int], max_iter=50) -> List[int]:
        """Aplica 2-opt até estagnar ou atingir max_iter."""
        atual = rota
        for _ in range(max_iter):
            atual, better = self._two_opt_once(atual)
            if not better:
                break
        return atual

    # ======================================================
    # -------------------- Evolução ------------------------
    # ======================================================
    def evoluir(self):
        populacao = self.inicializar_populacao()
        melhor_individuo, melhor_fitness = None, float('inf')
        sem_melhoria = 0
        limite_sem_melhoria = self.config.paciencia
        taxa_mutacao = self.taxa_mutacao_base

        for geracao in range(self.max_geracoes):
            fitness_scores = [self.calcular_fitness(ind) for ind in populacao]
            min_fit_idx = int(np.argmin(fitness_scores))
            melhor_atual = fitness_scores[min_fit_idx]

            melhora = melhor_fitness - melhor_atual
            if melhora > 1e-4:  # tolerância: 0.0001
                melhor_fitness = melhor_atual
                melhor_individuo = populacao[min_fit_idx].copy()
                sem_melhoria = 0
            else:
                sem_melhoria += 1

            self.historico_fitness.append(melhor_fitness)

            # elitismo
            pop_fit = sorted(zip(populacao, fitness_scores), key=lambda x: x[1])
            nova_pop = [ind for ind, _ in pop_fit[:self.elite_size]]

            # reprodução
            while len(nova_pop) < self.tamanho_populacao:
                pai1 = self.selecao_torneio(populacao, fitness_scores, k=3)
                pai2 = self.selecao_torneio(populacao, fitness_scores, k=3)

                f1, f2 = self.crossover_ox(pai1, pai2)

                # mutações estruturais (mantêm conjunto de nós)
                f1 = self.mutacao_inversao(f1, taxa_mutacao)
                if random.random() < self.config.prob_swap_mut:
                    f1 = self.mutacao_swap(f1, taxa_mutacao)
                f2 = self.mutacao_inversao(f2, taxa_mutacao)
                if random.random() < self.config.prob_swap_mut:
                    f2 = self.mutacao_swap(f2, taxa_mutacao)

                # mutação de reamostragem por família (troca um nó por outro da mesma família)
                f1 = self.mutacao_resample_familia(f1, taxa_mutacao)
                f2 = self.mutacao_resample_familia(f2, taxa_mutacao)

                # busca local (memetic)
                if random.random() < self.config.prob_2opt:
                    f1 = self.two_opt(f1, max_iter=50)
                if random.random() < self.config.prob_2opt:
                    f2 = self.two_opt(f2, max_iter=50)

                # reparo final por segurança
                f1 = self._reparo_por_familia(f1)
                f2 = self._reparo_por_familia(f2)

                nova_pop.extend([f1, f2])

            # política de imigrantes se estagnar
            if sem_melhoria >= limite_sem_melhoria:
                # injeta imigrantes aleatórios e aumenta mutação temporariamente
                qtd = int(self.tamanho_populacao * self.config.imigrantes)
                for _ in range(qtd):
                    nova_pop[-(_+1)] = self.gerar_individuo_aleatorio()
                taxa_mutacao = min(0.95, taxa_mutacao * self.config.mutation_boost)
                sem_melhoria = 0  # dá um "kick"
                # mantém o melhor até agora
                if melhor_individuo is not None:
                    nova_pop[0] = melhor_individuo.copy()
                print(f"⚡ Estagnação: injetando imigrantes e elevando mutação para {taxa_mutacao:.2f}")

            populacao = nova_pop[:self.tamanho_populacao]

            if geracao % 50 == 0:
                print(f"Geração {geracao}: melhor fitness = {melhor_fitness:.2f}")

        return melhor_individuo, melhor_fitness, self.historico_fitness

    # ======================================================
    # -------------------- Plotagens -----------------------
    # ======================================================
    def plotar_convergencia(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.historico_fitness, linewidth=2)
        plt.title('Convergência do Algoritmo Genético (melhor fitness)')
        plt.xlabel('Geração')
        plt.ylabel('Distância')
        plt.grid(True, alpha=0.4)
        plt.close()

    def plotar_rota(self, individuo, caminho: str = None, mostrar: bool = False):
        fig, ax = plt.subplots(figsize=(12, 8))

        # --- Plota as cidades por família ---
        cores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
        for l, familia in enumerate(self.familias):
            for no in familia:
                ax.scatter(self.coordenadas[no][0], self.coordenadas[no][1],
                           c=cores[l % len(cores)], s=100, alpha=0.7,
                           label=f'Família {l + 1}' if no == familia[0] else "")

        # --- Depósito ---
        ax.scatter(self.coordenadas[0][0], self.coordenadas[0][1],
                   c='black', s=200, marker='s', label='Depósito')

        # --- Rota ---
        for i in range(len(individuo) - 1):
            x1, y1 = self.coordenadas[individuo[i]]
            x2, y2 = self.coordenadas[individuo[i + 1]]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=2)

        # --- Índices ---
        for i, (x, y) in enumerate(self.coordenadas):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_title(f"Melhor Rota (Distância: {self.calcular_fitness(individuo):.2f})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if caminho:
            fig.savefig(caminho, dpi=300, bbox_inches='tight')
            print(f"📸 Rota salva em: {caminho}")
        if mostrar:
            plt.show()

        plt.close(fig)
