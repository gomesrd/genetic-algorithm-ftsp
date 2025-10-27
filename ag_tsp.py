import numpy as np
import random
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from configuracao import GeneticAlgorithmConfig


class GeneticAlgorithmTSP:
    def __init__(self, coordenadas, familias, visitas_familia, familia_no, config: GeneticAlgorithmConfig):

        self.coordenadas = coordenadas
        self.familias = familias
        self.visitas_familia = visitas_familia
        self.familia_no = familia_no
        self.config = config

        self.tamanho_populacao = config.tamanho_populacao
        self.taxa_mutacao_base = config.taxa_mutacao
        self.taxa_crossover = config.taxa_crossover
        self.elite_size = config.elite_size
        self.max_geracoes = config.max_geracoes

        self.historico_fitness = []

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        self._dist = self._precompute_dist()
        self._familia_sets = [set(f) for f in self.familias]

    def _precompute_dist(self):
        n = len(self.coordenadas)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            xi, yi = self.coordenadas[i]
            for j in range(i + 1, n):
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

    @staticmethod
    def _fitness_de(individuo, dist):
        s = 0.0
        for i in range(len(individuo) - 1):
            s += dist[individuo[i], individuo[i + 1]]
        return s

    def calcular_fitness(self, individuo):
        return self._fitness_de(individuo, self._dist)

    def _amostrar_nos_por_familia(self):
        individuo = [0]
        for l, familia in enumerate(self.familias):
            k = self.visitas_familia[l]
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
        melhor_custo = float('inf')
        melhor_pos = 1
        dist = self._dist
        for i in range(len(rota) - 1):
            a, b = rota[i], rota[i + 1]
            delta = dist[a, novo_no] + dist[novo_no, b] - dist[a, b]
            if delta < melhor_custo:
                melhor_custo = delta
                melhor_pos = i + 1
        return rota[:melhor_pos] + [novo_no] + rota[melhor_pos:]

    def _reparo_por_familia(self, individuo: List[int]) -> List[int]:
        dist = self._dist
        for _ in range(self.config.repeticoes_reparo_max):
            cont = self._contagem_por_familia(individuo)
            ok = True

            for fidx, (count, need) in enumerate(zip(cont, self.visitas_familia)):
                if count > need:
                    ok = False
                    posicoes = [i for i, no in enumerate(individuo[1:-1], start=1) if no in self._familia_sets[fidx]]
                    melhor_delta = float('inf')
                    melhor_pos = None
                    for p in posicoes:
                        a, x, b = individuo[p - 1], individuo[p], individuo[p + 1]
                        custo_efetivo = -(dist[a, b] - (dist[a, x] + dist[x, b]))
                        if custo_efetivo < melhor_delta:
                            melhor_delta = custo_efetivo
                            melhor_pos = p
                    if melhor_pos is not None:
                        individuo = individuo[:melhor_pos] + individuo[melhor_pos + 1:]

            if ok:
                cont = self._contagem_por_familia(individuo)
                for fidx, (count, need) in enumerate(zip(cont, self.visitas_familia)):
                    if count < need:
                        ok = False
                        atuais = set(individuo[1:-1])
                        candidatos = [n for n in self.familias[fidx] if n not in atuais]
                        if not candidatos:
                            return self.gerar_individuo_aleatorio()
                        melhor_add = None
                        melhor_delta = float('inf')
                        melhor_pos = None

                        for cand in candidatos:
                            for i in range(len(individuo) - 1):
                                a, b = individuo[i], individuo[i + 1]
                                delta = dist[a, cand] + dist[cand, b] - dist[a, b]
                                if delta < melhor_delta:
                                    melhor_delta = delta
                                    melhor_add = cand
                                    melhor_pos = i + 1
                        if melhor_add is not None:
                            individuo = individuo[:melhor_pos] + [melhor_add] + individuo[melhor_pos:]

            if ok:
                break

        return individuo if self.validar_individuo(individuo) else self.gerar_individuo_aleatorio()

    def validar_individuo(self, individuo):
        if individuo[0] != 0 or individuo[-1] != 0:
            return False
        cont_fam = self._contagem_por_familia(individuo)
        return all(cont_fam[i] == self.visitas_familia[i] for i in range(len(cont_fam)))

    def _inicializar_boa(self) -> List[int]:
        k = self.config.bestofk_init
        candidatos = [self._amostrar_nos_por_familia() for _ in range(k)]

        def nearest_insertion(rota):
            internos = rota[1:-1]
            if len(internos) <= 2:
                return rota
            dist = self._dist
            start = min(internos, key=lambda n: dist[0, n])
            tour = [0, start, 0]
            restantes = set(internos)
            restantes.remove(start)
            while restantes:
                melhor_no, melhor_dist = None, float('inf')
                for n in restantes:
                    d = min(dist[n, t] for t in tour if t != 0)
                    if d < melhor_dist:
                        melhor_dist = d
                        melhor_no = n
                tour = self._insercao_mais_barata(tour, melhor_no)
                restantes.remove(melhor_no)
            return tour

        candidatos = [nearest_insertion(ind) for ind in candidatos]
        candidatos.sort(key=self.calcular_fitness)
        return candidatos[0]

    def inicializar_populacao(self):
        pop = []
        n = len(self.coordenadas)
        frac_boas = (self.config.fracao_inicial_boas_grande if n > self.config.limite_inicializacao_boa
                     else self.config.fracao_inicial_boas)
        n_boa = int(self.tamanho_populacao * frac_boas)
        for _ in range(n_boa):
            pop.append(self._inicializar_boa())
        while len(pop) < self.tamanho_populacao:
            pop.append(self.gerar_individuo_aleatorio())
        return pop

    @staticmethod
    def _selecao_torneio_sobre_fit(pop_fit, k=3):
        participantes = random.sample(pop_fit, k)
        participantes.sort(key=lambda x: x[1])
        return participantes[0][0].copy()

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

        if not self.validar_individuo(filho1):
            filho1 = self._reparo_por_familia(filho1)
        if not self.validar_individuo(filho2):
            filho2 = self._reparo_por_familia(filho2)

        return filho1, filho2

    @staticmethod
    def _mutacao_inversao(individuo, taxa_mutacao):
        if random.random() > taxa_mutacao:
            return individuo
        i, j = sorted(random.sample(range(1, len(individuo) - 1), 2))
        individuo[i:j] = reversed(individuo[i:j])
        return individuo

    @staticmethod
    def _mutacao_swap(individuo, taxa_mutacao):
        if random.random() > taxa_mutacao:
            return individuo
        i, j = sorted(random.sample(range(1, len(individuo) - 1), 2))
        individuo[i], individuo[j] = individuo[j], individuo[i]
        return individuo

    def _mutacao_resample_familia(self, individuo, taxa_mutacao):
        if random.random() > taxa_mutacao * self.config.prob_resample_family:
            return individuo

        p = random.randint(1, len(individuo) - 2)
        no_antigo = individuo[p]
        fidx = self.familia_no.get(no_antigo, 0) - 1
        if fidx < 0:
            return individuo

        internos = individuo[1:-1]
        atuais = set(internos)
        candidatos = [n for n in self.familias[fidx] if n not in atuais or n == no_antigo]
        if len(candidatos) <= 1:
            return individuo

        sem = individuo[:p] + individuo[p + 1:]
        dist = self._dist
        melhor_no, melhor_delta, melhor_pos = None, float('inf'), None
        for cand in candidatos:
            if cand == no_antigo:
                continue
            for i in range(len(sem) - 1):
                a, b = sem[i], sem[i + 1]
                delta = dist[a, cand] + dist[cand, b] - dist[a, b]
                if delta < melhor_delta:
                    melhor_delta = delta
                    melhor_no = cand
                    melhor_pos = i + 1
        if melhor_no is not None:
            individuo = sem[:melhor_pos] + [melhor_no] + sem[melhor_pos:]
        return individuo

    def _two_opt_once(self, rota: List[int]) -> Tuple[List[int], bool]:
        dist = self._dist
        best_gain = 0.0
        best_i, best_k = None, None

        for i in range(1, len(rota) - 4, 2):
            a, b = rota[i - 1], rota[i]
            for k in range(i + 1, len(rota) - 1, 2):
                c, d = rota[k], rota[k + 1]
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                if delta < best_gain:
                    best_gain = delta
                    best_i, best_k = i, k
        if best_i is not None:
            nova = rota[:best_i] + list(reversed(rota[best_i:best_k + 1])) + rota[best_k + 1:]
            return nova, True
        return rota, False

    def two_opt(self, rota: List[int], max_iter=None) -> List[int]:
        if max_iter is None:
            max_iter = self.config.max_iter_2opt
        atual = rota
        for _ in range(max_iter):
            atual, better = self._two_opt_once(atual)
            if not better:
                break
        return atual

    def evoluir(self):
        base_pop = self.inicializar_populacao()
        pop_fit = [(ind, self._fitness_de(ind, self._dist)) for ind in base_pop]

        melhor_individuo, melhor_fitness = None, float('inf')
        sem_melhoria = 0
        taxa_mutacao = self.taxa_mutacao_base

        for geracao in range(self.max_geracoes):
            pop_fit.sort(key=lambda x: x[1])
            if pop_fit[0][1] + 1e-4 < melhor_fitness:
                melhor_individuo = pop_fit[0][0].copy()
                melhor_fitness = pop_fit[0][1]
                sem_melhoria = 0
            else:
                sem_melhoria += 1

            self.historico_fitness.append(melhor_fitness)

            nova_pop_fit = pop_fit[:self.elite_size]

            while len(nova_pop_fit) < self.tamanho_populacao:
                pai1 = self._selecao_torneio_sobre_fit(pop_fit, k=3)
                pai2 = self._selecao_torneio_sobre_fit(pop_fit, k=3)

                f1, f2 = self.crossover_ox(pai1, pai2)

                f1 = self._mutacao_inversao(f1, taxa_mutacao)
                if random.random() < self.config.prob_swap_mut:
                    f1 = self._mutacao_swap(f1, taxa_mutacao)
                f2 = self._mutacao_inversao(f2, taxa_mutacao)
                if random.random() < self.config.prob_swap_mut:
                    f2 = self._mutacao_swap(f2, taxa_mutacao)

                f1 = self._mutacao_resample_familia(f1, taxa_mutacao)
                f2 = self._mutacao_resample_familia(f2, taxa_mutacao)

                if random.random() < self.config.prob_2opt:
                    f1 = self.two_opt(f1, max_iter=self.config.max_iter_2opt)
                if random.random() < self.config.prob_2opt:
                    f2 = self.two_opt(f2, max_iter=self.config.max_iter_2opt)

                fit1 = self._fitness_de(f1, self._dist)
                fit2 = self._fitness_de(f2, self._dist)

                nova_pop_fit.append((f1, fit1))
                if len(nova_pop_fit) < self.tamanho_populacao:
                    nova_pop_fit.append((f2, fit2))

            pop_fit = nova_pop_fit[:self.tamanho_populacao]

            if geracao % 50 == 0:
                print(f"Geração {geracao}: melhor fitness = {melhor_fitness:.2f}")

            w = self.config.early_stop_window
            if geracao > self.config.early_stop_min_gen and len(self.historico_fitness) >= w:
                janela = self.historico_fitness[-w:]
                if np.std(janela) < self.config.early_stop_std_eps:
                    print(f"⏹ Convergência detectada na geração {geracao}. Encerrando cedo.")
                    break

        return melhor_individuo, melhor_fitness, self.historico_fitness

    def verificar_rota(self, individuo):
        if not self.validar_individuo(individuo):
            return False
        for no in individuo:
            if no < 0 or no >= len(self.coordenadas):
                print(f"⚠️ Nó inválido encontrado: {no}")
                return False
        print("✅ Rota válida: todos os nós existem e respeitam as famílias.")
        return True


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

        cores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
        for l, familia in enumerate(self.familias):
            for no in familia:
                ax.scatter(self.coordenadas[no][0], self.coordenadas[no][1],
                           c=cores[l % len(cores)], s=100, alpha=0.7,
                           label=f'Família {l + 1}' if no == familia[0] else "")

        ax.scatter(self.coordenadas[0][0], self.coordenadas[0][1],
                   c='black', s=200, marker='s', label='Depósito')

        for i in range(len(individuo) - 1):
            x1, y1 = self.coordenadas[individuo[i]]
            x2, y2 = self.coordenadas[individuo[i + 1]]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=2)

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