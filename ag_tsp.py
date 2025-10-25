import numpy as np
import random
import math
import matplotlib.pyplot as plt

class GeneticAlgorithmTSP:
    def __init__(self, coordenadas, familias, visitas_familia, familia_no, config):
        self.coordenadas = coordenadas
        self.familias = familias
        self.visitas_familia = visitas_familia
        self.familia_no = familia_no
        self.config = config

        self.tamanho_populacao = config.tamanho_populacao
        self.taxa_mutacao = config.taxa_mutacao
        self.taxa_crossover = config.taxa_crossover
        self.elite_size = config.elite_size
        self.max_geracoes = config.max_geracoes
        self.historico_fitness = []

    # ======================================================
    # ------------------ Métodos utilitários ----------------
    # ======================================================
    def distancia(self, i, j):
        return math.dist(self.coordenadas[i], self.coordenadas[j])

    def gerar_individuo_aleatorio(self):
        individuo = [0]
        for l, familia in enumerate(self.familias):
            nos_sel = random.sample(familia, self.visitas_familia[l])
            individuo.extend(nos_sel)
        random.shuffle(individuo[1:])
        return [0] + individuo[1:] + [0]

    def calcular_fitness(self, individuo):
        return sum(self.distancia(individuo[i], individuo[i + 1]) for i in range(len(individuo) - 1))

    def validar_individuo(self, individuo):
        if individuo[0] != 0 or individuo[-1] != 0:
            return False
        cont_fam = [0] * len(self.familias)
        for no in individuo[1:-1]:
            if no in self.familia_no:
                fidx = self.familia_no[no] - 1
                cont_fam[fidx] += 1
        return all(cont_fam[i] == self.visitas_familia[i] for i in range(len(cont_fam)))

    def reparar_individuo(self, individuo):
        return individuo if self.validar_individuo(individuo) else self.gerar_individuo_aleatorio()

    # ======================================================
    # ---------------- Inicialização & Seleção --------------
    # ======================================================
    def inicializar_populacao(self):
        return [self.gerar_individuo_aleatorio() for _ in range(self.tamanho_populacao)]

    def selecao_torneio(self, populacao, fitness_scores, k=3):
        participantes = random.sample(list(zip(populacao, fitness_scores)), k)
        participantes.sort(key=lambda x: x[1])
        return participantes[0][0].copy()

    # ======================================================
    # ------------------- Crossover -------------------------
    # ======================================================
    def crossover_ox(self, pai1, pai2):
        if random.random() > self.taxa_crossover:
            return pai1.copy(), pai2.copy()

        seq1, seq2 = pai1[1:-1], pai2[1:-1]
        size = len(seq1)
        start, end = sorted(random.sample(range(size), 2))
        filho1, filho2 = [None] * size, [None] * size
        filho1[start:end], filho2[start:end] = seq1[start:end], seq2[start:end]

        def preencher(filho, ref):
            p = end
            for cidade in ref[end:] + ref[:end]:
                if cidade not in filho:
                    filho[p % size] = cidade
                    p += 1

        preencher(filho1, seq2)
        preencher(filho2, seq1)
        filho1 = [0] + filho1 + [0]
        filho2 = [0] + filho2 + [0]
        return self.reparar_individuo(filho1), self.reparar_individuo(filho2)

    # ======================================================
    # -------------------- Mutação --------------------------
    # ======================================================
    def mutacao_inversao(self, individuo):
        if random.random() > self.taxa_mutacao:
            return individuo
        i, j = sorted(random.sample(range(1, len(individuo) - 1), 2))
        individuo[i:j] = reversed(individuo[i:j])
        return self.reparar_individuo(individuo)

    # ======================================================
    # -------------------- Evolução -------------------------
    # ======================================================
    def evoluir(self):
        populacao = self.inicializar_populacao()
        melhor_individuo, melhor_fitness = None, float('inf')
        sem_melhoria = 0
        limite_sem_melhoria = 300

        for geracao in range(self.max_geracoes):
            fitness_scores = [self.calcular_fitness(ind) for ind in populacao]
            min_fit_idx = np.argmin(fitness_scores)
            melhor_atual = fitness_scores[min_fit_idx]

            if melhor_atual + 1e-6 < melhor_fitness:
                melhor_fitness = melhor_atual
                melhor_individuo = populacao[min_fit_idx].copy()
                sem_melhoria = 0
            else:
                sem_melhoria += 1

            self.historico_fitness.append(melhor_fitness)
            pop_fit = sorted(zip(populacao, fitness_scores), key=lambda x: x[1])
            nova_pop = [ind for ind, _ in pop_fit[:self.elite_size]]

            while len(nova_pop) < self.tamanho_populacao:
                pai1 = self.selecao_torneio(populacao, fitness_scores)
                pai2 = self.selecao_torneio(populacao, fitness_scores)
                f1, f2 = self.crossover_ox(pai1, pai2)
                f1 = self.mutacao_inversao(f1)
                f2 = self.mutacao_inversao(f2)
                nova_pop.extend([f1, f2])

            populacao = nova_pop[:self.tamanho_populacao]

            if sem_melhoria >= limite_sem_melhoria:
                print(f"⏹ Parando na geração {geracao}: sem melhora.")
                break

            if geracao % 50 == 0:
                print(f"Geração {geracao}: melhor fitness = {melhor_fitness:.2f}")

        return melhor_individuo, melhor_fitness, self.historico_fitness

    # ======================================================
    # -------------------- Plotagens ------------------------
    # ======================================================
    def plotar_convergencia(self):
        """Plota o histórico de convergência do algoritmo genético."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.historico_fitness, color='blue', linewidth=2)
        plt.title('Convergência do Algoritmo Genético (melhor fitness)')
        plt.xlabel('Geração')
        plt.ylabel('Distância')
        plt.grid(True, alpha=0.4)
        # plt.show()
        plt.close()

    def plotar_rota(self, individuo, caminho: str = None, mostrar: bool = False):
        """Plota a rota encontrada e opcionalmente salva em arquivo."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # --- Plota as cidades ---
        cores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
        for l, familia in enumerate(self.familias):
            for no in familia:
                ax.scatter(self.coordenadas[no][0], self.coordenadas[no][1],
                           c=cores[l % len(cores)], s=100, alpha=0.7,
                           label=f'Família {l + 1}' if no == familia[0] else "")

        # --- Plota o depósito ---
        ax.scatter(self.coordenadas[0][0], self.coordenadas[0][1],
                   c='black', s=200, marker='s', label='Depósito')

        # --- Plota a rota ---
        for i in range(len(individuo) - 1):
            x1, y1 = self.coordenadas[individuo[i]]
            x2, y2 = self.coordenadas[individuo[i + 1]]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=2)

        # --- Numera os nós ---
        for i, (x, y) in enumerate(self.coordenadas):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_title(f"Melhor Rota (Distância: {self.calcular_fitness(individuo):.2f})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- salva ou mostra ---
        if caminho:
            fig.savefig(caminho, dpi=300, bbox_inches='tight')
            print(f"📸 Rota salva em: {caminho}")
        if mostrar:
            plt.show()

        plt.close(fig)
