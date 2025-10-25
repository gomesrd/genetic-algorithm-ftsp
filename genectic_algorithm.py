import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import os
import random
import math
from typing import List, Tuple, Dict
from datetime import datetime

class AlgoritmoGeneticoTSP:
    def __init__(self, coordenadas: List[Tuple[float, float]],
                 familias: List[List[int]],
                 visitas_familia: List[int],
                 familia_no: Dict[int, int],
                 tamanho_populacao: int = 100,
                 taxa_mutacao: float = 0.02,
                 taxa_crossover: float = 0.8,
                 elite_size: int = 20,
                 max_geracoes: int = 500):

        self.coordenadas = coordenadas
        self.familias = familias
        self.visitas_familia = visitas_familia
        self.familia_no = familia_no
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.elite_size = elite_size
        self.max_geracoes = max_geracoes
        self.historico_fitness = []
        self.m = len(coordenadas) - 1

    # ------------------------- utilitários -------------------------
    def distancia(self, i, j):
        return math.sqrt((self.coordenadas[i][0] - self.coordenadas[j][0]) ** 2 +
                         (self.coordenadas[i][1] - self.coordenadas[j][1]) ** 2)

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
        if self.validar_individuo(individuo):
            return individuo
        return self.gerar_individuo_aleatorio()

    # ------------------------- inicialização -------------------------
    def inicializar_populacao(self):
        return [self.gerar_individuo_aleatorio() for _ in range(self.tamanho_populacao)]

    # ------------------------- seleção -------------------------
    def selecao_roleta(self, populacao, fitness_scores):
        """Seleção proporcional ao inverso do fitness (quanto menor, melhor)."""
        inv_fit = [1.0 / f for f in fitness_scores]
        soma = sum(inv_fit)
        probs = [f / soma for f in inv_fit]
        idx = np.random.choice(len(populacao), p=probs)
        return populacao[idx].copy()

    def selecao_torneio(self, populacao, fitness_scores, k=3):
        participantes = random.sample(list(zip(populacao, fitness_scores)), k)
        participantes.sort(key=lambda x: x[1])
        return participantes[0][0].copy()

    # ------------------------- crossover -------------------------
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

    # ------------------------- mutação -------------------------
    def mutacao_inversao(self, individuo):
        """Mutação por inversão de segmento."""
        if random.random() > self.taxa_mutacao:
            return individuo
        i, j = sorted(random.sample(range(1, len(individuo) - 1), 2))
        individuo[i:j] = reversed(individuo[i:j])
        return self.reparar_individuo(individuo)

    # ------------------------- evolução principal -------------------------
    def evoluir(self):
        populacao = self.inicializar_populacao()
        melhor_individuo = None
        melhor_fitness = float('inf')
        sem_melhoria = 0
        limite_sem_melhoria = 300  # 🔹 Parar se não houver melhora em X gerações

        for geracao in range(self.max_geracoes):
            fitness_scores = [self.calcular_fitness(ind) for ind in populacao]
            min_fit_idx = np.argmin(fitness_scores)
            melhor_atual = fitness_scores[min_fit_idx]

            # --- Verifica se houve melhora ---
            if melhor_atual + 1e-6 < melhor_fitness:  # pequena tolerância numérica
                melhor_fitness = melhor_atual
                melhor_individuo = populacao[min_fit_idx].copy()
                sem_melhoria = 0
            else:
                sem_melhoria += 1

            self.historico_fitness.append(melhor_fitness)

            # --- elitismo ---
            pop_fit = list(zip(populacao, fitness_scores))
            pop_fit.sort(key=lambda x: x[1])
            nova_pop = [ind for ind, _ in pop_fit[:self.elite_size]]

            # --- reprodução ---
            while len(nova_pop) < self.tamanho_populacao:
                pai1 = self.selecao_torneio(populacao, fitness_scores)
                pai2 = self.selecao_torneio(populacao, fitness_scores)
                f1, f2 = self.crossover_ox(pai1, pai2)
                f1 = self.mutacao_inversao(f1)
                f2 = self.mutacao_inversao(f2)
                nova_pop.extend([f1, f2])

            populacao = nova_pop[:self.tamanho_populacao]

            # --- mutação adaptativa ---
            if sem_melhoria == 50:
                self.taxa_mutacao = min(self.taxa_mutacao * 1.3, 0.5)
            elif sem_melhoria == 100:
                self.taxa_mutacao = 0.2
            elif sem_melhoria == 200:
                self.taxa_mutacao = 0.3

            # --- reinicialização parcial ---
            if sem_melhoria in [150, 250]:
                qtd = int(0.2 * self.tamanho_populacao)
                for _ in range(qtd):
                    idx = random.randint(0, self.tamanho_populacao - 1)
                    populacao[idx] = self.gerar_individuo_aleatorio()
                print(f"⚠️ Reinicialização parcial na geração {geracao}.")
                # (não zera o sem_melhoria aqui — continua contando!)

            # --- critério de parada ---
            if sem_melhoria >= limite_sem_melhoria:
                print(
                    f"⏹ Parando na geração {geracao}: nenhuma melhora em {limite_sem_melhoria} gerações consecutivas.")
                break

            if geracao % 50 == 0:
                print(f"Geração {geracao}: Melhor fitness = {melhor_fitness:.2f} | mut={self.taxa_mutacao:.2f}")

        return melhor_individuo, melhor_fitness, self.historico_fitness

    # ------------------------- plotagem -------------------------
    def plotar_convergencia(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.historico_fitness)
        plt.title('Convergência do Algoritmo Genético (melhor fitness)')
        plt.xlabel('Geração')
        plt.ylabel('Distância')
        plt.grid(True)
        plt.show()
        plt.close()

    def plotar_rota(self, individuo: List[int]):
        """Plota a rota encontrada"""
        plt.figure(figsize=(12, 8))

        # Plota todas as cidades
        x_coords = [self.coordenadas[i][0] for i in range(len(self.coordenadas))]
        y_coords = [self.coordenadas[i][1] for i in range(len(self.coordenadas))]

        # Plota cidades por família
        cores = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for l, familia in enumerate(self.familias):
            for no in familia:
                plt.scatter(self.coordenadas[no][0], self.coordenadas[no][1],
                           c=cores[l % len(cores)], s=100, alpha=0.7,
                           label=f'Família {l+1}' if no == familia[0] else "")

        # Plota depósito
        plt.scatter(self.coordenadas[0][0], self.coordenadas[0][1],
                   c='black', s=200, marker='s', label='Depósito')

        # Plota rota
        for i in range(len(individuo) - 1):
            x1, y1 = self.coordenadas[individuo[i]]
            x2, y2 = self.coordenadas[individuo[i + 1]]
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.6, linewidth=2)

            # Adiciona setas
            dx, dy = x2 - x1, y2 - y1
            plt.arrow(x1, y1, dx * 0.8, dy * 0.8,
                     head_width=2, head_length=2, fc='black', ec='black', alpha=0.6)

        # Numera os nós
        for i, (x, y) in enumerate(self.coordenadas):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

        plt.title(f'Melhor Rota Encontrada (Distância: {self.calcular_fitness(individuo):.2f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        return None


def ler_tsp(arquivo):
    """Função para ler arquivo TSP (mantida igual ao original)"""
    coordenadas = []
    ler_coordenadas = False
    ler_familias = False
    aux = []
    v = []
    n = []

    try:
        with open(arquivo, 'r') as f:
            conteudo = f.readlines()
    except:
        conteudo = arquivo.split('\n')

    for linha in conteudo:
        linha = linha.strip()

        if linha == 'NODE_COORD_SECTION':
            ler_coordenadas = True
            continue
        if linha == 'FAMILY_SECTION':
            ler_familias = True
            ler_coordenadas = False
            continue
        if linha == 'EOF':
            ler_familias = False
            continue

        if ler_coordenadas and linha:
            partes = linha.split()
            x = float(partes[1])
            y = float(partes[2])
            coordenadas.append((x, y))
        else:
            if ler_familias and linha:
                partes = linha.split()
                aux2 = (int(partes[0]), int(partes[1]))
                aux.append(aux2)

    L = aux[0][0]
    V = aux[0][1]

    for elemento in aux[1:]:
        n.append(elemento[0])
        v.append(elemento[1])

    return coordenadas, L, V, n, v


def resolver_instancia(arquivo: str, caminho_saida: str):
    inicio = time.time()  # Inicia cronômetro

    try:
        # === Lê dados do arquivo ===
        C, L, V, n, v = ler_tsp(arquivo)
        m = len(C) - 1

        # === Constrói estruturas ===
        F = []
        Fi = dict()
        inicio_familia = 1
        for l in range(L):
            aux = list(range(inicio_familia, inicio_familia + n[l]))
            for no in aux:
                Fi[no] = l + 1
            inicio_familia = aux[-1] + 1
            F.append(aux)

        print(f"\n=== Executando instância: {arquivo} ===")

        # === Configurações do AG ===
        # config_ag = {
        #     "tamanho_populacao": 300,
        #     "taxa_mutacao": 0.15,
        #     "taxa_crossover": 0.8,
        #     "elite_size": 20,
        #     "max_geracoes": 1500
        # }

        if len(C) <= 100:
            config_ag = {
                "tamanho_populacao": 100,
                "taxa_mutacao": 0.15,
                "taxa_crossover": 0.8,
                "elite_size": 5,
                "max_geracoes": 2000
            }
        else:
            config_ag = {
                "tamanho_populacao": 400,
                "taxa_mutacao": 0.15,
                "taxa_crossover": 0.8,
                "elite_size": 20,
                "max_geracoes": 2000
            }

        # === Executa o algoritmo ===
        ag = AlgoritmoGeneticoTSP(
            coordenadas=C,
            familias=F,
            visitas_familia=v,
            familia_no=Fi,
            **config_ag
        )

        print("Executando Algoritmo Genético...")
        melhor_individuo, melhor_fitness, historico = ag.evoluir()

        tempo_total = time.time() - inicio  # Tempo total em segundos
        nome_instancia = os.path.basename(arquivo).replace(".tsp", "")

        # === Salva resultado em CSV ===
        with open(caminho_saida, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                nome_instancia,
                round(tempo_total, 2),
                round(melhor_fitness, 2),
                config_ag["tamanho_populacao"],
                config_ag["taxa_mutacao"],
                config_ag["taxa_crossover"],
                config_ag["elite_size"],
                config_ag["max_geracoes"]
            ])

        # === Exibe no terminal ===
        print(f"\n✅ Instância: {nome_instancia}")
        print(f"Tempo total: {tempo_total:.2f} segundos")
        print(f"Melhor distância: {melhor_fitness:.2f}")

        # === Plota ===
        if ag.validar_individuo(melhor_individuo):
            print("✓ Solução é válida!")
        else:
            print("✗ Solução é inválida!")

        # ag.plotar_convergencia()
        ag.plotar_rota(melhor_individuo)

    except Exception as e:
        print(f"\nErro ao executar {arquivo}: {e}")
        print("Pulando instância.")


def main():
    arquivos = [
         # '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/a280_20_1001_1001_2.tsp',
        # '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/att48_5_1001_1001_2.tsp',
        # '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/att48_5_1001_1002_2.tsp',
        # '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/att48_5_1001_1003_2.tsp',
        # '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/bier127_10_1001_1001_2.tsp',
        # '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/bier127_10_1001_1002_2.tsp',
        # '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/bier127_10_1001_1003_2.tsp',
        '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/burma14_3_1001_1001_2.tsp',
        '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/burma14_3_1001_1002_2.tsp',
        '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/burma14_3_1001_1003_2.tsp'
    ]

    # Gera string com data e hora atuais
    agora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Cria nome de arquivo com timestamp
    caminho_saida = os.path.join("resultados", f"resultados_ag_{agora}.csv")

    # Cria o CSV com cabeçalho
    with open(caminho_saida, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instancia", "tempo_execucao_s", "valor_otimo_ag",
            "tamanho_populacao", "taxa_mutacao", "taxa_crossover",
            "elite_size", "max_geracoes"
        ])

    # Executa todas as instâncias
    for arquivo in arquivos:
        resolver_instancia(arquivo, caminho_saida)

    print("\n✅ Resultados salvos em:", caminho_saida)


if __name__ == "__main__":
    main()
