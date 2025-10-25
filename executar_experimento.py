import time, os, csv
from datetime import datetime
from instancia_tsp import TSPInstance
from configuracao import GeneticAlgorithmConfig
from ag_tsp import GeneticAlgorithmTSP
import matplotlib.pyplot as plt


class TSPExperiment:
    @staticmethod
    def resolver_instancia(arquivo: str, caminho_saida: str, pasta_individual_base: str):
        inicio = time.time()
        try:
            instancia = TSPInstance(arquivo)
            config = GeneticAlgorithmConfig(len(instancia.coordenadas))
            ag = GeneticAlgorithmTSP(instancia.coordenadas, instancia.familias,
                                     instancia.v, instancia.familia_no, config)

            nome_instancia = os.path.basename(arquivo).replace(".tsp", "")
            print(f"\n=== Executando instância: {nome_instancia} ===")

            # Executa o algoritmo
            melhor_individuo, melhor_fitness, historico = ag.evoluir()
            tempo_total = time.time() - inicio

            # --- salva no CSV geral ---
            with open(caminho_saida, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    nome_instancia, round(tempo_total, 2), round(melhor_fitness, 2),
                    config.tamanho_populacao, config.taxa_mutacao,
                    config.taxa_crossover, config.elite_size, config.max_geracoes
                ])

            # --- cria pasta individual ---
            pasta_instancia = os.path.join(pasta_individual_base, nome_instancia)
            os.makedirs(pasta_instancia, exist_ok=True)

            # --- salva histórico de fitness ---
            caminho_historico = os.path.join(pasta_instancia, f"{nome_instancia}_historico.csv")
            with open(caminho_historico, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["geracao", "melhor_fitness"])
                for i, fit in enumerate(historico):
                    writer.writerow([i + 1, round(fit, 6)])

            # --- salva gráfico de convergência ---
            fig1 = plt.figure(figsize=(10, 6))
            plt.plot(historico, color='blue', linewidth=2)
            plt.title(f"Convergência - {nome_instancia}")
            plt.xlabel("Geração")
            plt.ylabel("Melhor Fitness")
            plt.grid(True, alpha=0.4)
            fig1.savefig(os.path.join(pasta_instancia, f"{nome_instancia}_convergencia.png"))
            plt.close(fig1)

            # --- salva gráfico da rota ---
            ag.plotar_rota(
                melhor_individuo,
                caminho=os.path.join(pasta_instancia, f"{nome_instancia}_rota.png")
            )
            # --- mensagens ---
            print(f"✅ {nome_instancia} | Tempo: {tempo_total:.2f}s | Fitness: {melhor_fitness:.2f}")
            print(f"📈 Histórico salvo em: {caminho_historico}")

        except Exception as e:
            print(f"❌ Erro ao executar {arquivo}: {e}")

    @staticmethod
    def main():
        arquivos = [
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/burma14_3_1001_1001_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/burma14_3_1001_1002_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/burma14_3_1001_1003_2.tsp'
        ]

        # === cria estrutura de diretórios ===
        agora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pasta_resultados = "resultados"
        pasta_individuais = os.path.join(pasta_resultados, "individuais", agora)
        os.makedirs(pasta_individuais, exist_ok=True)

        caminho_saida = os.path.join(pasta_resultados, f"resultados_ag_{agora}.csv")

        # === cria arquivo CSV geral ===
        with open(caminho_saida, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "instancia", "tempo_execucao_s", "valor_otimo_ag",
                "tamanho_populacao", "taxa_mutacao", "taxa_crossover",
                "elite_size", "max_geracoes"
            ])

        # === executa cada instância ===
        for arquivo in arquivos:
            TSPExperiment.resolver_instancia(arquivo, caminho_saida, pasta_individuais)

        print(f"\n📄 Resultados gerais: {caminho_saida}")
        print(f"📁 Resultados individuais: {pasta_individuais}")
