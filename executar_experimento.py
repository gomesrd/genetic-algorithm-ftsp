import time, os, csv
from datetime import datetime
from instancia_tsp import TSPInstance
from configuracao import GeneticAlgorithmConfig
from ag_tsp import GeneticAlgorithmTSP

class TSPExperiment:
    @staticmethod
    def resolver_instancia(arquivo: str, caminho_saida: str):
        inicio = time.time()
        try:
            instancia = TSPInstance(arquivo)
            config = GeneticAlgorithmConfig(len(instancia.coordenadas))
            ag = GeneticAlgorithmTSP(instancia.coordenadas, instancia.familias,
                                     instancia.v, instancia.familia_no, config)

            print(f"\n=== Executando instância: {arquivo} ===")
            melhor_individuo, melhor_fitness, _ = ag.evoluir()

            tempo_total = time.time() - inicio
            nome_instancia = os.path.basename(arquivo).replace(".tsp", "")

            with open(caminho_saida, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    nome_instancia, round(tempo_total, 2), round(melhor_fitness, 2),
                    config.tamanho_populacao, config.taxa_mutacao,
                    config.taxa_crossover, config.elite_size, config.max_geracoes
                ])

            print(f"✅ {nome_instancia} | Tempo: {tempo_total:.2f}s | Fitness: {melhor_fitness:.2f}")
            # Mostrar gráficos
            ag.plotar_convergencia()
            ag.plotar_rota(melhor_individuo)
        except Exception as e:
            print(f"❌ Erro ao executar {arquivo}: {e}")

    @staticmethod
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

        agora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("resultados", exist_ok=True)
        caminho_saida = os.path.join("resultados", f"resultados_ag_{agora}.csv")

        with open(caminho_saida, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "instancia", "tempo_execucao_s", "valor_otimo_ag",
                "tamanho_populacao", "taxa_mutacao", "taxa_crossover",
                "elite_size", "max_geracoes"
            ])

        for arquivo in arquivos:
            TSPExperiment.resolver_instancia(arquivo, caminho_saida)

        print("\n📄 Resultados salvos em:", caminho_saida)
