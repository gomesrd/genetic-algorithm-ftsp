import time, os, csv, statistics
from datetime import datetime
from instancia_tsp import TSPInstance
from configuracao import GeneticAlgorithmConfig
from ag_tsp import GeneticAlgorithmTSP
import matplotlib.pyplot as plt


class TSPExperiment:
    @staticmethod
    def resolver_instancia(arquivo: str, caminho_saida: str, pasta_individual_base: str, repeticoes: int = 1):
        global config
        nome_instancia = os.path.basename(arquivo).replace(".tsp", "")
        print(f"\n=== Executando instância: {nome_instancia} ({repeticoes} execuções) ===")

        pasta_instancia = os.path.join(pasta_individual_base, nome_instancia)
        os.makedirs(pasta_instancia, exist_ok=True)

        fitness_list = []
        tempo_list = []

        for r in range(1, repeticoes + 1):
            print(f"\n➡️ Execução {r}/{repeticoes} da instância {nome_instancia}...")

            inicio = time.time()
            instancia = TSPInstance(arquivo)
            config = GeneticAlgorithmConfig(len(instancia.coordenadas))
            ag = GeneticAlgorithmTSP(
                instancia.coordenadas, instancia.familias, instancia.v, instancia.familia_no, config
            )

            melhor_individuo, melhor_fitness, historico = ag.evoluir()
            assert ag.validar_individuo(melhor_individuo), "Rota final inválida!"
            assert ag.verificar_rota(melhor_individuo), "Rota final inválida!"
            tempo_total = time.time() - inicio

            nome_execucao = f"{nome_instancia}_exec{r}"
            caminho_exec = os.path.join(pasta_instancia, nome_execucao)
            os.makedirs(caminho_exec, exist_ok=True)

            caminho_hist = os.path.join(caminho_exec, f"{nome_execucao}_historico.csv")
            with open(caminho_hist, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["geracao", "melhor_fitness"])
                for i, fit in enumerate(historico):
                    writer.writerow([i + 1, round(fit, 6)])

            fig1 = plt.figure(figsize=(10, 6))
            plt.plot(historico, color='blue', linewidth=2)
            plt.title(f"Convergência - {nome_execucao}")
            plt.xlabel("Geração")
            plt.ylabel("Melhor Fitness")
            plt.grid(True, alpha=0.4)
            fig1.savefig(os.path.join(caminho_exec, f"{nome_execucao}_convergencia.png"))
            plt.close(fig1)

            ag.plotar_rota(
                melhor_individuo,
                caminho=os.path.join(caminho_exec, f"{nome_execucao}_rota.png")
            )

            fitness_list.append(melhor_fitness)
            tempo_list.append(tempo_total)
            print("\n📍 Coordenadas da melhor rota encontrada:")
            for no in melhor_individuo:
                x, y = ag.coordenadas[no]
                print(f"Nó {no:3d} -> ({x:.2f}, {y:.2f})")

            print(f"✅ Execução {r} concluída | Tempo: {tempo_total:.2f}s | Fitness: {melhor_fitness:.2f}")

        media_fitness = statistics.mean(fitness_list)
        desvio_fitness = statistics.stdev(fitness_list) if len(fitness_list) > 1 else 0.0
        melhor_fitness_absoluto = min(fitness_list)

        caminho_stats = os.path.join(pasta_instancia, f"{nome_instancia}_resumo.csv")
        with open(caminho_stats, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["execucao", "tempo_s", "melhor_fitness"])
            for i, (t, fval) in enumerate(zip(tempo_list, fitness_list), start=1):
                writer.writerow([i, round(t, 2), round(fval, 2)])
            writer.writerow([])
            writer.writerow(["Média", round(statistics.mean(tempo_list), 2), round(media_fitness, 2)])
            writer.writerow(["Desvio Padrão", "-", round(desvio_fitness, 2)])
            writer.writerow(["Melhor Fitness Absoluto", "-", round(melhor_fitness_absoluto, 2)])

        with open(caminho_saida, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                nome_instancia,
                instancia.best_bound,
                round(statistics.mean(tempo_list), 2),
                round(media_fitness, 2),
                round(desvio_fitness, 2),
                round(melhor_fitness_absoluto, 2),
                round(((instancia.best_bound - melhor_fitness_absoluto) / (100*instancia.best_bound) ), 4),
                config.tamanho_populacao,
                config.taxa_crossover,
                config.elite_size,
                config.max_geracoes
            ])

        print(f"\n📊 Estatísticas salvas em {caminho_stats}")
        print(f"📈 Média: {media_fitness:.2f} | Desvio: {desvio_fitness:.2f} | "
              f" Melhor fitness: {melhor_fitness_absoluto:.2f}")

    @staticmethod
    def main():
        arquivos = [
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/att48_5_1001_1001_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/att48_5_1001_1002_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/att48_5_1001_1003_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/bier127_10_1001_1001_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/bier127_10_1001_1002_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/bier127_10_1001_1003_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/burma14_3_1001_1001_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/burma14_3_1001_1002_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/burma14_3_1001_1003_2.tsp',
            '/Users/rdsgomes/Library/CloudStorage/GoogleDrive-dougsk8pg@gmail.com/My Drive/Development/Ads_fatec/TCC/burma14_3_1001_1003_2/a280_20_1001_1001_2.tsp'
        ]

        agora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pasta_resultados = "resultados"
        pasta_individuais = os.path.join(pasta_resultados, "individuais", agora)
        os.makedirs(pasta_individuais, exist_ok=True)

        caminho_saida = os.path.join(pasta_resultados, f"resultados_ag_{agora}.csv")

        with open(caminho_saida, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "instancia",
                "best_bound_pl",
                "tempo_medio_s",
                "media_fitness",
                "desvio_padrao_fitness",
                "melhor_fitness_absoluto",
                "% PL X AG",
                "tamanho_populacao",
                "taxa_mutacao",
                "taxa_crossover",
                "elite_size",
                "max_geracoes"
            ])

        for arquivo in arquivos:
            TSPExperiment.resolver_instancia(arquivo, caminho_saida, pasta_individuais, repeticoes=5)

        print(f"\n📄 Resultados gerais: {caminho_saida}")
        print(f"📁 Resultados individuais: {pasta_individuais}")
