class GeneticAlgorithmConfig:
    """
    Define as configurações do Algoritmo Genético.
    Parâmetros novos possuem defaults conservadores.
    """
    def __init__(self, num_cidades: int, seed: int = None):
        # -- parâmetros clássicos
        if num_cidades <= 100:
            self.tamanho_populacao = 160
            self.taxa_mutacao = 0.15            # base; adaptativa durante o run
            self.taxa_crossover = 0.9
            self.elite_size = 8
            self.max_geracoes = 1000
        else:
            self.tamanho_populacao = 300
            self.taxa_mutacao = 0.22
            self.taxa_crossover = 0.92
            self.elite_size = 12
            self.max_geracoes = 1000

        # -- novos botões úteis
        self.prob_2opt = 0.15                   # probabilidade de aplicar 2-opt num filho
        self.prob_swap_mut = 0.4               # probabilidade de aplicar swap mutation além da inversão
        self.prob_resample_family = 0.25       # chance de tentar trocar um nó por outro da mesma família
        self.bestofk_init = 6                 # número de amostras para seed "boa" (best-of-k)
        self.fracao_inicial_boas = 0.25        # % da população com inicialização melhor
        self.paciencia = 300                   # gerações sem melhora antes de ação corretiva
        self.imigrantes = 0.15                 # fração de imigrantes aleatórios quando há estagnação
        self.mutation_boost = 1.8              # multiplicador de mutação durante estagnação
        self.repeticoes_reparo_max = 3         # limite para tentativas de reparo
        self.seed = seed                        # opcional: reprodutibilidade

