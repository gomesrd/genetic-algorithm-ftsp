from typing import Optional

class GeneticAlgorithmConfig:

    def __init__(self, num_cidades: int, seed: Optional[int] = None):
        if num_cidades <= 100:
            self.tamanho_populacao = 140
            self.taxa_mutacao = 0.15
            self.taxa_crossover = 0.90
            self.elite_size = 8
            self.max_geracoes = 500
        elif num_cidades <= 200:
            self.tamanho_populacao = 400
            self.taxa_mutacao = 0.20
            self.taxa_crossover = 0.92
            self.elite_size = 12
            self.max_geracoes = 600
        else:
            self.tamanho_populacao = 300
            self.taxa_mutacao = 0.20
            self.taxa_crossover = 0.92
            self.elite_size = 12
            self.max_geracoes = 600

        self.prob_2opt = 0.10
        self.max_iter_2opt = 25

        self.prob_swap_mut = 0.35
        self.prob_resample_family = 0.20

        self.bestofk_init = 5
        self.fracao_inicial_boas = 0.20

        self.limite_inicializacao_boa = 100
        self.fracao_inicial_boas_grande = 0.0

        self.repeticoes_reparo_max = 3

        self.early_stop_window = 200
        self.early_stop_std_eps = 1e-3
        self.early_stop_min_gen = 400

        self.seed = seed