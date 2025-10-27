from typing import Optional


# ======================================================
# ---------------- Configuração (rápida) ----------------
# ======================================================
class GeneticAlgorithmConfig:
    """
    Define as configurações do Algoritmo Genético.
    Ajustada para velocidade mantendo boa qualidade.
    """
    def __init__(self, num_cidades: int, seed: Optional[int] = None):
        # parâmetros clássicos (ligeiramente mais enxutos)
        if num_cidades <= 100:
            self.tamanho_populacao = 140
            self.taxa_mutacao = 0.15       # base
            self.taxa_crossover = 0.90
            self.elite_size = 8
            self.max_geracoes = 200
        else:
            self.tamanho_populacao = 300
            self.taxa_mutacao = 0.20
            self.taxa_crossover = 0.92
            self.elite_size = 12
            self.max_geracoes = 500

        # Busca local (bem mais leve)
        self.prob_2opt = 0.05             # antes era 0.15
        self.max_iter_2opt = 10           # antes 50

        # Mutação extra
        self.prob_swap_mut = 0.35
        self.prob_resample_family = 0.20

        # Inicialização “boa”
        self.bestofk_init = 5
        self.fracao_inicial_boas = 0.20

        # Para instâncias grandes, desligar a seed “boa”
        self.desliga_gulosa_acima = 200
        self.fracao_inicial_boas_grande = 0.0

        # Política de estagnação leve (opcional)
        self.paciencia = 120
        self.imigrantes = 0.12
        self.mutation_boost = 1.6
        self.repeticoes_reparo_max = 3

        # Parada antecipada (convergência)
        self.early_stop_window = 100
        self.early_stop_std_eps = 1e-3
        self.early_stop_min_gen = 200

        self.seed = seed