class GeneticAlgorithmConfig:
    """Define as configurações do Algoritmo Genético."""
    def __init__(self, num_cidades: int):
        if num_cidades <= 100:
            self.tamanho_populacao = 100
            self.taxa_mutacao = 0.10
            self.taxa_crossover = 0.8
            self.elite_size = 5
            self.max_geracoes = 2000
        else:
            self.tamanho_populacao = 500
            self.taxa_mutacao = 0.25
            self.taxa_crossover = 0.85
            self.elite_size = 10
            self.max_geracoes = 2000

