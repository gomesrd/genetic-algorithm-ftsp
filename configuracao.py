class GeneticAlgorithmConfig:
    """Define as configurações do Algoritmo Genético."""
    def __init__(self, num_cidades: int):
        if num_cidades <= 100:
            self.tamanho_populacao = 100
            self.taxa_mutacao = 0.15
            self.taxa_crossover = 0.8
            self.elite_size = 5
            self.max_geracoes = 2000
        else:
            self.tamanho_populacao = 400
            self.taxa_mutacao = 0.15
            self.taxa_crossover = 0.8
            self.elite_size = 20
            self.max_geracoes = 2000
