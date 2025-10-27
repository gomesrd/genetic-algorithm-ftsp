from leitor_tsp import TSPReader

class TSPInstance:
    def __init__(self, arquivo: str):
        self.coordenadas, self.L, self.V, self.n, self.v, self.best_bound = TSPReader.ler_tsp(arquivo)
        self.m = len(self.coordenadas) - 1
        self.familias, self.familia_no = self._construir_familias()

    def _construir_familias(self):
        familias, familia_no = [], {}
        inicio_familia = 1
        for l in range(self.L):
            aux = list(range(inicio_familia, inicio_familia + self.n[l]))
            for no in aux:
                familia_no[no] = l + 1
            inicio_familia = aux[-1] + 1
            familias.append(aux)
        return familias, familia_no
