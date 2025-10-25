class TSPReader:
    @staticmethod
    def ler_tsp(arquivo: str):
        coordenadas, aux, v, n = [], [], [], []
        ler_coordenadas = ler_familias = False

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
                ler_familias, ler_coordenadas = True, False
                continue
            if linha == 'EOF':
                ler_familias = False
                continue

            if ler_coordenadas and linha:
                partes = linha.split()
                coordenadas.append((float(partes[1]), float(partes[2])))
            elif ler_familias and linha:
                partes = linha.split()
                aux.append((int(partes[0]), int(partes[1])))

        L, V = aux[0]
        for elemento in aux[1:]:
            n.append(elemento[0])
            v.append(elemento[1])

        return coordenadas, L, V, n, v
