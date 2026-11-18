"""Busca do ponto de coleta mais próximo no mapa simulado de Maceió.

Problema de busca:
  - estados: bairros do mapa;
  - estado inicial: bairro do usuário;
  - ações: ir para um bairro vizinho, com custo = distância da rua em km;
  - objetivo: qualquer bairro com ponto de coleta que aceite o(s) material(is).

Algoritmos (mesma interface, para comparação em comparar_buscas.py):
  - largura (BFS): menor número de ruas; ignora a distância;
  - custo_uniforme (Dijkstra): menor distância, sem heurística (referência de ótimo);
  - gulosa: expande o bairro que parece mais perto de um ponto (só h);
  - a_estrela (A*): f = g + h; ótimo com heurística admissível.

Heurística h(n) = menor distância em linha reta de n até algum bairro objetivo.
É admissível e consistente: o custo de cada rua é a distância em linha reta
multiplicada por um fator >= 1, e a distância em linha reta obedece à
desigualdade triangular.

O teste de objetivo é feito ao EXPANDIR o nó, em todos os algoritmos, para
que a contagem de nós expandidos seja comparável.
"""

import heapq
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import count

from src.decisao.mapa import Mapa, carregar_mapa, distancia_km
from src.integracao.contratos import RotaColeta


@dataclass(frozen=True)
class ResultadoBusca:
    caminho: list[str]
    custo_km: float
    nos_expandidos: int


def heuristica(mapa: Mapa, objetivos: set[str]) -> Callable[[str], float]:
    coords_objetivos = [mapa.coordenadas[o] for o in objetivos]

    def h(bairro: str) -> float:
        return min(distancia_km(mapa.coordenadas[bairro], c) for c in coords_objetivos)

    return h


def _custo(mapa: Mapa, caminho: list[str]) -> float:
    total = 0.0
    for a, b in zip(caminho, caminho[1:]):
        total += next(c for v, c in mapa.vizinhos[a] if v == b)
    return total


def _reconstruir(pais: dict[str, str | None], fim: str) -> list[str]:
    caminho = [fim]
    while pais[caminho[-1]] is not None:
        caminho.append(pais[caminho[-1]])
    return caminho[::-1]


def busca_largura(mapa: Mapa, origem: str, objetivos: set[str]) -> ResultadoBusca | None:
    fronteira = deque([origem])
    pais: dict[str, str | None] = {origem: None}
    expandidos = 0
    while fronteira:
        atual = fronteira.popleft()
        expandidos += 1
        if atual in objetivos:
            caminho = _reconstruir(pais, atual)
            return ResultadoBusca(caminho, _custo(mapa, caminho), expandidos)
        for vizinho, _ in mapa.vizinhos[atual]:
            if vizinho not in pais:
                pais[vizinho] = atual
                fronteira.append(vizinho)
    return None


def _melhor_primeiro(
    mapa: Mapa,
    origem: str,
    objetivos: set[str],
    prioridade: Callable[[float, float], float],
    h: Callable[[str], float],
) -> ResultadoBusca | None:
    """Busca de melhor escolha genérica: prioridade(g, h) define o algoritmo."""
    desempate = count()  # ordem de inserção, para resultados determinísticos
    fronteira = [(prioridade(0.0, h(origem)), next(desempate), origem)]
    g = {origem: 0.0}
    pais: dict[str, str | None] = {origem: None}
    expandidos_set: set[str] = set()
    while fronteira:
        _, _, atual = heapq.heappop(fronteira)
        if atual in expandidos_set:
            continue  # entrada antiga na fila, já expandida por um caminho melhor
        expandidos_set.add(atual)
        if atual in objetivos:
            caminho = _reconstruir(pais, atual)
            return ResultadoBusca(caminho, g[atual], len(expandidos_set))
        for vizinho, custo in mapa.vizinhos[atual]:
            novo_g = g[atual] + custo
            if vizinho not in expandidos_set and novo_g < g.get(vizinho, float("inf")):
                g[vizinho] = novo_g
                pais[vizinho] = atual
                heapq.heappush(fronteira, (prioridade(novo_g, h(vizinho)), next(desempate), vizinho))
    return None


def busca_custo_uniforme(mapa: Mapa, origem: str, objetivos: set[str]) -> ResultadoBusca | None:
    return _melhor_primeiro(mapa, origem, objetivos, lambda g, h: g, lambda _: 0.0)


def busca_gulosa(mapa: Mapa, origem: str, objetivos: set[str]) -> ResultadoBusca | None:
    return _melhor_primeiro(mapa, origem, objetivos, lambda g, h: h, heuristica(mapa, objetivos))


def busca_a_estrela(mapa: Mapa, origem: str, objetivos: set[str]) -> ResultadoBusca | None:
    return _melhor_primeiro(mapa, origem, objetivos, lambda g, h: g + h, heuristica(mapa, objetivos))


ALGORITMOS = {
    "largura": busca_largura,
    "custo_uniforme": busca_custo_uniforme,
    "gulosa": busca_gulosa,
    "a_estrela": busca_a_estrela,
}

NOMES_ALGORITMOS = {
    "largura": "Busca em largura (BFS)",
    "custo_uniforme": "Custo uniforme",
    "gulosa": "Busca gulosa",
    "a_estrela": "A*",
}


def buscar_ponto_coleta(origem: str, materiais: Iterable[str], algoritmo: str = "a_estrela") -> RotaColeta | None:
    """Rota do bairro de origem até o ponto de coleta mais próximo que aceita os materiais.

    Devolve None se nenhum ponto do mapa aceitar todos os materiais.
    """
    mapa = carregar_mapa()
    if origem not in mapa.coordenadas:
        raise ValueError(f"bairro desconhecido: {origem!r}")
    pontos = mapa.pontos_que_aceitam(materiais)
    if not pontos:
        return None

    resultado = ALGORITMOS[algoritmo](mapa, origem, {p.bairro for p in pontos})
    if resultado is None:
        return None
    destino = resultado.caminho[-1]
    ponto = next(p for p in pontos if p.bairro == destino)
    return RotaColeta(
        ponto=ponto.nome,
        bairro_destino=destino,
        caminho=resultado.caminho,
        distancia_km=round(resultado.custo_km, 2),
        nos_expandidos=resultado.nos_expandidos,
        algoritmo=algoritmo,
    )
