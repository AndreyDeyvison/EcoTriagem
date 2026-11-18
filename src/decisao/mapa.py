"""Mapa simulado de Maceió usado pela busca do ponto de coleta.

Os dados estão em dados/mapa/maceio_simulado.json: bairros (nós) com
coordenadas aproximadas, ruas (arestas) e pontos de coleta FICTÍCIOS.
"""

import json
import math
from dataclasses import dataclass
from functools import cache

from src.configuracao import RAIZ

ARQUIVO_MAPA = RAIZ / "dados" / "mapa" / "maceio_simulado.json"
RAIO_TERRA_KM = 6371.0


def distancia_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distância em linha reta (fórmula de haversine) entre (lat, lon) em graus."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * RAIO_TERRA_KM * math.asin(math.sqrt(h))


@dataclass(frozen=True)
class PontoColeta:
    nome: str
    bairro: str
    aceita: frozenset[str]


@dataclass(frozen=True)
class Mapa:
    coordenadas: dict[str, tuple[float, float]]
    vizinhos: dict[str, list[tuple[str, float]]]  # bairro -> [(vizinho, custo em km)]
    pontos_coleta: list[PontoColeta]

    @property
    def bairros(self) -> list[str]:
        return sorted(self.coordenadas)

    def pontos_que_aceitam(self, materiais) -> list[PontoColeta]:
        return [p for p in self.pontos_coleta if set(materiais) <= p.aceita]


@cache
def carregar_mapa() -> Mapa:
    dados = json.loads(ARQUIVO_MAPA.read_text(encoding="utf-8"))
    coordenadas = {nome: tuple(coord) for nome, coord in dados["bairros"].items()}

    vizinhos: dict[str, list[tuple[str, float]]] = {nome: [] for nome in coordenadas}
    for a, b, fator in dados["ruas"]:
        if a not in coordenadas or b not in coordenadas:
            raise ValueError(f"rua com bairro desconhecido: {a} - {b}")
        if fator < 1:
            raise ValueError(f"fator de tortuosidade < 1 em {a} - {b}: a heurística deixaria de ser admissível")
        custo = distancia_km(coordenadas[a], coordenadas[b]) * fator
        vizinhos[a].append((b, custo))
        vizinhos[b].append((a, custo))

    pontos = [PontoColeta(p["nome"], p["bairro"], frozenset(p["aceita"])) for p in dados["pontos_coleta"]]
    for p in pontos:
        if p.bairro not in coordenadas:
            raise ValueError(f"ponto de coleta em bairro desconhecido: {p.nome} ({p.bairro})")
    return Mapa(coordenadas=coordenadas, vizinhos=vizinhos, pontos_coleta=pontos)


# Deslocamento (pontos) e alinhamento dos rótulos que se sobreporiam com a posição padrão.
_POSICAO_ROTULO = {
    "Jacintinho": (0, -14, "center"),
    "Mangabeiras": (5, 6, "left"),
    "Ponta Verde": (5, -11, "left"),
}


def desenhar_mapa(rota=None, titulo: str | None = None):
    """Figura do mapa simulado, com a rota destacada se houver.

    rota: RotaColeta (ou None). Devolve uma matplotlib.figure.Figure.
    """
    from matplotlib.figure import Figure

    cor_rua, cor_bairro, cor_texto = "#c9c8c2", "#52514e", "#0b0b0b"
    cor_rota, cor_ponto = "#2a78d6", "#1baf7a"

    mapa = carregar_mapa()
    fig = Figure(figsize=(6.4, 6.4), dpi=110)
    ax = fig.subplots()

    def xy(bairro):
        lat, lon = mapa.coordenadas[bairro]
        return lon, lat

    for a, viz in mapa.vizinhos.items():
        for b, _ in viz:
            if a < b:
                (x1, y1), (x2, y2) = xy(a), xy(b)
                ax.plot([x1, x2], [y1, y2], color=cor_rua, linewidth=1, zorder=1)

    if rota is not None and len(rota.caminho) > 1:
        xs, ys = zip(*(xy(b) for b in rota.caminho))
        ax.plot(xs, ys, color=cor_rota, linewidth=3, solid_capstyle="round", zorder=2)

    destaque = set(rota.caminho) if rota is not None else set()
    bairros_com_ponto = {p.bairro for p in mapa.pontos_coleta}
    for bairro in mapa.coordenadas:
        x, y = xy(bairro)
        if bairro in bairros_com_ponto:
            ax.scatter(x, y, s=70, marker="s", color=cor_ponto, edgecolor="white", linewidth=1.5, zorder=3)
        else:
            ax.scatter(x, y, s=18, color=cor_rota if bairro in destaque else cor_bairro, zorder=3)
        dx, dy, alinhamento = _POSICAO_ROTULO.get(bairro, (5, 4, "left"))
        ax.annotate(
            bairro,
            (x, y),
            xytext=(dx, dy),
            ha=alinhamento,
            textcoords="offset points",
            fontsize=7,
            color=cor_texto if bairro in destaque else cor_bairro,
            fontweight="bold" if bairro in destaque else "normal",
        )

    # Legenda: o que é o quadrado e a linha azul.
    ax.scatter([], [], s=70, marker="s", color=cor_ponto, label="Ponto de coleta (fictício)")
    if rota is not None:
        ax.plot([], [], color=cor_rota, linewidth=3, label=f"Rota: {rota.distancia_km:.1f} km")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, frameon=False)

    ax.set_title(titulo or "Mapa simulado de Maceió", loc="left", fontsize=10, color=cor_texto)
    ax.set_aspect(1 / math.cos(math.radians(-9.63)))  # corrige a escala lon/lat nessa latitude
    ax.set_axis_off()
    fig.tight_layout()
    return fig
