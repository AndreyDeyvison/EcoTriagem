"""Compara os algoritmos de busca do ponto de coleta (requisito 4.1).

Para cada bairro de origem do mapa e cada material reciclável, roda BFS,
custo uniforme, busca gulosa e A* e mede:
  - custo do caminho (km) e se é o ótimo (igual ao do custo uniforme);
  - nós expandidos (esforço da busca);
  - tempo de execução.

Saídas:
  resultados/metricas/comparacao_buscas.json
  resultados/graficos/comparacao_buscas_nos.png
  resultados/graficos/comparacao_buscas_otimo.png
  resultados/graficos/mapa_rota_exemplo.png

Executar a partir da raiz do repositório:
    python -m src.decisao.comparar_buscas
"""

import json
import statistics
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.configuracao import RAIZ  # noqa: E402
from src.decisao.busca import ALGORITMOS, NOMES_ALGORITMOS, buscar_ponto_coleta  # noqa: E402
from src.decisao.mapa import carregar_mapa, desenhar_mapa  # noqa: E402
from src.decisao.regras import NOMES, RECICLAVEL  # noqa: E402

SAIDA_METRICAS = RAIZ / "resultados" / "metricas" / "comparacao_buscas.json"
PASTA_GRAFICOS = RAIZ / "resultados" / "graficos"
REPETICOES_TEMPO = 50  # cada busca é rápida demais para medir uma vez só
EXEMPLO = ("Trapiche da Barra", "vidro")

COR_BARRA = "#2a78d6"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e4e3df"


def comparar() -> dict:
    mapa = carregar_mapa()
    materiais = [m for m, reciclavel in RECICLAVEL.items() if reciclavel]
    resultados = {alg: [] for alg in ALGORITMOS}

    for origem in mapa.bairros:
        for material in materiais:
            objetivos = {p.bairro for p in mapa.pontos_que_aceitam([material])}
            otimo = ALGORITMOS["custo_uniforme"](mapa, origem, objetivos).custo_km
            for alg, funcao in ALGORITMOS.items():
                inicio = perf_counter()
                for _ in range(REPETICOES_TEMPO):
                    r = funcao(mapa, origem, objetivos)
                tempo_us = (perf_counter() - inicio) / REPETICOES_TEMPO * 1e6
                resultados[alg].append(
                    {
                        "custo_km": r.custo_km,
                        "otimo": r.custo_km <= otimo + 1e-9,
                        "excesso_pct": 100 * (r.custo_km - otimo) / otimo if otimo > 0 else 0.0,
                        "nos_expandidos": r.nos_expandidos,
                        "tempo_us": tempo_us,
                    }
                )

    resumo = {}
    for alg, linhas in resultados.items():
        resumo[alg] = {
            "nome": NOMES_ALGORITMOS[alg],
            "rotas_otimas_pct": round(100 * sum(l["otimo"] for l in linhas) / len(linhas), 2),
            "excesso_medio_pct": round(statistics.fmean(l["excesso_pct"] for l in linhas), 2),
            "excesso_max_pct": round(max(l["excesso_pct"] for l in linhas), 2),
            "custo_medio_km": round(statistics.fmean(l["custo_km"] for l in linhas), 3),
            "nos_expandidos_medio": round(statistics.fmean(l["nos_expandidos"] for l in linhas), 2),
            "tempo_medio_us": round(statistics.fmean(l["tempo_us"] for l in linhas), 2),
        }
    return {
        "mapa": "dados/mapa/maceio_simulado.json (simulado)",
        "n_bairros": len(mapa.bairros),
        "n_ruas": sum(len(v) for v in mapa.vizinhos.values()) // 2,
        "n_pontos_coleta": len(mapa.pontos_coleta),
        "materiais": materiais,
        "n_consultas": len(mapa.bairros) * len(materiais),
        "algoritmos": resumo,
    }


def _barras(rotulos, valores, formato, titulo, eixo_x, arquivo, limite=None):
    fig, ax = plt.subplots(figsize=(6.0, 0.5 * len(rotulos) + 1.0), dpi=200)
    posicoes = range(len(rotulos))
    ax.barh(posicoes, valores, height=0.55, color=COR_BARRA)
    ax.set_yticks(list(posicoes), rotulos, color=COR_TEXTO)
    ax.invert_yaxis()
    maximo = limite or max(valores) * 1.2
    ax.set_xlim(0, maximo)
    for posicao, valor in zip(posicoes, valores):
        ax.text(valor + maximo * 0.015, posicao, formato.format(valor), va="center", color=COR_TEXTO, fontsize=9)
    ax.set_xlabel(eixo_x, color=COR_TEXTO_SECUNDARIO)
    ax.set_title(titulo, loc="left", color=COR_TEXTO, fontsize=11)
    ax.grid(axis="x", color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COR_TEXTO_SECUNDARIO)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO, length=0)
    fig.tight_layout()
    fig.savefig(arquivo, facecolor="white")
    plt.close(fig)


def salvar_graficos(metricas: dict) -> list:
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    algs = metricas["algoritmos"]
    rotulos = [a["nome"] for a in algs.values()]
    arquivos = [
        PASTA_GRAFICOS / "comparacao_buscas_nos.png",
        PASTA_GRAFICOS / "comparacao_buscas_otimo.png",
        PASTA_GRAFICOS / "mapa_rota_exemplo.png",
    ]
    _barras(
        rotulos,
        [a["nos_expandidos_medio"] for a in algs.values()],
        "{:.1f}",
        "Nós expandidos por busca (média)",
        "nós expandidos",
        arquivos[0],
    )
    _barras(
        rotulos,
        [a["rotas_otimas_pct"] for a in algs.values()],
        "{:.1f}%",
        "Rotas de menor distância encontradas",
        "% das consultas",
        arquivos[1],
        limite=100,
    )
    origem, material = EXEMPLO
    rota = buscar_ponto_coleta(origem, [material])
    fig = desenhar_mapa(rota, f"A*: {origem} → {rota.ponto} ({NOMES[material]})")
    fig.savefig(arquivos[2], facecolor="white", dpi=200)
    return arquivos


def main() -> None:
    metricas = comparar()
    SAIDA_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_METRICAS.write_text(json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    graficos = salvar_graficos(metricas)

    print(
        f"{metricas['n_consultas']} consultas = {metricas['n_bairros']} bairros de origem x "
        f"{len(metricas['materiais'])} materiais ({metricas['n_ruas']} ruas, {metricas['n_pontos_coleta']} pontos)\n"
    )
    print(f"{'algoritmo':<24} {'ótimas':>8} {'excesso méd.':>13} {'excesso máx.':>13} {'nós exp.':>9} {'tempo':>10}")
    for a in metricas["algoritmos"].values():
        print(
            f"{a['nome']:<24} {a['rotas_otimas_pct']:>7.1f}% {a['excesso_medio_pct']:>12.1f}% "
            f"{a['excesso_max_pct']:>12.1f}% {a['nos_expandidos_medio']:>9.1f} {a['tempo_medio_us']:>8.1f} µs"
        )
    print("\nArquivos gerados:")
    for arquivo in [SAIDA_METRICAS, *graficos]:
        print(f"  {arquivo.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
