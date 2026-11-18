"""Análise exploratória do TrashNet, sobre a divisão já fixada.

Script executável em vez de notebook: roda sempre igual, gera os arquivos
sozinho e entra no fluxo de reprodutibilidade do repositório (um notebook
exigiria execução manual e guardaria saídas que envelhecem).

O que produz:

resultados/metricas/
    distribuicao_trashnet.csv   contagem e percentual por classe e conjunto
    distribuicao_trashnet.json  o mesmo, com dimensões, modos de cor e desbalanceamento
resultados/graficos/
    distribuicao_classes_trashnet.png

Executar a partir da raiz do repositório:
    python -m src.machine_learning.analise_exploratoria
"""

import csv
import json
from collections import Counter
from datetime import date

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.machine_learning.dados_visao import (
    CLASSES_ORDENADAS,
    CONJUNTOS,
    RAIZ,
    carregar_divisao,
    contagem,
    por_conjunto,
)

PASTA_METRICAS = RAIZ / "resultados" / "metricas"
PASTA_GRAFICOS = RAIZ / "resultados" / "graficos"

ARQUIVO_CSV = PASTA_METRICAS / "distribuicao_trashnet.csv"
ARQUIVO_JSON = PASTA_METRICAS / "distribuicao_trashnet.json"
ARQUIVO_GRAFICO = PASTA_GRAFICOS / "distribuicao_classes_trashnet.png"

COLUNAS = ("classe", "treino", "validacao", "teste", "total", "percentual_do_dataset")


def tabela_distribuicao(registros) -> list[dict]:
    """Uma linha por classe, com a contagem em cada conjunto e o percentual."""
    grupos = por_conjunto(registros)
    contagens = {c: contagem(rs) for c, rs in grupos.items()}
    total_geral = len(registros)

    linhas = []
    for classe in CLASSES_ORDENADAS:
        valores = {c: contagens[c][classe] for c in CONJUNTOS}
        total = sum(valores.values())
        linhas.append(
            {
                "classe": classe,
                **valores,
                "total": total,
                "percentual_do_dataset": round(100 * total / total_geral, 2),
            }
        )
    return linhas


def inspecionar_imagens(registros) -> dict:
    """Dimensões e modos de cor, lidos do cabeçalho (sem decodificar o pixel)."""
    dimensoes: Counter = Counter()
    modos: Counter = Counter()
    larguras, alturas = [], []

    for registro in registros:
        with Image.open(registro.caminho) as imagem:
            largura, altura = imagem.size
            dimensoes[f"{largura}x{altura}"] += 1
            modos[imagem.mode] += 1
        larguras.append(largura)
        alturas.append(altura)

    larguras_arr, alturas_arr = np.array(larguras), np.array(alturas)
    return {
        "dimensoes": dict(dimensoes.most_common()),
        "modos_de_cor": dict(modos.most_common()),
        "largura": {
            "minima": int(larguras_arr.min()),
            "maxima": int(larguras_arr.max()),
            "media": round(float(larguras_arr.mean()), 1),
        },
        "altura": {
            "minima": int(alturas_arr.min()),
            "maxima": int(alturas_arr.max()),
            "media": round(float(alturas_arr.mean()), 1),
        },
    }


def _grafico(linhas: list[dict]) -> None:
    classes = [linha["classe"] for linha in linhas]
    posicoes = np.arange(len(classes))
    largura = 0.26

    figura, eixo = plt.subplots(figsize=(9, 5))
    for deslocamento, conjunto in zip((-largura, 0.0, largura), CONJUNTOS):
        valores = [linha[conjunto] for linha in linhas]
        barras = eixo.bar(posicoes + deslocamento, valores, largura, label=conjunto)
        eixo.bar_label(barras, fontsize=8, padding=1)

    eixo.set_xticks(posicoes, classes)
    eixo.set_xlabel("classe de resíduo")
    eixo.set_ylabel("número de imagens")
    eixo.set_title("TrashNet — distribuição das classes por conjunto (divisão fixa)")
    eixo.legend(title="conjunto")
    eixo.grid(axis="y", alpha=0.3)
    eixo.set_axisbelow(True)
    figura.tight_layout()
    figura.savefig(ARQUIVO_GRAFICO, dpi=160)
    plt.close(figura)


def main() -> None:
    PASTA_METRICAS.mkdir(parents=True, exist_ok=True)
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

    registros = carregar_divisao()
    linhas = tabela_distribuicao(registros)

    print(f"{'classe':<9} {'treino':>7} {'valid.':>7} {'teste':>7} {'total':>7} {'% dataset':>10}")
    for linha in linhas:
        print(
            f"{linha['classe']:<9} {linha['treino']:>7} {linha['validacao']:>7} "
            f"{linha['teste']:>7} {linha['total']:>7} {linha['percentual_do_dataset']:>9.2f}%"
        )
    grupos = por_conjunto(registros)
    print(
        f"{'TOTAL':<9} {len(grupos['treino']):>7} {len(grupos['validacao']):>7} "
        f"{len(grupos['teste']):>7} {len(registros):>7} {100.0:>9.2f}%"
    )

    maior = max(linhas, key=lambda linha: linha["total"])
    menor = min(linhas, key=lambda linha: linha["total"])
    razao = maior["total"] / menor["total"]
    print(
        f"\nDesbalanceamento: {menor['classe']} tem {menor['total']} imagens contra "
        f"{maior['total']} de {maior['classe']} — razão de {razao:.1f}x."
    )
    print(
        f"A classe {menor['classe']} é {menor['percentual_do_dataset']:.1f}% do dataset e fica com "
        f"apenas {menor['teste']} imagens no teste, o que torna as métricas dela as mais instáveis."
    )

    print("\nLendo dimensões e modos de cor...")
    imagens = inspecionar_imagens(registros)
    for dimensao, quantidade in list(imagens["dimensoes"].items())[:5]:
        print(f"  {dimensao:<12} {quantidade:>5} imagens")
    print(f"  modos de cor: {imagens['modos_de_cor']}")

    with ARQUIVO_CSV.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS)
        escritor.writeheader()
        escritor.writerows(linhas)

    ARQUIVO_JSON.write_text(
        json.dumps(
            {
                "gerado_em": date.today().isoformat(),
                "fonte": "dados/processados/divisao_trashnet.csv",
                "total_de_imagens": len(registros),
                "imagens_por_conjunto": {c: len(rs) for c, rs in grupos.items()},
                "distribuicao": linhas,
                "desbalanceamento": {
                    "classe_minoritaria": menor["classe"],
                    "imagens_da_minoritaria": menor["total"],
                    "classe_majoritaria": maior["classe"],
                    "imagens_da_majoritaria": maior["total"],
                    "razao": round(razao, 2),
                },
                "imagens": imagens,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _grafico(linhas)
    print(f"\nTabela salva em {ARQUIVO_CSV.relative_to(RAIZ)} e {ARQUIVO_JSON.relative_to(RAIZ)}")
    print(f"Gráfico salvo em {ARQUIVO_GRAFICO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
