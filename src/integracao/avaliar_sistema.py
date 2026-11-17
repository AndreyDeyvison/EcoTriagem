"""Avaliação de ponta a ponta do EcoTriagem (Visão + PLN + Decisão juntos).

Cada caso combina uma imagem do conjunto de TESTE do TrashNet (material
conhecido) com uma pergunta de dados/avaliacao/perguntas_teste.csv (intenção
conhecida). O caso está correto quando o sistema faz o que deveria:
  - pergunta sobre descarte -> responde com a lixeira da cor certa para o
    material verdadeiro e entende a intenção certa;
  - pergunta fora do escopo -> pede para reformular.

Para cada erro, registra em qual etapa ele nasceu (PLN, visão ou decisão).
Também mede o tempo de cada etapa.

Saídas:
  resultados/metricas/avaliacao_sistema.json
  resultados/graficos/avaliacao_desfechos.png
  resultados/graficos/avaliacao_por_classe.png
  resultados/exemplos/avaliacao/  (exemplos de cada desfecho)

Executar a partir da raiz do repositório:
    python -m src.integracao.avaliar_sistema
"""

import csv
import json
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.configuracao import RAIZ  # noqa: E402
from src.decisao.regras import LIXEIRAS, NOMES, decidir  # noqa: E402
from src.integracao.contratos import ResultadoPipeline  # noqa: E402
from src.integracao.pipeline import analisar_imagem, carregar_modelos  # noqa: E402
from src.integracao.registro import salvar_exemplo  # noqa: E402
from src.pln.intencoes import interpretar_pergunta  # noqa: E402

ARQUIVO_DIVISAO = RAIZ / "dados" / "processados" / "divisao_trashnet.csv"
ARQUIVO_PERGUNTAS = RAIZ / "dados" / "avaliacao" / "perguntas_teste.csv"
SAIDA_METRICAS = RAIZ / "resultados" / "metricas" / "avaliacao_sistema.json"
PASTA_GRAFICOS = RAIZ / "resultados" / "graficos"
PASTA_EXEMPLOS = RAIZ / "resultados" / "exemplos" / "avaliacao"
EXEMPLOS_POR_DESFECHO = 3

DESFECHOS = {
    "correto": "Correto",
    "nova_foto": "Pediu nova foto",
    "erro_pln": "Erro do PLN",
    "erro_visao": "Erro da visão",
    "erro_decisao": "Erro da decisão",
}

# Cores dos gráficos (paleta de referência, modo claro; figura estática para o artigo).
COR_BARRA = "#2a78d6"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e4e3df"


def classificar_desfecho(classe_real: str, intencao_esperada: str, resultado: ResultadoPipeline) -> str:
    """Diz se o caso acertou e, se errou, em qual etapa o erro nasceu.

    A ordem de verificação segue o fluxo da decisão: primeiro o PLN (se a
    pergunta foi mal entendida, o resto não importa), depois o pedido de nova
    foto, depois o material identificado pela visão.
    """
    decisao = resultado.decisao
    if intencao_esperada == "fora_do_escopo":
        return "correto" if decisao.acao == "reformular_pergunta" else "erro_pln"
    if resultado.pln.intencao != intencao_esperada or decisao.acao == "reformular_pergunta":
        return "erro_pln"
    if decisao.acao == "pedir_nova_foto":
        return "nova_foto"

    cor_esperada = LIXEIRAS[classe_real]
    # Decisao.lixeira tem o formato "<cor> (<material>)", ex.: "azul (papel)".
    cor_respondida = decisao.lixeira.split()[0] if decisao.lixeira else None
    if cor_respondida == cor_esperada:
        return "correto"
    if LIXEIRAS[resultado.visao.classe] != cor_esperada:
        return "erro_visao"
    return "erro_decisao"


def carregar_imagens_teste() -> list[tuple[Path, str]]:
    with ARQUIVO_DIVISAO.open(encoding="utf-8") as f:
        return [(RAIZ / l["caminho"], l["classe"]) for l in csv.DictReader(f) if l["conjunto"] == "teste"]


def carregar_perguntas() -> list[tuple[str, str]]:
    with ARQUIVO_PERGUNTAS.open(encoding="utf-8") as f:
        return [(l["texto"], l["intencao"]) for l in csv.DictReader(f)]


def _resumo_tempos(tempos_ms: list[float]) -> dict[str, float]:
    p95 = statistics.quantiles(tempos_ms, n=20)[-1] if len(tempos_ms) > 1 else tempos_ms[0]
    return {"media_ms": round(statistics.fmean(tempos_ms), 3), "p95_ms": round(p95, 3)}


def _porcentagem(parte: int, total: int) -> float:
    return round(100 * parte / total, 2) if total else 0.0


def avaliar() -> tuple[dict, dict[str, list]]:
    """Roda todos os casos. Devolve (métricas, exemplos por desfecho)."""
    imagens = carregar_imagens_teste()
    perguntas = carregar_perguntas()
    if not imagens or not imagens[0][0].exists():
        raise SystemExit("Imagens de teste não encontradas. Rode antes: python -m src.machine_learning.baixar_dados")

    # Carregar os modelos antes de medir tempo, para não contar o carregamento.
    status_modelos = carregar_modelos()

    # PLN: uma vez por pergunta (o resultado não depende da imagem).
    tempos_pln, resultados_pln = [], []
    for texto, _ in perguntas:
        inicio = perf_counter()
        resultados_pln.append(interpretar_pergunta(texto))
        tempos_pln.append((perf_counter() - inicio) * 1000)

    desfechos = Counter()
    por_classe = defaultdict(Counter)
    por_intencao = defaultdict(Counter)
    por_regra = defaultdict(Counter)
    tempos_visao, tempos_decisao = [], []
    acertos_visao = 0
    exemplos: dict[str, list] = defaultdict(list)

    for caminho, classe_real in imagens:
        with Image.open(caminho) as arquivo:
            imagem = arquivo.copy()

        # Visão: uma vez por imagem, como na interface.
        inicio = perf_counter()
        visao = analisar_imagem(imagem)
        tempos_visao.append((perf_counter() - inicio) * 1000)
        acertos_visao += visao.classe == classe_real

        for (texto, intencao_esperada), resultado_pln in zip(perguntas, resultados_pln):
            inicio = perf_counter()
            decisao = decidir(visao, resultado_pln)
            tempos_decisao.append((perf_counter() - inicio) * 1000)

            resultado = ResultadoPipeline(visao=visao, pln=resultado_pln, decisao=decisao)
            desfecho = classificar_desfecho(classe_real, intencao_esperada, resultado)
            desfechos[desfecho] += 1
            por_classe[classe_real][desfecho] += 1
            por_intencao[intencao_esperada][desfecho] += 1
            por_regra[decisao.regra][desfecho] += 1

            # Guarda poucos exemplos de cada desfecho, de imagens diferentes.
            guardados = exemplos[desfecho]
            if len(guardados) < EXEMPLOS_POR_DESFECHO and all(g[0] != caminho for g in guardados):
                guardados.append((caminho, imagem, resultado, classe_real, intencao_esperada))

    total = sum(desfechos.values())
    acertos_pln = sum(r.intencao == i for r, (_, i) in zip(resultados_pln, perguntas))
    metricas = {
        "modelos": {nome: ("treinado" if real else "stub") for nome, real in status_modelos.items()},
        "n_imagens": len(imagens),
        "n_perguntas": len(perguntas),
        "n_casos": total,
        "taxa_acerto_sistema_pct": _porcentagem(desfechos["correto"], total),
        "desfechos": {
            chave: {"n": desfechos[chave], "pct": _porcentagem(desfechos[chave], total)} for chave in DESFECHOS
        },
        "acuracia_visao_classe_pct": _porcentagem(acertos_visao, len(imagens)),
        "acuracia_pln_intencao_pct": _porcentagem(acertos_pln, len(perguntas)),
        "acerto_por_classe_pct": {
            c: _porcentagem(cont["correto"], sum(cont.values())) for c, cont in sorted(por_classe.items())
        },
        "acerto_por_intencao_pct": {
            i: _porcentagem(cont["correto"], sum(cont.values())) for i, cont in sorted(por_intencao.items())
        },
        # Quantas vezes cada regra da decisão foi aplicada e quanto ela acertou.
        "regras": {
            r: {
                "n": sum(cont.values()),
                "pct": _porcentagem(sum(cont.values()), total),
                "acerto_pct": _porcentagem(cont["correto"], sum(cont.values())),
            }
            for r, cont in sorted(por_regra.items())
        },
        "tempo": {
            "visao_por_imagem": _resumo_tempos(tempos_visao),
            "pln_por_pergunta": _resumo_tempos(tempos_pln),
            "decisao_por_caso": _resumo_tempos(tempos_decisao),
        },
    }
    return metricas, exemplos


def _grafico_barras(rotulos: list[str], valores: list[float], titulo: str, arquivo: Path) -> None:
    altura = 0.45 * len(rotulos) + 1.0
    fig, ax = plt.subplots(figsize=(6.0, altura), dpi=200)
    posicoes = range(len(rotulos))
    ax.barh(posicoes, valores, height=0.55, color=COR_BARRA)
    ax.set_yticks(list(posicoes), rotulos, color=COR_TEXTO)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% dos casos", color=COR_TEXTO_SECUNDARIO)
    ax.set_title(titulo, loc="left", color=COR_TEXTO, fontsize=11)
    for posicao, valor in zip(posicoes, valores):
        ax.text(valor + 1.5, posicao, f"{valor:.1f}%", va="center", color=COR_TEXTO, fontsize=9)
    ax.grid(axis="x", color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COR_TEXTO_SECUNDARIO)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO, length=0)
    fig.tight_layout()
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(arquivo, facecolor="white")
    plt.close(fig)


def salvar_graficos(metricas: dict) -> list[Path]:
    arquivos = [PASTA_GRAFICOS / "avaliacao_desfechos.png", PASTA_GRAFICOS / "avaliacao_por_classe.png"]
    _grafico_barras(
        list(DESFECHOS.values()),
        [metricas["desfechos"][c]["pct"] for c in DESFECHOS],
        "Desfecho dos casos na avaliação de ponta a ponta",
        arquivos[0],
    )
    por_classe = metricas["acerto_por_classe_pct"]
    _grafico_barras(
        [NOMES[c].capitalize() for c in por_classe],
        list(por_classe.values()),
        "Acerto do sistema por material verdadeiro",
        arquivos[1],
    )
    return arquivos


def salvar_exemplos(exemplos: dict[str, list]) -> None:
    """Recria resultados/exemplos/avaliacao/ com os exemplos desta execução."""
    if PASTA_EXEMPLOS.exists():
        shutil.rmtree(PASTA_EXEMPLOS)
    for desfecho in DESFECHOS:
        for caminho, imagem, resultado, classe_real, intencao_esperada in exemplos.get(desfecho, []):
            salvar_exemplo(
                PASTA_EXEMPLOS,
                imagem,
                resultado,
                extras={
                    "desfecho": desfecho,
                    "classe_real": classe_real,
                    "intencao_esperada": intencao_esperada,
                    "imagem_origem": caminho.relative_to(RAIZ).as_posix(),
                },
            )


def imprimir_resumo(metricas: dict) -> None:
    stubs = [nome for nome, estado in metricas["modelos"].items() if estado == "stub"]
    if stubs:
        print(f"ATENÇÃO: {" e ".join(stubs)} usando stub; os números abaixo NÃO são resultados do sistema final.\n")
    print(
        f"{metricas['n_casos']} casos = {metricas['n_imagens']} imagens de teste x {metricas['n_perguntas']} perguntas\n"
    )
    print(f"Acerto do sistema completo: {metricas['taxa_acerto_sistema_pct']:.1f}%")
    for chave, rotulo in DESFECHOS.items():
        d = metricas["desfechos"][chave]
        print(f"  {rotulo:<16} {d['n']:>6}  ({d['pct']:.1f}%)")
    print(f"\nAcurácia isolada da visão (classe exata): {metricas['acuracia_visao_classe_pct']:.1f}%")
    print(f"Acurácia isolada do PLN (intenção):       {metricas['acuracia_pln_intencao_pct']:.1f}%")
    print("\nAcerto por material verdadeiro:")
    for classe, pct in metricas["acerto_por_classe_pct"].items():
        print(f"  {NOMES[classe]:<26} {pct:.1f}%")
    print("\nAcerto por intenção esperada:")
    for intencao, pct in metricas["acerto_por_intencao_pct"].items():
        print(f"  {intencao:<16} {pct:.1f}%")
    print("\nRegras da decisão (aplicações / acerto quando aplicada):")
    for regra, r in metricas["regras"].items():
        print(f"  {regra}  {r['n']:>6} ({r['pct']:.1f}%)  acerto {r['acerto_pct']:.1f}%")
    print("\nTempo por etapa (média / p95):")
    for etapa, t in metricas["tempo"].items():
        print(f"  {etapa:<18} {t['media_ms']:.2f} ms / {t['p95_ms']:.2f} ms")


def main() -> None:
    metricas, exemplos = avaliar()
    SAIDA_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    SAIDA_METRICAS.write_text(json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    graficos = salvar_graficos(metricas)
    salvar_exemplos(exemplos)

    imprimir_resumo(metricas)
    print("\nArquivos gerados:")
    for arquivo in [SAIDA_METRICAS, *graficos, PASTA_EXEMPLOS]:
        print(f"  {arquivo.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
