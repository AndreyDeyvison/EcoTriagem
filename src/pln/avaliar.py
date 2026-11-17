"""Avalia o classificador de intenções no conjunto de teste.

Roda uma única vez, depois de `treinar.py` ter escolhido o modelo: o conjunto
de teste não participou da escolha, então o número aqui é a estimativa honesta
de desempenho.

O que produz:

resultados/metricas/
    pln_relatorio_teste.json    acurácia, F1-macro e métricas por classe
    pln_relatorio_teste.csv     a mesma tabela por classe, lado a lado com o baseline
    pln_erros.csv               todas as frases erradas, com a confiança dada
    pln_limiar.csv              varredura do limiar de confiança
resultados/graficos/
    pln_matriz_confusao.png
    pln_limiar_confianca.png
resultados/exemplos/
    pln_entrada_saida.md        exemplos de entrada e saída (frases novas)

Executar a partir da raiz do repositório:
    python -m src.pln.avaliar
"""

import csv
import json
from datetime import date

import matplotlib

matplotlib.use("Agg")  # sem janela: só salva arquivo

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.configuracao import INTENCOES, LIMIAR_CONFIANCA_PLN
from src.pln import baseline, dados, modelos, objetos

PASTA_METRICAS = modelos.RAIZ / "resultados" / "metricas"
PASTA_GRAFICOS = modelos.RAIZ / "resultados" / "graficos"
PASTA_EXEMPLOS = modelos.RAIZ / "resultados" / "exemplos"

LIMIARES = np.round(np.arange(0.0, 1.0, 0.05), 2)

# Frases novas, que não estão no dataset, para a tabela de exemplos do artigo.
EXEMPLOS = (
    "o que é esse objeto da foto?",
    "e ai onde que eu jogo isso?",
    "isso eh recilavel?",
    "qual lixiera eu uso?",
    "preciso lavar antes de jogar fora?",
    "tem algum ecoponto perto da ufal?",
    "e essa pilha velha?",
    "onde jogo esse pote de vidro vazio?",
    "a caixa de leite vai junto com o papelao?",
    "e esse?",
    "qual cor de lixeira?",
    "bom dia tudo certo?",
    "qual o resultado do flamengo ontem?",
    "?????",
)


def _taxa(acertos: int, total: int) -> float:
    return acertos / total if total else 0.0


def varrer_limiar(
    verdadeiros: list[str], previstos: list[str], confiancas: np.ndarray
) -> list[dict]:
    """Mede o efeito do limiar de confiança sobre o comportamento do sistema.

    Abaixo do limiar o sistema não responde: pede uma reformulação. Isso é bom
    quando a pergunta era mesmo fora do escopo e ruim quando era uma pergunta
    legítima. As colunas separam os dois efeitos:

    - `cobertura`: fração de perguntas que o sistema aceita responder;
    - `acuracia_aceitas`: acerto da intenção entre as aceitas;
    - `escopo_atendido`: das perguntas legítimas, quantas foram respondidas
      com a intenção certa;
    - `fora_escopo_recusado`: das perguntas fora do escopo, quantas o sistema
      recusou (previu `fora_do_escopo` ou ficou abaixo do limiar);
    - `acerto_sistema`: as duas situações certas somadas sobre o total — é o
      número que justifica (ou corrige) o limiar adotado.
    """
    verdadeiros_arr = np.array(verdadeiros)
    previstos_arr = np.array(previstos)
    eh_fora = verdadeiros_arr == "fora_do_escopo"
    total = len(verdadeiros)

    linhas = []
    for limiar in LIMIARES:
        aceitas = confiancas >= limiar
        recusado = (previstos_arr == "fora_do_escopo") | ~aceitas
        correto = previstos_arr == verdadeiros_arr

        escopo_ok = int(np.sum(~eh_fora & correto & aceitas))
        fora_ok = int(np.sum(eh_fora & recusado))
        linhas.append(
            {
                "limiar": float(limiar),
                "cobertura": _taxa(int(np.sum(aceitas)), total),
                "acuracia_aceitas": _taxa(int(np.sum(correto & aceitas)), int(np.sum(aceitas))),
                "escopo_atendido": _taxa(escopo_ok, int(np.sum(~eh_fora))),
                "fora_escopo_recusado": _taxa(fora_ok, int(np.sum(eh_fora))),
                "acerto_sistema": _taxa(escopo_ok + fora_ok, total),
            }
        )
    return linhas


def _grafico_matriz_confusao(verdadeiros: list[str], previstos: list[str]) -> None:
    rotulos = list(INTENCOES)
    matriz = confusion_matrix(verdadeiros, previstos, labels=rotulos)
    figura, eixo = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay(matriz, display_labels=rotulos).plot(
        ax=eixo, cmap="Blues", colorbar=False, values_format="d"
    )
    eixo.set_title("PLN — matriz de confusão (conjunto de teste)")
    eixo.set_xlabel("intenção prevista")
    eixo.set_ylabel("intenção correta")
    plt.setp(eixo.get_xticklabels(), rotation=35, ha="right")
    figura.tight_layout()
    figura.savefig(PASTA_GRAFICOS / "pln_matriz_confusao.png", dpi=160)
    plt.close(figura)


def _grafico_limiar(linhas: list[dict]) -> None:
    limiares = [linha["limiar"] for linha in linhas]
    figura, eixo = plt.subplots(figsize=(8, 5))
    series = (
        ("cobertura", "Cobertura (perguntas respondidas)"),
        ("acuracia_aceitas", "Acurácia entre as respondidas"),
        ("acerto_sistema", "Acerto do sistema (responder/recusar certo)"),
    )
    for coluna, rotulo in series:
        eixo.plot(limiares, [linha[coluna] for linha in linhas], marker="o", ms=3, label=rotulo)
    eixo.axvline(
        LIMIAR_CONFIANCA_PLN,
        color="crimson",
        linestyle="--",
        label=f"limiar adotado ({LIMIAR_CONFIANCA_PLN})",
    )
    eixo.set_xlabel("limiar de confiança do PLN")
    eixo.set_ylabel("proporção")
    eixo.set_ylim(0, 1.02)
    eixo.set_title("PLN — efeito do limiar de confiança (conjunto de teste)")
    eixo.grid(alpha=0.3)
    eixo.legend(loc="lower left", fontsize=9)
    figura.tight_layout()
    figura.savefig(PASTA_GRAFICOS / "pln_limiar_confianca.png", dpi=160)
    plt.close(figura)


def _salvar_csv(caminho, colunas: tuple[str, ...], linhas: list[dict]) -> None:
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas)


def _tabela_exemplos(classificador: modelos.ClassificadorIntencoes) -> str:
    dataset, _ = dados.carregar()
    conhecidas = {texto.strip().lower() for texto in dataset}
    linhas = [
        "# PLN — exemplos de entrada e saída",
        "",
        f"Modelo: `{classificador.nome}` — {classificador.descricao}  ",
        f"Limiar de confiança do sistema: {LIMIAR_CONFIANCA_PLN}  ",
        f"Gerado por `python -m src.pln.avaliar` em {date.today().isoformat()}",
        "",
        "Frases escritas depois do dataset, nenhuma usada no treino ou no teste.",
        "A última coluna é o item extra do PLN (`src/pln/objetos.py`): o objeto",
        "citado no texto, que a Visão sozinha não tem como saber.",
        "",
        "| pergunta | intenção | confiança | sistema responde? | objeto citado |",
        "|---|---|---|---|---|",
    ]
    for texto in EXEMPLOS:
        if texto.strip().lower() in conhecidas:
            print(f"  [aviso] exemplo '{texto}' está no dataset; troque por uma frase nova")
        intencao, confianca = classificador.classificar(texto)
        responde = intencao != "fora_do_escopo" and confianca >= LIMIAR_CONFIANCA_PLN
        objeto = objetos.extrair_objeto(texto)
        if objeto is None:
            descricao_objeto = "—"
        else:
            detalhe = objeto.classe_sugerida or ("descarte especial" if objeto.especial else "ambíguo")
            descricao_objeto = f"{objeto.objeto} ({detalhe})"
        linhas.append(
            f"| {texto} | `{intencao}` | {confianca:.2f} | "
            f"{'sim' if responde else 'não — pede reformulação'} | {descricao_objeto} |"
        )
    return "\n".join(linhas) + "\n"


def main() -> None:
    for pasta in (PASTA_METRICAS, PASTA_GRAFICOS, PASTA_EXEMPLOS):
        pasta.mkdir(parents=True, exist_ok=True)

    _, _, teste_x, teste_y, origem_teste = dados.dividir()
    classificador = modelos.carregar()
    print(f"Modelo: {classificador.nome} — {classificador.descricao}")
    print(f"Teste: {len(teste_x)} frases ({origem_teste})\n")

    probabilidades = classificador.prever_proba(teste_x)
    classes = classificador.classes
    indices = probabilidades.argmax(axis=1)
    previstos = [classes[i] for i in indices]
    confiancas = probabilidades.max(axis=1)
    previstos_baseline = baseline.prever(teste_x)

    acuracia = accuracy_score(teste_y, previstos)
    f1_macro = f1_score(teste_y, previstos, average="macro", zero_division=0)
    acuracia_baseline = accuracy_score(teste_y, previstos_baseline)
    f1_baseline = f1_score(teste_y, previstos_baseline, average="macro", zero_division=0)

    print(f"{'':<24}{'acurácia':>10}{'F1-macro':>10}")
    print(f"{'baseline palavras-chave':<24}{acuracia_baseline:>10.3f}{f1_baseline:>10.3f}")
    print(f"{'modelo treinado':<24}{acuracia:>10.3f}{f1_macro:>10.3f}")

    relatorio = classification_report(
        teste_y, previstos, labels=list(INTENCOES), output_dict=True, zero_division=0
    )
    relatorio_baseline = classification_report(
        teste_y, previstos_baseline, labels=list(INTENCOES), output_dict=True, zero_division=0
    )

    print(f"\n{'intenção':<16}{'precisão':>9}{'revocação':>11}{'F1':>7}{'F1 base':>9}{'n':>5}")
    linhas_classe = []
    for intencao in INTENCOES:
        m = relatorio[intencao]
        linhas_classe.append(
            {
                "intencao": intencao,
                "precisao": round(m["precision"], 4),
                "revocacao": round(m["recall"], 4),
                "f1": round(m["f1-score"], 4),
                "f1_baseline": round(relatorio_baseline[intencao]["f1-score"], 4),
                "suporte": int(m["support"]),
            }
        )
        print(
            f"{intencao:<16}{m['precision']:>9.3f}{m['recall']:>11.3f}"
            f"{m['f1-score']:>7.3f}{relatorio_baseline[intencao]['f1-score']:>9.3f}"
            f"{int(m['support']):>5}"
        )
    _salvar_csv(
        PASTA_METRICAS / "pln_relatorio_teste.csv",
        ("intencao", "precisao", "revocacao", "f1", "f1_baseline", "suporte"),
        linhas_classe,
    )

    erros = [
        {
            "texto": texto,
            "esperado": esperado,
            "previsto": previsto,
            "confianca": round(float(confianca), 4),
            "acima_do_limiar": "sim" if confianca >= LIMIAR_CONFIANCA_PLN else "nao",
        }
        for texto, esperado, previsto, confianca in zip(teste_x, teste_y, previstos, confiancas)
        if esperado != previsto
    ]
    erros.sort(key=lambda erro: -erro["confianca"])
    _salvar_csv(
        PASTA_METRICAS / "pln_erros.csv",
        ("texto", "esperado", "previsto", "confianca", "acima_do_limiar"),
        erros,
    )
    print(f"\n{len(erros)} erro(s) no teste:")
    for erro in erros:
        print(f"  {erro['texto']!r} -> {erro['previsto']} (esperado {erro['esperado']}; "
              f"confiança {erro['confianca']:.2f})")

    linhas_limiar = varrer_limiar(teste_y, previstos, confiancas)
    _salvar_csv(
        PASTA_METRICAS / "pln_limiar.csv",
        ("limiar", "cobertura", "acuracia_aceitas", "escopo_atendido",
         "fora_escopo_recusado", "acerto_sistema"),
        [{c: round(v, 4) for c, v in linha.items()} for linha in linhas_limiar],
    )
    no_limiar = min(linhas_limiar, key=lambda l: abs(l["limiar"] - LIMIAR_CONFIANCA_PLN))
    # Entre os limiares de melhor acerto, o maior: é o quanto se pode exigir de
    # confiança sem custo nenhum — a recomendação prática para o módulo de decisão.
    melhor_limiar = max(linhas_limiar, key=lambda l: (l["acerto_sistema"], l["limiar"]))
    print(f"\nLimiar adotado ({LIMIAR_CONFIANCA_PLN}): acerto do sistema "
          f"{no_limiar['acerto_sistema']:.3f}; cobertura {no_limiar['cobertura']:.3f}")
    print(f"Maior limiar sem perda no teste ({melhor_limiar['limiar']:.2f}): acerto do "
          f"sistema {melhor_limiar['acerto_sistema']:.3f}; cobertura "
          f"{melhor_limiar['cobertura']:.3f}")

    _grafico_matriz_confusao(teste_y, previstos)
    _grafico_limiar(linhas_limiar)

    (PASTA_EXEMPLOS / "pln_entrada_saida.md").write_text(
        _tabela_exemplos(classificador), encoding="utf-8"
    )

    (PASTA_METRICAS / "pln_relatorio_teste.json").write_text(
        json.dumps(
            {
                "gerado_em": date.today().isoformat(),
                "modelo": classificador.nome,
                "descricao": classificador.descricao,
                "origem_do_teste": origem_teste,
                "frases_teste": len(teste_x),
                "modelo_treinado": {"acuracia": acuracia, "f1_macro": f1_macro},
                "baseline_palavras_chave": {
                    "acuracia": acuracia_baseline,
                    "f1_macro": f1_baseline,
                },
                "por_classe": linhas_classe,
                "erros": erros,
                "limiar": {
                    "adotado": LIMIAR_CONFIANCA_PLN,
                    "no_limiar_adotado": no_limiar,
                    "maior_limiar_sem_perda": melhor_limiar,
                    "varredura": linhas_limiar,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nMétricas em {PASTA_METRICAS.relative_to(modelos.RAIZ)}/")
    print(f"Gráficos em {PASTA_GRAFICOS.relative_to(modelos.RAIZ)}/")
    print(f"Exemplos em {(PASTA_EXEMPLOS / 'pln_entrada_saida.md').relative_to(modelos.RAIZ)}")


if __name__ == "__main__":
    main()
