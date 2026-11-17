"""Compara os modelos de intenção e salva o melhor em `modelos/`.

A comparação usa validação cruzada estratificada de 5 partes **sobre o
conjunto de treino**. Com poucas centenas de frases, um único conjunto de
validação teria ~50 exemplos e a diferença entre dois modelos ficaria dentro do
ruído; a validação cruzada usa todo o treino para comparar e ainda informa o
desvio-padrão entre as partes.

O conjunto de teste não é tocado aqui — ele só aparece em `avaliar.py`, uma
única vez, depois que o modelo já está escolhido.

Executar a partir da raiz do repositório:
    python -m src.pln.treinar
"""

import csv
import json
from datetime import date

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.pln import baseline, dados, modelos
from src.pln.modelos import ClassificadorIntencoes

PARTES = 5
SEED = 42

PASTA_METRICAS = modelos.RAIZ / "resultados" / "metricas"
ARQUIVO_COMPARACAO_CSV = PASTA_METRICAS / "pln_comparacao_modelos.csv"
ARQUIVO_COMPARACAO_JSON = PASTA_METRICAS / "pln_comparacao_modelos.json"

COLUNAS = ("modelo", "descricao", "f1_macro", "f1_macro_desvio", "acuracia", "tipo")


def _particoes() -> StratifiedKFold:
    return StratifiedKFold(n_splits=PARTES, shuffle=True, random_state=SEED)


def avaliar_baseline(textos: list[str], intencoes: list[str]) -> dict:
    """Mede o baseline nas mesmas partes de validação dos outros modelos.

    O baseline não treina (são regras fixas), mas é medido nos mesmos
    subconjuntos para que o número seja comparável linha a linha.
    """
    y = np.array(intencoes)
    f1s, acuracias = [], []
    for _, indices_validacao in _particoes().split(textos, y):
        verdadeiros = y[indices_validacao]
        previstos = baseline.prever([textos[i] for i in indices_validacao])
        f1s.append(f1_score(verdadeiros, previstos, average="macro", zero_division=0))
        acuracias.append(accuracy_score(verdadeiros, previstos))
    return {
        "modelo": "baseline_palavras_chave",
        "descricao": "Baseline: casamento de palavras-chave (versão anterior do stub)",
        "f1_macro": float(np.mean(f1s)),
        "f1_macro_desvio": float(np.std(f1s)),
        "acuracia": float(np.mean(acuracias)),
        "tipo": "baseline",
    }


def avaliar_modelo(nome: str, textos: list[str], intencoes: list[str]) -> dict:
    """Validação cruzada de um modelo do catálogo."""
    _, _, ablacao = modelos.CATALOGO[nome]
    resultado = cross_validate(
        modelos.construir(nome),
        textos,
        intencoes,
        cv=_particoes(),
        scoring=("f1_macro", "accuracy"),
        n_jobs=1,
    )
    return {
        "modelo": nome,
        "descricao": modelos.descricao(nome),
        "f1_macro": float(resultado["test_f1_macro"].mean()),
        "f1_macro_desvio": float(resultado["test_f1_macro"].std()),
        "acuracia": float(resultado["test_accuracy"].mean()),
        "tipo": "ablacao" if ablacao else "candidato",
    }


def _salvar_comparacao(linhas: list[dict], contexto: dict) -> None:
    PASTA_METRICAS.mkdir(parents=True, exist_ok=True)
    with ARQUIVO_COMPARACAO_CSV.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS)
        escritor.writeheader()
        for linha in linhas:
            escritor.writerow({c: linha[c] for c in COLUNAS})
    ARQUIVO_COMPARACAO_JSON.write_text(
        json.dumps({**contexto, "modelos": linhas}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _imprimir_tabela(linhas: list[dict]) -> None:
    print(f"\n{'modelo':<32} {'F1-macro':>12} {'acurácia':>9}  tipo")
    print("-" * 70)
    for linha in linhas:
        f1 = f"{linha['f1_macro']:.3f} ±{linha['f1_macro_desvio']:.3f}"
        print(f"{linha['modelo']:<32} {f1:>12} {linha['acuracia']:>9.3f}  {linha['tipo']}")


def main() -> None:
    treino_x, treino_y, teste_x, _, origem_teste = dados.dividir()
    print(f"Treino: {len(treino_x)} frases | Teste: {len(teste_x)} frases ({origem_teste})")
    print("Distribuição do treino:")
    print(dados.resumo(treino_y))
    print(f"\nValidação cruzada estratificada de {PARTES} partes (seed {SEED}):")

    linhas = [avaliar_baseline(treino_x, treino_y)]
    for nome in modelos.CATALOGO:
        print(f"  treinando {nome}...")
        linhas.append(avaliar_modelo(nome, treino_x, treino_y))

    linhas.sort(key=lambda linha: linha["f1_macro"], reverse=True)
    _imprimir_tabela(linhas)

    melhor = max(
        (linha for linha in linhas if linha["tipo"] == "candidato"),
        key=lambda linha: linha["f1_macro"],
    )
    nome_melhor = melhor["modelo"]
    print(f"\nMelhor modelo pelo F1-macro da validação cruzada: {nome_melhor}")

    pipeline = modelos.construir(nome_melhor)
    pipeline.fit(treino_x, treino_y)
    classificador = ClassificadorIntencoes(
        nome=nome_melhor, descricao=modelos.descricao(nome_melhor), pipeline=pipeline
    )
    caminho = modelos.salvar(classificador)

    _salvar_comparacao(
        linhas,
        {
            "gerado_em": date.today().isoformat(),
            "seed": SEED,
            "partes_validacao_cruzada": PARTES,
            "frases_treino": len(treino_x),
            "frases_teste": len(teste_x),
            "origem_do_teste": origem_teste,
            "modelo_escolhido": nome_melhor,
        },
    )

    print(f"Modelo salvo em {caminho.relative_to(modelos.RAIZ)}")
    print(f"Comparação salva em {ARQUIVO_COMPARACAO_CSV.relative_to(modelos.RAIZ)}")
    print("\nPróximo passo: python -m src.pln.avaliar")


if __name__ == "__main__":
    main()
