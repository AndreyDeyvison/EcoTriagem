"""Carregamento e divisão do dataset de intenções.

Treino e avaliação usam exatamente esta mesma divisão (seed fixa), para que o
número do relatório seja reproduzível e o conjunto de teste nunca influencie a
escolha do modelo.

Se `dados/pln/intencoes_teste.csv` existir, ele é usado como conjunto de teste
e todo o `intencoes.csv` vira treino. Esse arquivo é para as frases escritas
pelos outros integrantes: um teste escrito por outra pessoa não fica viciado no
jeito de escrever de quem montou o treino.
"""

import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.configuracao import INTENCOES

SEED = 42
PROPORCAO_TESTE = 0.2

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO = RAIZ / "dados" / "pln" / "intencoes.csv"
ARQUIVO_TESTE = RAIZ / "dados" / "pln" / "intencoes_teste.csv"


def carregar(caminho: Path = ARQUIVO) -> tuple[list[str], list[str]]:
    """Lê um CSV `texto,intencao` e valida os rótulos contra INTENCOES."""
    if not caminho.exists():
        raise FileNotFoundError(f"dataset não encontrado: {caminho}")

    textos: list[str] = []
    intencoes: list[str] = []
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        faltando = {"texto", "intencao"} - set(leitor.fieldnames or [])
        if faltando:
            raise ValueError(f"{caminho.name}: colunas ausentes {sorted(faltando)}")
        for numero, linha in enumerate(leitor, start=2):
            texto = (linha["texto"] or "").strip()
            intencao = (linha["intencao"] or "").strip()
            if not texto:
                raise ValueError(f"{caminho.name}: linha {numero} com texto vazio")
            if intencao not in INTENCOES:
                raise ValueError(
                    f"{caminho.name}: linha {numero} com intenção desconhecida "
                    f"{intencao!r}; use uma de {INTENCOES}"
                )
            textos.append(texto)
            intencoes.append(intencao)

    if not textos:
        raise ValueError(f"{caminho.name}: dataset vazio")
    return textos, intencoes


def dividir() -> tuple[list[str], list[str], list[str], list[str], str]:
    """Devolve (treino_x, treino_y, teste_x, teste_y, origem_do_teste)."""
    textos, intencoes = carregar()

    if ARQUIVO_TESTE.exists():
        teste_x, teste_y = carregar(ARQUIVO_TESTE)
        return textos, intencoes, teste_x, teste_y, ARQUIVO_TESTE.name

    treino_x, teste_x, treino_y, teste_y = train_test_split(
        textos,
        intencoes,
        test_size=PROPORCAO_TESTE,
        stratify=intencoes,
        random_state=SEED,
    )
    return treino_x, treino_y, teste_x, teste_y, f"divisão estratificada {1 - PROPORCAO_TESTE:.0%}/{PROPORCAO_TESTE:.0%} (seed {SEED})"


def resumo(intencoes: list[str]) -> str:
    """Contagem por intenção, para imprimir no console."""
    contagem = Counter(intencoes)
    linhas = [f"  {i:<16} {contagem.get(i, 0):>4}" for i in INTENCOES]
    return "\n".join(linhas) + f"\n  {'TOTAL':<16} {len(intencoes):>4}"
