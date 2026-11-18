"""Leitura da divisão fixa do TrashNet, compartilhada por todos os scripts de visão.

A divisão treino/validação/teste foi fixada uma única vez em
`dados/processados/divisao_trashnet.csv` (ver `dividir_dados.py`) e é lida
sempre daqui. Nenhum script de treino ou avaliação pode refazer a divisão:
isso mudaria os conjuntos entre execuções e invalidaria a comparação entre o
baseline e a CNN.

O CSV também já trata as imagens duplicadas e as de rótulo conflitante do
TrashNet, então ele é a única fonte de verdade sobre quais imagens existem.
"""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.configuracao import CLASSES

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_DIVISAO = RAIZ / "dados" / "processados" / "divisao_trashnet.csv"

CONJUNTOS = ("treino", "validacao", "teste")

# Ordem fixa das classes: o índice de uma classe é o índice de saída da rede.
# Gravado junto com o modelo para que a inferência nunca troque os rótulos.
CLASSES_ORDENADAS = tuple(sorted(CLASSES))


@dataclass(frozen=True)
class Registro:
    """Uma imagem do dataset, com o rótulo e o conjunto a que pertence."""

    caminho: Path
    classe: str
    conjunto: str

    @property
    def indice(self) -> int:
        return CLASSES_ORDENADAS.index(self.classe)


def carregar_divisao(verificar_arquivos: bool = True) -> list[Registro]:
    """Lê o CSV da divisão e devolve os registros com caminho absoluto."""
    if not ARQUIVO_DIVISAO.exists():
        raise FileNotFoundError(
            f"divisão não encontrada em {ARQUIVO_DIVISAO}. Rode antes:\n"
            "  python -m src.machine_learning.baixar_dados\n"
            "  python -m src.machine_learning.dividir_dados"
        )

    registros: list[Registro] = []
    with ARQUIVO_DIVISAO.open(encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        faltando = {"caminho", "classe", "conjunto"} - set(leitor.fieldnames or [])
        if faltando:
            raise ValueError(f"{ARQUIVO_DIVISAO.name}: colunas ausentes {sorted(faltando)}")
        for numero, linha in enumerate(leitor, start=2):
            classe, conjunto = linha["classe"], linha["conjunto"]
            if classe not in CLASSES:
                raise ValueError(f"linha {numero}: classe desconhecida {classe!r}")
            if conjunto not in CONJUNTOS:
                raise ValueError(f"linha {numero}: conjunto desconhecido {conjunto!r}")
            registros.append(Registro(RAIZ / linha["caminho"], classe, conjunto))

    if verificar_arquivos:
        ausentes = [r.caminho for r in registros if not r.caminho.exists()]
        if ausentes:
            raise FileNotFoundError(
                f"{len(ausentes)} imagem(ns) da divisão não estão em disco "
                f"(ex.: {ausentes[0]}). Rode: python -m src.machine_learning.baixar_dados"
            )
    return registros


def por_conjunto(registros: list[Registro]) -> dict[str, list[Registro]]:
    """Agrupa os registros em treino/validação/teste, nessa ordem."""
    return {c: [r for r in registros if r.conjunto == c] for c in CONJUNTOS}


def contagem(registros: list[Registro]) -> Counter:
    return Counter(r.classe for r in registros)


def pesos_por_classe(registros_treino: list[Registro]) -> list[float]:
    """Peso de cada classe para a função de perda, na ordem de CLASSES_ORDENADAS.

    Peso inversamente proporcional à frequência, normalizado para média 1 —
    assim a escala da perda não muda junto com os pesos. A classe `lixo`, com
    137 imagens no dataset inteiro, recebe o maior peso.
    """
    contagens = contagem(registros_treino)
    total = sum(contagens.values())
    n_classes = len(CLASSES_ORDENADAS)
    brutos = [total / (n_classes * contagens[c]) for c in CLASSES_ORDENADAS]
    media = sum(brutos) / n_classes
    return [p / media for p in brutos]


def resumo(registros: list[Registro]) -> str:
    """Tabela classe x conjunto, para imprimir no console."""
    grupos = por_conjunto(registros)
    contagens = {c: contagem(rs) for c, rs in grupos.items()}
    linhas = [f"{'classe':<9} {'treino':>7} {'valid.':>7} {'teste':>7} {'total':>7}"]
    for classe in CLASSES_ORDENADAS:
        valores = [contagens[c][classe] for c in CONJUNTOS]
        linhas.append(
            f"{classe:<9} " + " ".join(f"{v:>7}" for v in valores) + f" {sum(valores):>7}"
        )
    totais = [len(grupos[c]) for c in CONJUNTOS]
    linhas.append(f"{'TOTAL':<9} " + " ".join(f"{t:>7}" for t in totais) + f" {sum(totais):>7}")
    return "\n".join(linhas)
