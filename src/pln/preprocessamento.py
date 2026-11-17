"""Pré-processamento do texto das perguntas.

A normalização é usada tanto no treino quanto na inferência (é o
`preprocessor` dos vetorizadores), para que o modelo nunca veja no uso real
um texto tratado de forma diferente do que viu no treino.

Decisão importante: as negações ("não", "nem", "sem", "nunca") NÃO entram na
lista de stopwords. "isso não é reciclável?" e "isso é reciclável?" têm a
mesma intenção, mas em outras tarefas a negação inverte o sentido, e remover
"não" é um erro clássico de PLN em português. A lista `STOPWORDS` já exclui
as negações; `remover_stopwords` nunca as descarta.
"""

import re
import unicodedata

# Negações: preservadas sempre.
NEGACOES = frozenset({"nao", "nem", "sem", "nunca", "nenhum", "nenhuma"})

# Stopwords do português (lista enxuta, já sem acento porque são aplicadas
# depois da normalização). As negações são removidas desta lista no final.
STOPWORDS = frozenset(
    {
        "a", "ao", "aos", "as", "da", "das", "de", "do", "dos", "e", "em", "na",
        "nas", "no", "nos", "o", "os", "um", "uma", "uns", "umas", "pra", "para",
        "por", "com", "que", "se", "sobre", "ser", "sou", "era", "ja", "ai",
        "aqui", "esse", "essa", "esses", "essas", "este", "esta", "isso", "isto",
        "aquele", "aquela", "ele", "ela", "eu", "me", "meu", "minha", "eh",
        "voce", "vc", "te", "tu", "mais", "muito", "tem", "ter", "sao", "foi",
        "the", "of",
    }
) - NEGACOES

_NAO_ALFANUMERICO = re.compile(r"[^a-z0-9\s]+")
_ESPACOS = re.compile(r"\s+")
_REPETIDOS = re.compile(r"(.)\1{2,}")


def remover_acentos(texto: str) -> str:
    """"reciclável" -> "reciclavel". Resolve metade das variações de escrita."""
    decomposto = unicodedata.normalize("NFKD", texto)
    return decomposto.encode("ascii", "ignore").decode("ascii")


def normalizar(texto: str) -> str:
    """Minúsculas, sem acento, sem pontuação e sem espaços sobrando.

    Também encurta letras repetidas de ênfase ("aaaaaa" -> "aa", "kkkkk" -> "kk")
    para que "ooooi" e "oi" compartilhem subsequências de caracteres.
    """
    texto = remover_acentos(texto).lower()
    texto = _NAO_ALFANUMERICO.sub(" ", texto)
    texto = _REPETIDOS.sub(r"\1\1", texto)
    return _ESPACOS.sub(" ", texto).strip()


def tokenizar(texto: str) -> list[str]:
    """Normaliza e quebra em palavras."""
    normalizado = normalizar(texto)
    return normalizado.split() if normalizado else []


def remover_stopwords(tokens: list[str]) -> list[str]:
    """Descarta stopwords, mas nunca as negações."""
    return [t for t in tokens if t in NEGACOES or t not in STOPWORDS]


def normalizar_sem_stopwords(texto: str) -> str:
    """Variação usada por um dos modelos comparados (ver src/pln/modelos.py)."""
    return " ".join(remover_stopwords(tokenizar(texto)))


def normalizar_sem_negacoes(texto: str) -> str:
    """Variação só para o experimento de ablação: remove também as negações.

    Serve para medir no artigo o quanto se perde ao tratar "não" como
    stopword. Não é usada em produção.
    """
    tokens = [t for t in tokenizar(texto) if t not in STOPWORDS and t not in NEGACOES]
    return " ".join(tokens)


def tem_conteudo(texto: str) -> bool:
    """False para texto vazio ou só com pontuação/emoji (ex.: "?????").

    Entradas assim não têm nenhuma característica aproveitável: o
    classificador as trata como fora do escopo por regra (src/pln/modelos.py).
    """
    return bool(normalizar(texto))
