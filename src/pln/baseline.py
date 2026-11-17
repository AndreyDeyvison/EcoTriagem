"""Baseline de palavras-chave: a versão que existia antes do classificador.

Guardado aqui de propósito. É o ponto de comparação do artigo ("de quanto o
modelo treinado ganha de uma busca por palavras-chave?") e também o plano B
usado por `interpretar_pergunta` enquanto o modelo não foi treinado.

A lógica é a original: a primeira intenção cuja palavra-chave aparece no texto
vence — ou seja, a ordem do dicionário é a prioridade. Nada aqui é aprendido
dos dados, e é justamente essa a limitação que o enunciado pede para superar.
"""

from src.pln.preprocessamento import normalizar

# A ordem importa: a primeira intenção com palavra encontrada vence.
PALAVRAS_CHAVE: dict[str, tuple[str, ...]] = {
    "ponto_coleta": ("ecoponto", "ponto de coleta", "onde entrego", "mais perto", "mais proximo"),
    "cor_lixeira": ("cor", "lixeira"),
    "como_preparar": ("lavar", "limpar", "preparar", "amassar", "antes de"),
    "e_reciclavel": ("reciclavel", "recicla"),
    "como_descartar": ("descart", "jogo", "jogar", "onde coloco", "onde ponho"),
    "identificar": ("o que e", "que material", "qual material", "identific"),
}

# Confiança fixa que o stub devolvia: alta quando casou, zero quando não casou.
CONFIANCA_ACERTO = 0.9


def classificar(texto: str) -> tuple[str, float]:
    """Devolve (intenção, confiança) por casamento de palavra-chave."""
    normalizado = normalizar(texto)
    for intencao, palavras in PALAVRAS_CHAVE.items():
        if any(palavra in normalizado for palavra in palavras):
            return intencao, CONFIANCA_ACERTO
    return "fora_do_escopo", 0.0


def prever(textos: list[str]) -> list[str]:
    """Só os rótulos, para comparar com os modelos treinados."""
    return [classificar(t)[0] for t in textos]
