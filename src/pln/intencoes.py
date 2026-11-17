"""Reconhecimento da intenção da pergunta do usuário.

STUB: casamento simples de palavras-chave, só para a integração funcionar.
Isso NÃO conta como PLN para o enunciado; deve ser trocado por um
classificador treinado (ex.: TF-IDF + Regressão Logística) com dataset e
métricas próprias. Responsável: PLN. Ao substituir, manter a assinatura
e o tipo de retorno.
"""

import unicodedata

from src.integracao.contratos import ResultadoPLN

# A ordem importa: a primeira intenção com palavra encontrada vence.
_PALAVRAS_CHAVE = {
    "ponto_coleta": ("ecoponto", "ponto de coleta", "onde entrego", "mais perto", "mais proximo"),
    "cor_lixeira": ("cor", "lixeira"),
    "como_preparar": ("lavar", "limpar", "preparar", "amassar", "antes de"),
    "e_reciclavel": ("reciclavel", "recicla"),
    "como_descartar": ("descart", "jogo", "jogar", "onde coloco", "onde ponho"),
    "identificar": ("o que e", "que material", "qual material", "identific"),
}


def _normalizar(texto: str) -> str:
    sem_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acentos.lower().strip()


def interpretar_pergunta(texto: str) -> ResultadoPLN:
    """Recebe a pergunta em linguagem natural e devolve a intenção reconhecida."""
    normalizado = _normalizar(texto)
    for intencao, palavras in _PALAVRAS_CHAVE.items():
        if any(p in normalizado for p in palavras):
            return ResultadoPLN(intencao=intencao, confianca=0.9, texto_original=texto)
    return ResultadoPLN(intencao="fora_do_escopo", confianca=0.0, texto_original=texto)
