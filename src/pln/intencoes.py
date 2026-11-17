"""Reconhecimento da intenção da pergunta do usuário.

Responsável: PLN (Andrey).

Enquanto não existir um modelo treinado em CAMINHO_MODELO, a função
interpretar_pergunta usa casamento de palavras-chave (stub). Isso NÃO conta
como PLN para o enunciado; deve virar um classificador treinado (ex.:
TF-IDF + Regressão Logística) com dataset e métricas próprias. Para ligar
o modelo real:
  1. salvar o modelo treinado em CAMINHO_MODELO;
  2. implementar o carregamento em carregar_modelo();
  3. implementar a inferência no fim de interpretar_pergunta().
Manter a assinatura e o tipo de retorno (ResultadoPLN).
"""

import unicodedata
from functools import cache

from src.configuracao import PASTA_MODELOS
from src.integracao.contratos import ResultadoPLN

CAMINHO_MODELO = PASTA_MODELOS / "pln_intencoes.joblib"

# Stub: a ordem importa, a primeira intenção com palavra encontrada vence.
_PALAVRAS_CHAVE = {
    "ponto_coleta": ("ecoponto", "ponto de coleta", "onde entrego", "mais perto", "mais proximo"),
    "cor_lixeira": ("cor", "lixeira"),
    "como_preparar": ("lavar", "limpar", "preparar", "amassar", "antes de"),
    "e_reciclavel": ("reciclavel", "recicla"),
    "como_descartar": ("descart", "jogo", "jogar", "onde coloco", "onde ponho"),
    "identificar": ("o que e", "que material", "qual material", "identific"),
}


@cache
def carregar_modelo():
    """Carrega o modelo uma única vez. Devolve None enquanto ele não existir."""
    if not CAMINHO_MODELO.exists():
        return None
    # TODO(Andrey): carregar e devolver o modelo (ex.: joblib.load).
    return None


def usando_stub() -> bool:
    return carregar_modelo() is None


def _normalizar(texto: str) -> str:
    sem_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acentos.lower().strip()


def _interpretar_por_palavras_chave(texto: str) -> ResultadoPLN:
    normalizado = _normalizar(texto)
    for intencao, palavras in _PALAVRAS_CHAVE.items():
        if any(p in normalizado for p in palavras):
            return ResultadoPLN(intencao=intencao, confianca=0.9, texto_original=texto)
    return ResultadoPLN(intencao="fora_do_escopo", confianca=0.0, texto_original=texto)


def interpretar_pergunta(texto: str) -> ResultadoPLN:
    """Recebe a pergunta em linguagem natural e devolve a intenção reconhecida."""
    modelo = carregar_modelo()
    if modelo is None:
        return _interpretar_por_palavras_chave(texto)
    # TODO(Andrey): rodar o modelo e montar ResultadoPLN com a intenção e a confiança.
    raise NotImplementedError("inferência do modelo de PLN ainda não implementada")
