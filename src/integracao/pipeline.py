"""Fluxo único do sistema: imagem + pergunta -> resposta (requisito 4.5)."""

from PIL import Image

from src.decisao.regras import decidir
from src.integracao.contratos import ResultadoPipeline
from src.pln.intencoes import interpretar_pergunta
from src.visao_computacional.classificador import classificar_imagem


def processar(imagem: Image.Image, pergunta: str) -> ResultadoPipeline:
    visao = classificar_imagem(imagem)
    pln = interpretar_pergunta(pergunta)
    decisao = decidir(visao, pln)
    return ResultadoPipeline(visao=visao, pln=pln, decisao=decisao)
