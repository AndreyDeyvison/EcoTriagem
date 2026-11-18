"""Fluxo único do sistema: imagem + pergunta -> resposta (requisito 4.5).

O fluxo é dividido em duas etapas para permitir várias perguntas sobre a
mesma foto sem reprocessar a imagem:
    visao = analisar_imagem(foto)              # uma vez por foto
    responder_pergunta(visao, "é reciclável?") # uma vez por pergunta
"""

from PIL import Image

from src.decisao.regras import decidir
from src.integracao.contratos import ResultadoPipeline, ResultadoVisao
from src.integracao.imagem import preparar_imagem
from src.pln import intencoes
from src.visao_computacional import classificador

# Módulos com modelo treinado: nome exibido -> módulo que expõe carregar_modelo().
_MODULOS_COM_MODELO = {
    "Visão Computacional": classificador,
    "PLN": intencoes,
}


def carregar_modelos() -> dict[str, bool]:
    """Carrega todos os modelos de uma vez (chamar ao iniciar o app).

    Devolve, para cada módulo, True se o modelo real foi carregado e False se
    ele ainda está usando o stub.
    """
    return {nome: modulo.carregar_modelo() is not None for nome, modulo in _MODULOS_COM_MODELO.items()}


def analisar_imagem(imagem: Image.Image) -> ResultadoVisao:
    return classificador.classificar_imagem(preparar_imagem(imagem))


def responder_pergunta(visao: ResultadoVisao, pergunta: str, origem: str | None = None) -> ResultadoPipeline:
    """origem: bairro do usuário, usado quando a pergunta é sobre ponto de coleta."""
    resultado_pln = intencoes.interpretar_pergunta(pergunta)
    decisao = decidir(visao, resultado_pln, origem)
    return ResultadoPipeline(visao=visao, pln=resultado_pln, decisao=decisao)


def processar(imagem: Image.Image, pergunta: str) -> ResultadoPipeline:
    """Atalho para o fluxo completo de uma foto com uma pergunta."""
    return responder_pergunta(analisar_imagem(imagem), pergunta)
