"""Classificação do resíduo a partir da imagem.

STUB: devolve sempre o mesmo resultado para a integração funcionar
enquanto o modelo real (ex.: MobileNetV2 treinado no TrashNet) é feito.
Responsável: Visão Computacional & ML. Ao substituir, manter a assinatura
e o tipo de retorno.
"""

from PIL import Image

from src.integracao.contratos import ResultadoVisao


def classificar_imagem(imagem: Image.Image) -> ResultadoVisao:
    """Recebe a foto do resíduo e devolve a classe prevista com a confiança."""
    return ResultadoVisao(
        classe="plastico",
        confianca=0.87,
        top3=[("plastico", 0.87), ("vidro", 0.08), ("metal", 0.03)],
    )
