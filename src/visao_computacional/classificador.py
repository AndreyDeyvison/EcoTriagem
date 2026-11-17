"""Classificação do resíduo a partir da imagem.

Responsável: Visão Computacional & ML (Matteus).

Enquanto não existir um modelo treinado em CAMINHO_MODELO, a função
classificar_imagem devolve um resultado fixo (stub) para o resto do sistema
funcionar. Para ligar o modelo real:
  1. salvar o modelo treinado em CAMINHO_MODELO;
  2. implementar o carregamento em carregar_modelo();
  3. implementar a inferência no fim de classificar_imagem().
Manter a assinatura e o tipo de retorno (ResultadoVisao).
"""

from functools import cache

from PIL import Image

from src.configuracao import PASTA_MODELOS
from src.integracao.contratos import ResultadoVisao

CAMINHO_MODELO = PASTA_MODELOS / "visao_trashnet.pt"


@cache
def carregar_modelo():
    """Carrega o modelo uma única vez (as chamadas seguintes reaproveitam).

    Devolve None enquanto o modelo não existir; aí o módulo usa o stub.
    """
    if not CAMINHO_MODELO.exists():
        return None
    # TODO(Matteus): carregar e devolver o modelo (ex.: torch.load + model.eval()).
    return None


def usando_stub() -> bool:
    return carregar_modelo() is None


def classificar_imagem(imagem: Image.Image) -> ResultadoVisao:
    """Recebe a foto do resíduo (já em RGB) e devolve a classe prevista com a confiança."""
    modelo = carregar_modelo()
    if modelo is None:
        return ResultadoVisao(
            classe="plastico",
            confianca=0.87,
            top3=[("plastico", 0.87), ("vidro", 0.08), ("metal", 0.03)],
        )
    # TODO(Matteus): pré-processar (224x224, normalização ImageNet), rodar o
    # modelo e montar ResultadoVisao com a classe, a confiança e o top 3.
    raise NotImplementedError("inferência do modelo de visão ainda não implementada")
