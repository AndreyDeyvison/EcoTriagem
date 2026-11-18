"""Classificação do resíduo a partir da imagem.

Usa a MobileNetV2 ajustada ao TrashNet por `src.machine_learning.treinar`.
O checkpoint é autodescritivo: carrega consigo a arquitetura, a ordem das
classes e os parâmetros de normalização, de modo que a inferência nunca
precisa repetir (nem divergir de) o que foi usado no treino.

O modelo é carregado **uma única vez**, na primeira chamada, e fica em cache no
módulo. Inferência na CPU de propósito: uma imagem por vez roda em dezenas de
milissegundos, e isso evita depender de acelerador na máquina que servir a
interface.

Enquanto `modelos/classificador_residuos.pt` não existir, `classificar_imagem`
devolve o resultado fixo do stub e avisa uma vez no console, em vez de quebrar
o sistema — é o que mantém o protótipo integrado funcionando para quem ainda
não treinou (ver `pipeline.carregar_modelos()` e `usando_stub()`). Quem precisa
do modelo de verdade chama `carregar_modelo_obrigatorio()`, que levanta
`FileNotFoundError` com a instrução de treino.
"""

from functools import cache

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

from src.configuracao import PASTA_MODELOS
from src.integracao.contratos import ResultadoVisao

CAMINHO_MODELO = PASTA_MODELOS / "classificador_residuos.pt"

MENSAGEM_SEM_MODELO = (
    "O modelo de visão ainda não foi treinado: a classificação da imagem está "
    "usando o stub e NÃO é um resultado real. "
    "Treine com: python -m src.machine_learning.treinar"
)

# Resultado fixo do stub, usado só enquanto não há modelo treinado.
_STUB = ResultadoVisao(
    classe="plastico",
    confianca=0.87,
    top3=[("plastico", 0.87), ("vidro", 0.08), ("metal", 0.03)],
)

_DISPOSITIVO = torch.device("cpu")


class ModeloVisao:
    """Modelo treinado + o pré-processamento que ele espera."""

    def __init__(self, rede: nn.Module, classes: list[str], transformar: transforms.Compose):
        self.rede = rede
        self.classes = classes
        self.transformar = transformar

    def probabilidades(self, imagem: Image.Image) -> list[tuple[str, float]]:
        """(classe, probabilidade) para todas as classes, da maior para a menor."""
        if imagem.mode != "RGB":
            imagem = imagem.convert("RGB")
        lote = self.transformar(imagem).unsqueeze(0).to(_DISPOSITIVO)
        with torch.no_grad():
            saida = self.rede(lote)
        probabilidades = torch.softmax(saida, dim=1)[0].tolist()
        pares = list(zip(self.classes, probabilidades))
        pares.sort(key=lambda par: par[1], reverse=True)
        return pares


def _construir_rede(arquitetura: str, n_classes: int) -> nn.Module:
    if arquitetura != "mobilenet_v2":
        raise ValueError(f"arquitetura não suportada no checkpoint: {arquitetura!r}")
    rede = mobilenet_v2(weights=None)
    rede.classifier[1] = nn.Linear(rede.classifier[1].in_features, n_classes)
    return rede


@cache
def carregar_modelo() -> ModeloVisao | None:
    """Carrega o modelo uma única vez (as chamadas seguintes reaproveitam).

    Devolve None enquanto o modelo não existir; aí o módulo usa o stub.
    """
    if not CAMINHO_MODELO.exists():
        return None

    checkpoint = torch.load(CAMINHO_MODELO, map_location=_DISPOSITIVO, weights_only=True)
    classes = list(checkpoint["classes"])
    rede = _construir_rede(checkpoint["arquitetura"], len(classes))
    rede.load_state_dict(checkpoint["state_dict"])
    rede.eval().to(_DISPOSITIVO)

    normalizacao = checkpoint["normalizacao"]
    lado = int(checkpoint["tamanho_entrada"])
    transformar = transforms.Compose(
        [
            transforms.Resize((lado, lado)),
            transforms.ToTensor(),
            transforms.Normalize(normalizacao["media"], normalizacao["desvio"]),
        ]
    )
    return ModeloVisao(rede, classes, transformar)


def carregar_modelo_obrigatorio() -> ModeloVisao:
    """Como `carregar_modelo`, mas falha alto quando não há modelo treinado."""
    modelo = carregar_modelo()
    if modelo is None:
        raise FileNotFoundError(
            f"modelo de visão não encontrado em {CAMINHO_MODELO}. "
            "Treine com: python -m src.machine_learning.treinar"
        )
    return modelo


def usando_stub() -> bool:
    return carregar_modelo() is None


def descricao_do_modelo() -> str:
    """Texto curto sobre o que está classificando, para a interface exibir."""
    modelo = carregar_modelo()
    if modelo is None:
        return "stub (modelo de visão não treinado)"
    return f"MobileNetV2 ajustada ao TrashNet ({len(modelo.classes)} classes)"


_avisou_sem_modelo = False


def classificar_imagem(imagem: Image.Image) -> ResultadoVisao:
    """Recebe a foto do resíduo (já em RGB) e devolve a classe prevista com a confiança."""
    global _avisou_sem_modelo

    modelo = carregar_modelo()
    if modelo is None:
        if not _avisou_sem_modelo:
            _avisou_sem_modelo = True
            print(f"[visao] {MENSAGEM_SEM_MODELO}")
        return _STUB

    ordenadas = modelo.probabilidades(imagem)
    classe, confianca = ordenadas[0]
    return ResultadoVisao(
        classe=classe,
        confianca=float(confianca),
        top3=[(c, float(p)) for c, p in ordenadas[:3]],
    )


__all__ = [
    "CAMINHO_MODELO",
    "MENSAGEM_SEM_MODELO",
    "carregar_modelo",
    "carregar_modelo_obrigatorio",
    "classificar_imagem",
    "descricao_do_modelo",
    "usando_stub",
]
