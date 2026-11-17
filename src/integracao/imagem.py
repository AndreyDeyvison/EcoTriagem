"""Normalização das imagens antes da Visão Computacional.

Fotos reais chegam em formatos que o modelo não espera: deitadas (celular
grava a rotação em EXIF), com transparência (PNG), em tons de cinza ou
enormes. Tudo passa por aqui antes de chegar ao classificador.
"""

from PIL import Image, ImageOps

# Lado maior máximo, em pixels. O modelo usa 224x224; 1024 sobra com folga
# e evita processar fotos de 12 MP.
TAMANHO_MAXIMO = 1024

# Fundo branco para imagens com transparência: é o fundo do TrashNet.
COR_FUNDO = (255, 255, 255)


def preparar_imagem(imagem: Image.Image) -> Image.Image:
    """Devolve uma cópia em RGB, na orientação correta e de tamanho limitado."""
    imagem = ImageOps.exif_transpose(imagem)  # sempre devolve uma cópia

    if imagem.mode != "RGB":
        if imagem.mode in ("RGBA", "LA", "PA") or "transparency" in imagem.info:
            com_alfa = imagem.convert("RGBA")
            fundo = Image.new("RGB", com_alfa.size, COR_FUNDO)
            fundo.paste(com_alfa, mask=com_alfa.getchannel("A"))
            imagem = fundo
        else:
            imagem = imagem.convert("RGB")

    imagem.thumbnail((TAMANHO_MAXIMO, TAMANHO_MAXIMO))
    return imagem
