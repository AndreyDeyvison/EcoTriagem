import io

from PIL import Image

from src.integracao.imagem import COR_FUNDO, TAMANHO_MAXIMO, preparar_imagem

ORIENTACAO_EXIF = 0x0112


def _jpeg_com_orientacao(largura, altura, orientacao):
    exif = Image.Exif()
    exif[ORIENTACAO_EXIF] = orientacao
    buffer = io.BytesIO()
    Image.new("RGB", (largura, altura), "red").save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)
    return Image.open(buffer)


def test_foto_de_celular_deitada_e_desvirada():
    # Orientação 6 = câmera girada 90°: a foto gravada 40x20 deve ser exibida 20x40.
    resultado = preparar_imagem(_jpeg_com_orientacao(40, 20, 6))
    assert resultado.size == (20, 40)


def test_png_transparente_vira_rgb_com_fundo_branco():
    imagem = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    resultado = preparar_imagem(imagem)
    assert resultado.mode == "RGB"
    assert resultado.getpixel((5, 5)) == COR_FUNDO


def test_paleta_com_transparencia_vira_rgb_com_fundo_branco():
    imagem = Image.new("P", (10, 10), 0)
    imagem.info["transparency"] = 0
    resultado = preparar_imagem(imagem)
    assert resultado.mode == "RGB"
    assert resultado.getpixel((5, 5)) == COR_FUNDO


def test_tons_de_cinza_vira_rgb():
    assert preparar_imagem(Image.new("L", (10, 10), 128)).mode == "RGB"


def test_imagem_grande_e_reduzida_mantendo_proporcao():
    resultado = preparar_imagem(Image.new("RGB", (4000, 3000)))
    assert max(resultado.size) == TAMANHO_MAXIMO
    assert resultado.size == (TAMANHO_MAXIMO, TAMANHO_MAXIMO * 3 // 4)


def test_imagem_pequena_nao_e_ampliada():
    assert preparar_imagem(Image.new("RGB", (100, 50))).size == (100, 50)


def test_imagem_original_nao_e_alterada():
    original = Image.new("RGBA", (4000, 3000))
    preparar_imagem(original)
    assert original.size == (4000, 3000)
    assert original.mode == "RGBA"
