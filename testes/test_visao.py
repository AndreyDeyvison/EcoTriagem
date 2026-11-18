"""Testes do módulo de Visão Computacional & ML.

Os testes que dependem do modelo treinado ou das imagens do TrashNet são
pulados quando eles não estão em disco. O modelo vem versionado, mas as imagens
do TrashNet não (46 MB): quem clona precisa rodar `baixar_dados` para os testes
que leem o dataset.
"""

import numpy as np
import pytest
from PIL import Image

from src.configuracao import CLASSES, LIMIAR_CONFIANCA_VISAO
from src.integracao.contratos import ResultadoVisao
from src.machine_learning import avaliar, dados_visao, treinar_baseline
from src.visao_computacional import classificador

sem_modelo = pytest.mark.skipif(
    not classificador.CAMINHO_MODELO.exists(),
    reason="modelo não treinado; rode python -m src.machine_learning.treinar",
)


def _tem_imagens() -> bool:
    try:
        dados_visao.carregar_divisao()
    except FileNotFoundError:
        return False
    return True


sem_imagens = pytest.mark.skipif(
    not _tem_imagens(), reason="TrashNet não baixado; rode python -m src.machine_learning.baixar_dados"
)


# ------------------------------------------------------ contrato da inferência


@pytest.mark.parametrize(
    ("modo", "tamanho"),
    [("RGB", (512, 384)), ("RGBA", (300, 200)), ("L", (224, 224)), ("P", (128, 128))],
)
def test_classificar_imagem_respeita_o_contrato(modo, tamanho):
    """Vale para o modelo treinado e para o stub: a saída é sempre válida."""
    resultado = classificador.classificar_imagem(Image.new(modo, tamanho))

    assert isinstance(resultado, ResultadoVisao)
    assert resultado.classe in CLASSES
    assert isinstance(resultado.confianca, float)
    assert 0.0 <= resultado.confianca <= 1.0

    assert len(resultado.top3) == 3
    for classe, probabilidade in resultado.top3:
        assert classe in CLASSES
        assert 0.0 <= probabilidade <= 1.0

    probabilidades = [p for _, p in resultado.top3]
    assert probabilidades == sorted(probabilidades, reverse=True), "top3 deve vir ordenado"
    assert resultado.top3[0][0] == resultado.classe
    assert resultado.top3[0][1] == pytest.approx(resultado.confianca)
    assert len({c for c, _ in resultado.top3}) == 3, "top3 não pode repetir classe"


def test_imagem_em_escala_de_cinza_nao_quebra():
    assert classificador.classificar_imagem(Image.new("L", (64, 64), 128)).classe in CLASSES


def test_modelo_carregado_uma_unica_vez():
    primeiro = classificador.carregar_modelo()
    for _ in range(3):
        classificador.classificar_imagem(Image.new("RGB", (64, 64)))
    assert classificador.carregar_modelo() is primeiro


@sem_modelo
def test_probabilidades_somam_um():
    modelo = classificador.carregar_modelo()
    pares = modelo.probabilidades(Image.new("RGB", (224, 224), (90, 120, 80)))
    assert len(pares) == len(CLASSES)
    assert sum(p for _, p in pares) == pytest.approx(1.0, abs=1e-5)


@sem_modelo
def test_checkpoint_tem_os_metadados_da_inferencia():
    import torch

    checkpoint = torch.load(classificador.CAMINHO_MODELO, map_location="cpu", weights_only=True)
    for chave in ("arquitetura", "state_dict", "classes", "normalizacao", "tamanho_entrada", "seed"):
        assert chave in checkpoint, f"falta {chave} no checkpoint"
    assert sorted(checkpoint["classes"]) == sorted(CLASSES)
    assert checkpoint["seed"] == 42
    assert set(checkpoint["normalizacao"]) == {"media", "desvio"}


def test_carregar_modelo_obrigatorio_falha_com_mensagem_clara(tmp_path, monkeypatch):
    monkeypatch.setattr(classificador, "CAMINHO_MODELO", tmp_path / "inexistente.pt")
    classificador.carregar_modelo.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="src.machine_learning.treinar"):
            classificador.carregar_modelo_obrigatorio()
        assert classificador.usando_stub()
        assert classificador.classificar_imagem(Image.new("RGB", (64, 64))).classe in CLASSES
    finally:
        classificador.carregar_modelo.cache_clear()


# --------------------------------------------------------- divisão do dataset


def test_divisao_tem_os_tamanhos_fixados():
    grupos = dados_visao.por_conjunto(dados_visao.carregar_divisao(verificar_arquivos=False))
    assert len(grupos["treino"]) == 1764
    assert len(grupos["validacao"]) == 378
    assert len(grupos["teste"]) == 379


def test_nenhuma_imagem_aparece_em_dois_conjuntos():
    registros = dados_visao.carregar_divisao(verificar_arquivos=False)
    caminhos = [r.caminho for r in registros]
    assert len(caminhos) == len(set(caminhos)), "imagem repetida na divisão"


def test_indice_da_classe_bate_com_a_ordem_do_modelo():
    registros = dados_visao.carregar_divisao(verificar_arquivos=False)
    for registro in registros[:50]:
        assert dados_visao.CLASSES_ORDENADAS[registro.indice] == registro.classe
    assert sorted(dados_visao.CLASSES_ORDENADAS) == sorted(CLASSES)


def test_pesos_compensam_o_desbalanceamento():
    treino = dados_visao.por_conjunto(dados_visao.carregar_divisao(verificar_arquivos=False))["treino"]
    pesos = dados_visao.pesos_por_classe(treino)
    assert len(pesos) == len(CLASSES)
    contagens = dados_visao.contagem(treino)
    menor = min(contagens, key=contagens.get)
    maior = max(contagens, key=contagens.get)
    indice = dados_visao.CLASSES_ORDENADAS.index
    assert pesos[indice(menor)] > pesos[indice(maior)], "a classe rara precisa pesar mais"
    assert np.mean(pesos) == pytest.approx(1.0)


# ------------------------------------------------------ baseline e avaliação


@sem_imagens
def test_caracteristicas_do_baseline_tem_o_tamanho_esperado():
    imagem = Image.new("RGB", (treinar_baseline.LADO, treinar_baseline.LADO), (40, 160, 90))
    hsv = treinar_baseline.histograma_hsv(imagem)
    hog = treinar_baseline.hog(imagem)
    assert hsv.shape == (treinar_baseline.BINS_MATIZ + treinar_baseline.BINS_SATURACAO
                         + treinar_baseline.BINS_VALOR,)
    assert hog.shape == (treinar_baseline.GRADE ** 2 * treinar_baseline.BINS_ORIENTACAO,)
    assert hsv.sum() == pytest.approx(3.0), "um histograma normalizado por canal"
    assert np.isfinite(hog).all()


def test_varredura_do_limiar_e_coerente():
    verdadeiros = ["papel", "papel", "vidro", "metal"]
    previstos = ["papel", "vidro", "vidro", "metal"]
    confiancas = np.array([0.95, 0.40, 0.80, 0.55])

    linhas = avaliar.varrer_limiar(verdadeiros, previstos, confiancas)
    coberturas = [linha["cobertura"] for linha in linhas]
    assert coberturas == sorted(coberturas, reverse=True), "cobertura não pode subir com o limiar"
    assert all(0.0 <= linha["acuracia_aceitas"] <= 1.0 for linha in linhas)
    assert linhas[0]["limiar"] == pytest.approx(0.30)

    no_limiar = next(l for l in linhas if l["limiar"] == pytest.approx(LIMIAR_CONFIANCA_VISAO))
    assert no_limiar["aceitas"] == 2  # só 0.95 e 0.80 passam de 0.60
    assert no_limiar["pedidos_de_nova_foto"] == 2


@pytest.mark.parametrize(
    ("nome", "valido"),
    [
        ("plastico_andrey_03.jpg", True),
        ("vidro_matteus_12.JPG", True),
        ("papel_caio_1.png", True),
        ("foto1.jpg", False),
        ("plastico-andrey-03.jpg", False),
    ],
)
def test_padrao_de_nome_das_fotos_da_equipe(nome, valido):
    assert bool(avaliar.PADRAO_FOTO_EQUIPE.match(nome)) is valido
