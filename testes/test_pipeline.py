import pytest
from PIL import Image

from src.configuracao import ACOES, CLASSES, INTENCOES
from src.decisao.regras import LIXEIRAS, NOMES, PREPARO, RECICLAVEL, decidir
from src.integracao.contratos import ResultadoPLN, ResultadoVisao
from src.integracao.pipeline import processar


def _visao(classe="plastico", confianca=0.9):
    return ResultadoVisao(classe=classe, confianca=confianca, top3=[(classe, confianca)])


def _pln(intencao="como_descartar", confianca=0.9):
    return ResultadoPLN(intencao=intencao, confianca=confianca, texto_original="teste")


def test_pipeline_devolve_contratos_validos():
    resultado = processar(Image.new("RGB", (224, 224)), "onde descarto isso?")
    assert resultado.visao.classe in CLASSES
    assert resultado.pln.intencao in INTENCOES
    assert resultado.decisao.acao in ACOES
    assert resultado.decisao.resposta


@pytest.mark.parametrize("tabela", [NOMES, LIXEIRAS, RECICLAVEL, PREPARO])
def test_regras_cobrem_todas_as_classes(tabela):
    assert set(tabela) == set(CLASSES)


@pytest.mark.parametrize("classe", CLASSES)
@pytest.mark.parametrize("intencao", [i for i in INTENCOES if i != "fora_do_escopo"])
def test_toda_combinacao_classe_intencao_tem_resposta(classe, intencao):
    decisao = decidir(_visao(classe), _pln(intencao))
    assert decisao.acao == "responder"
    assert decisao.lixeira.startswith(LIXEIRAS[classe])


def test_plastico_vai_na_lixeira_vermelha():
    assert decidir(_visao("plastico"), _pln("como_descartar")).lixeira.startswith("vermelha")


def test_baixa_confianca_da_visao_pede_nova_foto():
    assert decidir(_visao(confianca=0.3), _pln()).acao == "pedir_nova_foto"


def test_pergunta_fora_do_escopo_pede_reformulacao():
    assert decidir(_visao(), _pln("fora_do_escopo", 0.0)).acao == "reformular_pergunta"


def test_contrato_rejeita_classe_desconhecida():
    with pytest.raises(ValueError):
        ResultadoVisao(classe="plastic", confianca=0.9)


def test_contrato_rejeita_confianca_fora_do_intervalo():
    with pytest.raises(ValueError):
        ResultadoPLN(intencao="identificar", confianca=1.5, texto_original="x")
