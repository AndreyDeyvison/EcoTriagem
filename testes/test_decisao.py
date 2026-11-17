import pytest

from src.decisao.regras import REGRAS, decidir, probabilidade_por_lixeira
from src.integracao.contratos import ResultadoPLN, ResultadoVisao


def _visao(*top3):
    classe, confianca = top3[0]
    return ResultadoVisao(classe=classe, confianca=confianca, top3=list(top3))


def _pln(intencao="como_descartar", confianca=0.9):
    return ResultadoPLN(intencao=intencao, confianca=confianca, texto_original="x")


# --- R1 -------------------------------------------------------------------

def test_r1_fora_do_escopo():
    decisao = decidir(_visao(("vidro", 0.9)), _pln("fora_do_escopo", 0.0))
    assert (decisao.acao, decisao.regra) == ("reformular_pergunta", "R1")


def test_r1_pln_pouco_confiante_mesmo_com_intencao_valida():
    decisao = decidir(_visao(("vidro", 0.9)), _pln("como_descartar", 0.3))
    assert (decisao.acao, decisao.regra) == ("reformular_pergunta", "R1")


def test_r1_tem_prioridade_sobre_r4():
    decisao = decidir(_visao(("vidro", 0.3), ("plastico", 0.3)), _pln("fora_do_escopo", 0.0))
    assert decisao.regra == "R1"


# --- R2 -------------------------------------------------------------------

def test_r2_visao_confiante():
    decisao = decidir(_visao(("vidro", 0.85), ("plastico", 0.1)), _pln())
    assert (decisao.acao, decisao.regra, decisao.lixeira) == ("responder", "R2", "verde (vidro)")


def test_r2_exatamente_no_limiar():
    assert decidir(_visao(("metal", 0.6)), _pln()).regra == "R2"


# --- R3 (heurística da lixeira) ------------------------------------------

def test_probabilidade_por_lixeira_soma_materiais_da_mesma_lixeira():
    por_lixeira = probabilidade_por_lixeira(_visao(("papel", 0.45), ("papelao", 0.30), ("vidro", 0.20)))
    assert por_lixeira == pytest.approx({"azul": 0.75, "verde": 0.20})


def test_r3_papel_e_papelao_somam_acima_do_limiar():
    decisao = decidir(_visao(("papel", 0.45), ("papelao", 0.30), ("vidro", 0.20)), _pln())
    assert (decisao.acao, decisao.regra) == ("responder", "R3")
    assert decisao.lixeira == "azul (papel ou papelão)"


def test_r3_lixeira_mais_provavel_pode_nao_ser_a_do_primeiro_palpite():
    # vidro é o palpite individual mais forte, mas a lixeira azul soma 0,65
    decisao = decidir(_visao(("vidro", 0.35), ("papel", 0.33), ("papelao", 0.32)), _pln())
    assert decisao.regra == "R3"
    assert decisao.lixeira.startswith("azul")


def test_r3_respostas_mencionam_os_dois_materiais():
    visao = _visao(("papel", 0.45), ("papelao", 0.30), ("vidro", 0.20))
    assert "papel ou papelão" in decidir(visao, _pln("identificar")).resposta
    preparo = decidir(visao, _pln("como_preparar")).resposta
    assert "Se for papel:" in preparo and "Se for papelão:" in preparo
    assert decidir(visao, _pln("e_reciclavel")).resposta.startswith("Sim")


# --- R4 -------------------------------------------------------------------

def test_r4_palpites_em_lixeiras_diferentes():
    decisao = decidir(_visao(("vidro", 0.40), ("plastico", 0.35), ("metal", 0.20)), _pln())
    assert (decisao.acao, decisao.regra, decisao.lixeira) == ("pedir_nova_foto", "R4", None)


def test_r4_sem_top3_usa_so_a_classe_prevista():
    visao = ResultadoVisao(classe="vidro", confianca=0.4)
    assert decidir(visao, _pln()).regra == "R4"


def test_toda_decisao_informa_uma_regra_documentada():
    casos = [
        (_visao(("vidro", 0.9)), _pln()),
        (_visao(("papel", 0.45), ("papelao", 0.3)), _pln()),
        (_visao(("vidro", 0.4), ("metal", 0.4)), _pln()),
        (_visao(("vidro", 0.9)), _pln("fora_do_escopo", 0.0)),
    ]
    assert {decidir(v, p).regra for v, p in casos} == set(REGRAS)
