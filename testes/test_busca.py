from collections import deque

import pytest

from src.decisao.busca import (
    ALGORITMOS,
    busca_a_estrela,
    busca_custo_uniforme,
    busca_largura,
    buscar_ponto_coleta,
    heuristica,
)
from src.decisao.mapa import carregar_mapa, distancia_km
from src.decisao.regras import RECICLAVEL, decidir
from src.integracao.contratos import ResultadoPLN, ResultadoVisao

MAPA = carregar_mapa()
RECICLAVEIS = [m for m, r in RECICLAVEL.items() if r]
CONSULTAS = [(o, m) for o in MAPA.bairros for m in RECICLAVEIS]


def _objetivos(material):
    return {p.bairro for p in MAPA.pontos_que_aceitam([material])}


# --- mapa -----------------------------------------------------------------

def test_distancia_haversine_conhecida():
    # 1 grau de latitude ~ 111,2 km
    assert distancia_km((0.0, 0.0), (1.0, 0.0)) == pytest.approx(111.19, abs=0.05)


def test_mapa_e_conexo():
    inicio = MAPA.bairros[0]
    vistos, fila = {inicio}, deque([inicio])
    while fila:
        for v, _ in MAPA.vizinhos[fila.popleft()]:
            if v not in vistos:
                vistos.add(v)
                fila.append(v)
    assert vistos == set(MAPA.bairros)


def test_todo_material_reciclavel_tem_ponto_de_coleta():
    for material in RECICLAVEIS:
        assert MAPA.pontos_que_aceitam([material]), material


# --- algoritmos -----------------------------------------------------------

@pytest.mark.parametrize(("origem", "material"), CONSULTAS)
def test_a_estrela_encontra_o_caminho_otimo(origem, material):
    objetivos = _objetivos(material)
    assert busca_a_estrela(MAPA, origem, objetivos).custo_km == pytest.approx(
        busca_custo_uniforme(MAPA, origem, objetivos).custo_km
    )


@pytest.mark.parametrize(("origem", "material"), CONSULTAS)
def test_a_estrela_nunca_expande_mais_que_custo_uniforme(origem, material):
    objetivos = _objetivos(material)
    a = busca_a_estrela(MAPA, origem, objetivos)
    u = busca_custo_uniforme(MAPA, origem, objetivos)
    assert a.nos_expandidos <= u.nos_expandidos


def test_heuristica_e_admissivel():
    # h(n) nunca passa do custo real até o objetivo mais próximo
    for material in RECICLAVEIS:
        objetivos = _objetivos(material)
        h = heuristica(MAPA, objetivos)
        for bairro in MAPA.bairros:
            assert h(bairro) <= busca_custo_uniforme(MAPA, bairro, objetivos).custo_km + 1e-9


def test_heuristica_e_consistente():
    for material in RECICLAVEIS:
        h = heuristica(MAPA, _objetivos(material))
        for a, vizinhos in MAPA.vizinhos.items():
            for b, custo in vizinhos:
                assert h(a) <= custo + h(b) + 1e-9


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_caminho_valido_do_inicio_ao_objetivo(algoritmo):
    objetivos = _objetivos("vidro")
    r = ALGORITMOS[algoritmo](MAPA, "Benedito Bentes", objetivos)
    assert r.caminho[0] == "Benedito Bentes"
    assert r.caminho[-1] in objetivos
    for a, b in zip(r.caminho, r.caminho[1:]):
        assert b in {v for v, _ in MAPA.vizinhos[a]}, f"{a} e {b} não são vizinhos"


def test_bfs_minimiza_numero_de_ruas():
    for origem, material in CONSULTAS:
        objetivos = _objetivos(material)
        bfs = busca_largura(MAPA, origem, objetivos)
        ucs = busca_custo_uniforme(MAPA, origem, objetivos)
        assert len(bfs.caminho) <= len(ucs.caminho)


# --- buscar_ponto_coleta e integração com a decisão ------------------------

def test_origem_que_ja_tem_ponto():
    rota = buscar_ponto_coleta("Jatiúca", ["plastico"])
    assert rota.caminho == ["Jatiúca"] and rota.distancia_km == 0


def test_varios_materiais_exigem_ponto_que_aceite_todos():
    rota = buscar_ponto_coleta("Centro", ["papel", "papelao"])
    ponto = next(p for p in MAPA.pontos_coleta if p.nome == rota.ponto)
    assert {"papel", "papelao"} <= ponto.aceita


def test_rejeito_nao_tem_ponto():
    assert buscar_ponto_coleta("Centro", ["lixo"]) is None


def test_bairro_desconhecido():
    with pytest.raises(ValueError):
        buscar_ponto_coleta("Atlântida", ["vidro"])


def _decidir_ponto(classe, origem):
    visao = ResultadoVisao(classe=classe, confianca=0.9, top3=[(classe, 0.9)])
    pln = ResultadoPLN(intencao="ponto_coleta", confianca=0.9, texto_original="x")
    return decidir(visao, pln, origem)


def test_decisao_ponto_coleta_com_bairro_traz_rota():
    decisao = _decidir_ponto("vidro", "Benedito Bentes")
    assert decisao.rota is not None
    assert decisao.rota.ponto in decisao.resposta
    assert decisao.lixeira.startswith("verde")


def test_decisao_ponto_coleta_sem_bairro_pede_bairro():
    decisao = _decidir_ponto("vidro", None)
    assert decisao.rota is None and "Seu bairro" in decisao.resposta


def test_decisao_ponto_coleta_rejeito():
    decisao = _decidir_ponto("lixo", "Centro")
    assert decisao.rota is None and "lixo comum" in decisao.resposta
