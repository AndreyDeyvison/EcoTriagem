"""Testes do módulo de PLN (dataset, pré-processamento, modelo e integração).

Os testes que dependem do modelo treinado são pulados quando
`modelos/pln_intencoes.joblib` não existe — quem clona o repositório não
recebe a pasta `modelos/`, e isso não deve quebrar a suíte.
"""

import csv
from collections import Counter

import pytest

from src.configuracao import CLASSES, INTENCOES, LIMIAR_CONFIANCA_PLN
from src.integracao.contratos import ResultadoPLN
from src.pln import baseline, dados, modelos, objetos
from src.pln.intencoes import aceita, interpretar_pergunta
from src.pln.preprocessamento import (
    NEGACOES,
    STOPWORDS,
    normalizar,
    normalizar_sem_stopwords,
    remover_stopwords,
    tem_conteudo,
    tokenizar,
)

MINIMO_POR_INTENCAO = 40

sem_modelo = pytest.mark.skipif(
    not modelos.ARQUIVO_MODELO.exists(),
    reason="modelo não treinado; rode python -m src.pln.treinar",
)


# --------------------------------------------------------------- dataset


def _linhas_do_dataset() -> list[dict[str, str]]:
    with dados.ARQUIVO.open(encoding="utf-8", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def test_dataset_existe_e_carrega():
    textos, intencoes = dados.carregar()
    assert len(textos) == len(intencoes) > 0


def test_dataset_cobre_todas_as_intencoes_com_frases_suficientes():
    _, intencoes = dados.carregar()
    contagem = Counter(intencoes)
    assert set(contagem) == set(INTENCOES)
    poucas = {i: n for i, n in contagem.items() if n < MINIMO_POR_INTENCAO}
    assert not poucas, f"intenções com menos de {MINIMO_POR_INTENCAO} frases: {poucas}"


def test_dataset_nao_tem_frases_repetidas():
    textos, _ = dados.carregar()
    repetidas = [t for t, n in Counter(t.strip().lower() for t in textos).items() if n > 1]
    assert not repetidas, f"frases repetidas: {repetidas}"


def test_dataset_nao_tem_frase_com_espaco_sobrando():
    assert [l["texto"] for l in _linhas_do_dataset() if l["texto"] != l["texto"].strip()] == []


def test_dataset_rejeita_intencao_desconhecida(tmp_path):
    arquivo = tmp_path / "ruim.csv"
    arquivo.write_text("texto,intencao\noi,saudacao\n", encoding="utf-8")
    with pytest.raises(ValueError, match="intenção desconhecida"):
        dados.carregar(arquivo)


def test_divisao_e_estratificada_e_sem_vazamento():
    treino_x, treino_y, teste_x, teste_y, _ = dados.dividir()
    assert set(treino_y) == set(teste_y) == set(INTENCOES)
    if not dados.ARQUIVO_TESTE.exists():
        assert not set(treino_x) & set(teste_x), "a mesma frase está no treino e no teste"


def test_divisao_e_reproduzivel():
    assert dados.dividir()[2] == dados.dividir()[2]


# -------------------------------------------------- pré-processamento


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("Isso é RECICLÁVEL???", "isso e reciclavel"),
        ("Onde  jogo   isso?", "onde jogo isso"),
        ("Não!", "nao"),
        ("ooooi", "ooi"),
    ],
)
def test_normalizar(entrada, esperado):
    assert normalizar(entrada) == esperado


def test_negacoes_nao_sao_stopwords():
    assert not NEGACOES & STOPWORDS
    assert "nao" in remover_stopwords(tokenizar("isso não é reciclável?"))
    assert "nao" in normalizar_sem_stopwords("isso não é reciclável?").split()


def test_tem_conteudo_separa_texto_degenerado():
    assert tem_conteudo("isso recicla?")
    assert not tem_conteudo("?????")
    assert not tem_conteudo("   ")


# ------------------------------------------------------------- baseline


@pytest.mark.parametrize("texto", ["onde jogo isso?", "isso é reciclável?", "oi", "asdasd"])
def test_baseline_devolve_intencao_valida(texto):
    intencao, confianca = baseline.classificar(texto)
    assert intencao in INTENCOES
    assert 0.0 <= confianca <= 1.0


def test_baseline_e_pior_que_o_modelo_no_teste():
    """Garante que o classificador treinado ganha do casamento de palavras-chave.

    É a comparação que justifica trocar o stub; se o modelo regredir até esse
    ponto, o teste falha.
    """
    if not modelos.ARQUIVO_MODELO.exists():
        pytest.skip("modelo não treinado")
    from sklearn.metrics import f1_score

    _, _, teste_x, teste_y, _ = dados.dividir()
    classificador = modelos.carregar()
    f1_modelo = f1_score(teste_y, classificador.prever(teste_x), average="macro", zero_division=0)
    f1_baseline = f1_score(teste_y, baseline.prever(teste_x), average="macro", zero_division=0)
    assert f1_modelo > f1_baseline


# ---------------------------------------------------- objetos citados


@pytest.mark.parametrize(
    ("texto", "objeto", "classe"),
    [
        ("onde jogo essa latinha?", "latinha", "metal"),
        ("onde jogo essa garrafa de vidro?", "garrafa de vidro", "vidro"),
        ("e esse jornal velho?", "jornal", "papel"),
        ("onde descarto essa pilha?", "pilha", None),
        ("isso é reciclável?", None, None),
    ],
)
def test_extrair_objeto(texto, objeto, classe):
    encontrado = objetos.extrair_objeto(texto)
    if objeto is None:
        assert encontrado is None
        return
    assert encontrado is not None
    assert encontrado.objeto.lower().startswith(objeto.split()[0])
    assert encontrado.classe_sugerida == classe


def test_termo_composto_vence_o_simples():
    assert objetos.extrair_objeto("essa garrafa de vidro").classe_sugerida == "vidro"
    assert objetos.extrair_objeto("essa garrafa").classe_sugerida is None


def test_objeto_especial_tem_motivo():
    pilha = objetos.extrair_objeto("onde entrego essa pilha?")
    assert pilha.especial and pilha.motivo


def test_objeto_mais_informativo_vence():
    citado = objetos.extrair_objeto("a pilha dessa lanterna de plastico")
    assert citado.objeto == "pilha"


def test_lexico_so_usa_classes_conhecidas():
    for termo, (_, classe, _) in objetos.LEXICO.items():
        assert classe is None or classe in CLASSES, termo


# --------------------------------------------------------- integração


@pytest.mark.parametrize(
    "texto",
    ["onde jogo isso?", "isso eh recilavel?", "oi", "?????", "", "   ", "qwerty 123"],
)
def test_interpretar_pergunta_respeita_o_contrato(texto):
    resultado = interpretar_pergunta(texto)
    assert isinstance(resultado, ResultadoPLN)
    assert resultado.intencao in INTENCOES
    assert 0.0 <= resultado.confianca <= 1.0
    assert resultado.texto_original == texto


def test_texto_vazio_fica_fora_do_escopo():
    assert interpretar_pergunta("?????").intencao == "fora_do_escopo"


def test_aceita_usa_o_limiar_do_sistema():
    assert not aceita(ResultadoPLN("como_descartar", LIMIAR_CONFIANCA_PLN - 0.01, "x"))
    assert aceita(ResultadoPLN("como_descartar", LIMIAR_CONFIANCA_PLN, "x"))
    assert not aceita(ResultadoPLN("fora_do_escopo", 1.0, "oi"))


@sem_modelo
@pytest.mark.parametrize(
    ("texto", "intencao"),
    [
        ("onde devo jogar isso?", "como_descartar"),
        ("isso é reciclável?", "e_reciclavel"),
        ("qual a cor da lixeira?", "cor_lixeira"),
        ("preciso lavar antes?", "como_preparar"),
        ("onde fica o ecoponto mais próximo?", "ponto_coleta"),
        ("o que é isso?", "identificar"),
        ("bom dia", "fora_do_escopo"),
    ],
)
def test_modelo_acerta_as_perguntas_tipicas(texto, intencao):
    resultado = interpretar_pergunta(texto)
    assert resultado.intencao == intencao
    assert resultado.confianca >= LIMIAR_CONFIANCA_PLN


@sem_modelo
def test_modelo_e_robusto_a_erro_de_digitacao_e_falta_de_acento():
    for texto in ("isso eh recilavel?", "isso e reciclavel", "ISSO É RECICLÁVEL?"):
        assert interpretar_pergunta(texto).intencao == "e_reciclavel"


@sem_modelo
def test_prever_proba_tem_forma_e_soma_corretas():
    classificador = modelos.carregar()
    textos = ["onde jogo isso?", "?????", "oi"]
    probabilidades = classificador.prever_proba(textos)
    assert probabilidades.shape == (len(textos), len(INTENCOES))
    assert probabilidades.sum(axis=1).round(6).tolist() == [1.0] * len(textos)


@sem_modelo
def test_modelo_carrega_uma_unica_vez():
    from src.pln import intencoes

    intencoes.recarregar()
    interpretar_pergunta("onde jogo isso?")
    primeiro = intencoes._obter_classificador()
    interpretar_pergunta("isso recicla?")
    assert intencoes._obter_classificador() is primeiro
