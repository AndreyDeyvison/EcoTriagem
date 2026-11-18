import json
from collections import Counter

import pytest
from PIL import Image

from src.configuracao import INTENCOES
from src.integracao.avaliar_sistema import carregar_perguntas, classificar_desfecho
from src.integracao.contratos import Decisao, ResultadoPipeline, ResultadoPLN, ResultadoVisao
from src.integracao.registro import ARQUIVO_REGISTRO, salvar_exemplo


def _resultado(classe_prevista="vidro", intencao="como_descartar", acao="responder", lixeira="verde (vidro)"):
    return ResultadoPipeline(
        visao=ResultadoVisao(classe=classe_prevista, confianca=0.9),
        pln=ResultadoPLN(intencao=intencao, confianca=0.9, texto_original="x"),
        decisao=Decisao(resposta="...", acao=acao, lixeira=lixeira),
    )


@pytest.mark.parametrize(
    ("classe_real", "intencao_esperada", "resultado", "esperado"),
    [
        ("vidro", "como_descartar", _resultado(), "correto"),
        # papel e papelão vão na mesma lixeira: para o usuário, a resposta está certa
        ("papelao", "como_descartar", _resultado("papel", lixeira="azul (papel)"), "correto"),
        ("plastico", "como_descartar", _resultado(), "erro_visao"),
        ("vidro", "como_descartar", _resultado(intencao="e_reciclavel"), "erro_pln"),
        ("vidro", "como_descartar", _resultado(acao="reformular_pergunta", lixeira=None), "erro_pln"),
        ("vidro", "como_descartar", _resultado(acao="pedir_nova_foto", lixeira=None), "nova_foto"),
        ("vidro", "como_descartar", _resultado(lixeira="cinza (rejeito)"), "erro_decisao"),
        ("vidro", "fora_do_escopo", _resultado(intencao="fora_do_escopo", acao="reformular_pergunta", lixeira=None), "correto"),
        ("vidro", "fora_do_escopo", _resultado(intencao="identificar"), "erro_pln"),
    ],
)
def test_classificar_desfecho(classe_real, intencao_esperada, resultado, esperado):
    assert classificar_desfecho(classe_real, intencao_esperada, resultado) == esperado


def test_perguntas_de_teste_cobrem_todas_as_intencoes():
    perguntas = carregar_perguntas()
    contagem = Counter(intencao for _, intencao in perguntas)
    assert set(contagem) == set(INTENCOES)
    assert min(contagem.values()) >= 5
    assert len({texto for texto, _ in perguntas}) == len(perguntas), "pergunta repetida"


def test_salvar_exemplo_grava_imagem_e_registro(tmp_path):
    salvar_exemplo(tmp_path, Image.new("RGBA", (20, 20)), _resultado(), extras={"desfecho": "correto"})
    salvar_exemplo(tmp_path, Image.new("RGB", (20, 20)), _resultado())

    assert sorted(p.name for p in tmp_path.glob("*.jpg")) == ["exemplo_001.jpg", "exemplo_002.jpg"]
    linhas = (tmp_path / ARQUIVO_REGISTRO).read_text(encoding="utf-8").splitlines()
    primeiro = json.loads(linhas[0])
    assert len(linhas) == 2
    assert primeiro["imagem"] == "exemplo_001.jpg"
    assert primeiro["decisao"]["lixeira"] == "verde (vidro)"
    assert primeiro["desfecho"] == "correto"


def test_perguntas_de_avaliacao_nao_aparecem_no_dataset_do_pln():
    """Se uma pergunta da avaliação estiver no treino do PLN, a avaliação mede memorização."""
    import csv
    import re
    import unicodedata

    from src.configuracao import RAIZ

    def palavras(texto):
        sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
        return frozenset(re.findall(r"\w+", sem_acento))

    arquivo_pln = RAIZ / "dados" / "pln" / "intencoes.csv"
    if not arquivo_pln.exists():
        pytest.skip("dataset do PLN ausente")
    with arquivo_pln.open(encoding="utf-8") as f:
        treino = {palavras(linha["texto"]) for linha in csv.DictReader(f)}
    repetidas = [texto for texto, _ in carregar_perguntas() if palavras(texto) in treino]
    assert not repetidas, f"perguntas da avaliação presentes no dataset do PLN: {repetidas}"
