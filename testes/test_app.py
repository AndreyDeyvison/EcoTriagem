from PIL import Image

import app


def test_foto_inicia_conversa_com_material_identificado():
    visao, conversa, material, etapas = app.ao_receber_imagem(Image.new("RGB", (64, 64)))
    assert visao is not None
    assert len(conversa) == 1 and conversa[0]["role"] == "assistant"
    assert material
    assert "Visão Computacional" in etapas


def test_remover_foto_limpa_a_tela():
    visao, conversa, material, etapas = app.ao_receber_imagem(None)
    assert visao is None and conversa == [] and material is None
    assert etapas == app.ETAPAS_VAZIAS


def test_pergunta_sem_foto_pede_a_foto():
    _, conversa, _ = app.ao_perguntar("isso é reciclável?", [], None)
    assert conversa[-1]["content"] == "Envie uma foto do resíduo primeiro."


def test_conversa_acumula_perguntas_e_respostas():
    visao, conversa, _, _ = app.ao_receber_imagem(Image.new("RGB", (64, 64)))
    caixa, conversa, etapas = app.ao_perguntar("isso é reciclável?", conversa, visao)
    caixa, conversa, etapas = app.ao_perguntar("qual a cor da lixeira?", conversa, visao)
    assert caixa == ""
    assert [m["role"] for m in conversa] == ["assistant", "user", "assistant", "user", "assistant"]
    assert "cor_lixeira" in etapas


def test_pergunta_vazia_e_ignorada():
    _, conversa, _ = app.ao_perguntar("   ", [], None)
    assert conversa == []


def test_registrar_salva_cada_pergunta(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "registrar_interacoes", True)
    monkeypatch.setattr(app, "PASTA_REGISTRO", tmp_path)
    foto = Image.new("RGB", (64, 64))
    visao, conversa, _, _ = app.ao_receber_imagem(foto)
    app.ao_perguntar("isso é reciclável?", conversa, visao, foto)
    assert (tmp_path / "exemplo_001.jpg").exists()
    assert (tmp_path / "exemplos.jsonl").exists()
