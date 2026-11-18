"""Interface web do EcoTriagem (Gradio).

Executar a partir da raiz do repositório:
    python app.py               # interface normal
    python app.py --registrar   # também salva cada pergunta/resposta em resultados/exemplos/interface/

Fluxo: o usuário envia uma foto -> a Visão analisa uma única vez -> o
usuário conversa sobre ela -> cada pergunta passa pelo PLN e pela Decisão,
reaproveitando a análise da foto.
"""

import argparse

import gradio as gr

from src.configuracao import LIMIAR_CONFIANCA_VISAO, RAIZ
from src.decisao.busca import NOMES_ALGORITMOS
from src.decisao.mapa import carregar_mapa, desenhar_mapa
from src.decisao.regras import NOMES
from src.integracao.contratos import ResultadoPipeline, ResultadoVisao
from src.integracao.pipeline import analisar_imagem, carregar_modelos, responder_pergunta
from src.integracao.registro import salvar_exemplo

STATUS_MODELOS = carregar_modelos()

PASTA_REGISTRO = RAIZ / "resultados" / "exemplos" / "interface"
registrar_interacoes = False  # ligado por --registrar

EXEMPLOS_IMAGEM = sorted(str(p) for p in (RAIZ / "dados" / "exemplos").glob("*.jpg"))
EXEMPLOS_PERGUNTA = [
    "O que é isso?",
    "Onde descarto isso?",
    "Isso é reciclável?",
    "Qual a cor da lixeira?",
    "Preciso lavar antes?",
    "Onde fica o ecoponto mais perto?",
]

ETAPAS_VAZIAS = "_Envie uma foto e faça uma pergunta para ver cada etapa do processamento._"


def _aviso_modo_demonstracao() -> str:
    provisorios = [nome for nome, real in STATUS_MODELOS.items() if not real]
    if not provisorios:
        return ""
    modulos = " e ".join(provisorios)
    verbo = "ainda não tem" if len(provisorios) == 1 else "ainda não têm"
    return f"**Modo de demonstração:** {modulos} {verbo} modelo treinado; as respostas usam uma versão provisória."


def _probabilidades(visao: ResultadoVisao) -> dict[str, float]:
    return {NOMES[classe]: prob for classe, prob in visao.top3}


def _tabela_etapas(visao: ResultadoVisao, resultado: ResultadoPipeline | None = None) -> str:
    linhas = [
        "| Etapa | Saída | Confiança |",
        "|---|---|---|",
        f"| 1. Visão Computacional | {NOMES[visao.classe]} | {visao.confianca:.0%} |",
    ]
    if resultado is not None:
        decisao = resultado.decisao
        saida_decisao = f"`{decisao.acao}` pela regra **{decisao.regra}**" + (
            f" → lixeira {decisao.lixeira}" if decisao.lixeira else ""
        )
        linhas += [
            f"| 2. PLN (intenção) | `{resultado.pln.intencao}` | {resultado.pln.confianca:.0%} |",
            f"| 3. Decisão | {saida_decisao} | — |",
        ]
        rota = decisao.rota
        if rota is not None:
            linhas.append(
                f"| 4. Busca {NOMES_ALGORITMOS[rota.algoritmo]} | {' → '.join(rota.caminho)} "
                f"({rota.distancia_km:.1f} km) | {rota.nos_expandidos} nós expandidos |"
            )
    return "\n".join(linhas)


def ao_receber_imagem(imagem):
    """Analisa a foto uma única vez e reinicia a conversa."""
    if imagem is None:
        return None, [], None, ETAPAS_VAZIAS

    visao = analisar_imagem(imagem)
    nome = NOMES[visao.classe]
    if visao.confianca >= LIMIAR_CONFIANCA_VISAO:
        mensagem = f"Identifiquei **{nome}** ({visao.confianca:.0%} de confiança). O que você quer saber sobre ele?"
    else:
        mensagem = (
            f"Não tenho certeza do material; o mais provável é **{nome}** ({visao.confianca:.0%}). "
            "Se puder, envie outra foto com o objeto sozinho e bem iluminado."
        )
    conversa = [{"role": "assistant", "content": mensagem}]
    return visao, conversa, _probabilidades(visao), _tabela_etapas(visao)


def ao_perguntar(pergunta, conversa, visao, imagem=None, bairro=None):
    """Responde a uma pergunta sobre a foto já analisada.

    Devolve (caixa de texto, conversa, painel de etapas, mapa da rota).
    """
    pergunta = (pergunta or "").strip()
    if not pergunta:
        return "", conversa, gr.skip(), gr.skip()

    conversa = conversa + [{"role": "user", "content": pergunta}]
    if visao is None:
        conversa.append({"role": "assistant", "content": "Envie uma foto do resíduo primeiro."})
        return "", conversa, gr.skip(), gr.skip()

    resultado = responder_pergunta(visao, pergunta, origem=bairro or None)
    conversa.append({"role": "assistant", "content": resultado.decisao.resposta})
    if registrar_interacoes and imagem is not None:
        salvar_exemplo(PASTA_REGISTRO, imagem, resultado, extras={"bairro": bairro})

    rota = resultado.decisao.rota
    if rota is None:
        mapa = gr.Plot(visible=False)
    else:
        mapa = gr.Plot(value=desenhar_mapa(rota, f"Rota até {rota.ponto}"), visible=True)
    return "", conversa, _tabela_etapas(visao, resultado), mapa


with gr.Blocks(title="EcoTriagem") as demo:
    visao_atual = gr.State(None)

    gr.Markdown(
        "# EcoTriagem\n"
        "Envie a foto de um resíduo e pergunte como descartá-lo. "
        "O sistema identifica o material pela imagem, interpreta sua pergunta e recomenda o descarte "
        "segundo as cores da coleta seletiva (CONAMA 275/2001)."
    )
    aviso = _aviso_modo_demonstracao()
    if aviso:
        gr.Markdown(aviso)

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            imagem = gr.Image(
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                label="Foto do resíduo",
                height=320,
            )
            if EXEMPLOS_IMAGEM:
                gr.Examples(EXEMPLOS_IMAGEM, inputs=imagem, label="Fotos de exemplo (conjunto de teste)")
            material = gr.Label(label="Material identificado", num_top_classes=3)
            bairro = gr.Dropdown(
                choices=carregar_mapa().bairros,
                value=None,
                label="Seu bairro (mapa simulado de Maceió)",
                info="Usado quando você pergunta pelo ponto de coleta mais próximo.",
            )

        with gr.Column(scale=3):
            conversa = gr.Chatbot(label="Conversa", height=360, placeholder="Envie uma foto para começar.")
            with gr.Row():
                pergunta = gr.Textbox(
                    show_label=False,
                    placeholder="Pergunte algo sobre o resíduo, ex.: onde descarto isso?",
                    scale=5,
                )
                enviar = gr.Button("Enviar", variant="primary", scale=1, min_width=90)
            gr.Examples(EXEMPLOS_PERGUNTA, inputs=pergunta, label="Perguntas de exemplo")
            with gr.Accordion("Como o sistema chegou à resposta", open=True):
                etapas = gr.Markdown(ETAPAS_VAZIAS)
            mapa_rota = gr.Plot(label="Rota até o ponto de coleta (mapa simulado, pontos fictícios)", visible=False)

    imagem.change(ao_receber_imagem, inputs=imagem, outputs=[visao_atual, conversa, material, etapas])
    for evento in (enviar.click, pergunta.submit):
        evento(
            ao_perguntar,
            inputs=[pergunta, conversa, visao_atual, imagem, bairro],
            outputs=[pergunta, conversa, etapas, mapa_rota],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interface web do EcoTriagem")
    parser.add_argument(
        "--registrar",
        action="store_true",
        help=f"salva cada pergunta e resposta em {PASTA_REGISTRO.relative_to(RAIZ)}/",
    )
    registrar_interacoes = parser.parse_args().registrar
    if registrar_interacoes:
        print(f"Registrando interações em {PASTA_REGISTRO.relative_to(RAIZ)}/")
    demo.launch(theme=gr.themes.Soft(primary_hue="green"))
