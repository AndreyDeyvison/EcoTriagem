"""Interface web do EcoTriagem (Gradio).

Executar a partir da raiz do repositório:  python app.py
"""

import gradio as gr

from src.decisao.regras import NOMES
from src.integracao.pipeline import processar


def responder(imagem, pergunta):
    if imagem is None:
        return "Envie uma foto do resíduo.", None, None
    if not pergunta or not pergunta.strip():
        return "Escreva uma pergunta sobre o resíduo.", None, None

    resultado = processar(imagem, pergunta)
    probabilidades = {NOMES[c]: p for c, p in resultado.visao.top3}
    detalhes = {
        "visao": {"classe": resultado.visao.classe, "confianca": resultado.visao.confianca},
        "pln": {"intencao": resultado.pln.intencao, "confianca": resultado.pln.confianca},
        "decisao": {"acao": resultado.decisao.acao, "lixeira": resultado.decisao.lixeira},
    }
    return resultado.decisao.resposta, probabilidades, detalhes


with gr.Blocks(title="EcoTriagem") as demo:
    gr.Markdown("# EcoTriagem\nEnvie a foto de um resíduo e pergunte como descartá-lo.")
    with gr.Row():
        with gr.Column():
            imagem = gr.Image(type="pil", label="Foto do resíduo")
            pergunta = gr.Textbox(label="Pergunta", placeholder="Ex.: onde descarto isso?")
            enviar = gr.Button("Enviar", variant="primary")
        with gr.Column():
            resposta = gr.Textbox(label="Resposta", interactive=False)
            classes = gr.Label(label="Material identificado", num_top_classes=3)
            detalhes = gr.JSON(label="Detalhes de cada etapa")

    enviar.click(responder, inputs=[imagem, pergunta], outputs=[resposta, classes, detalhes])
    pergunta.submit(responder, inputs=[imagem, pergunta], outputs=[resposta, classes, detalhes])


if __name__ == "__main__":
    demo.launch()
