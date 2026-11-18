"""Vocabulário comum do EcoTriagem: classes, intenções e limiares.

Todos os módulos importam daqui. Assim ninguém usa nomes diferentes
para a mesma coisa (ex.: "plastic" num módulo e "plastico" em outro).
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Modelos treinados ficam aqui (versionados; regeneráveis pelos scripts de treino).
PASTA_MODELOS = RAIZ / "modelos"

# Classes de resíduo reconhecidas pela Visão Computacional (base: TrashNet).
CLASSES = ("papel", "papelao", "plastico", "vidro", "metal", "lixo")

# Nome das pastas do TrashNet -> nome usado no projeto.
CLASSES_TRASHNET = {
    "paper": "papel",
    "cardboard": "papelao",
    "plastic": "plastico",
    "glass": "vidro",
    "metal": "metal",
    "trash": "lixo",
}

# Intenções reconhecidas pelo PLN.
INTENCOES = (
    "identificar",      # "o que é isso?", "que material é esse?"
    "como_descartar",   # "onde jogo isso?", "como descarto?"
    "e_reciclavel",     # "isso é reciclável?"
    "cor_lixeira",      # "qual a cor da lixeira?"
    "como_preparar",    # "preciso lavar antes?"
    "ponto_coleta",     # "onde fica o ecoponto mais perto?"
    "fora_do_escopo",   # qualquer coisa que não seja sobre descarte
)

# Ações possíveis do módulo de decisão.
ACOES = ("responder", "pedir_nova_foto", "reformular_pergunta")

# Abaixo destes valores o sistema não confia na previsão.
LIMIAR_CONFIANCA_VISAO = 0.6
LIMIAR_CONFIANCA_PLN = 0.5
