"""Sistema baseado em regras que combina Visão e PLN numa resposta.

VERSÃO INICIAL: cobre as regras básicas para o fluxo funcionar.
Responsável: Agente de decisão & busca, que deve expandir as regras e
implementar a busca do ponto de coleta (ex.: A* num mapa simulado).

Cores das lixeiras segundo a Resolução CONAMA nº 275/2001.
"""

from src.configuracao import LIMIAR_CONFIANCA_PLN, LIMIAR_CONFIANCA_VISAO
from src.integracao.contratos import Decisao, ResultadoPLN, ResultadoVisao

NOMES = {
    "papel": "papel",
    "papelao": "papelão",
    "plastico": "plástico",
    "vidro": "vidro",
    "metal": "metal",
    "lixo": "rejeito (não reciclável)",
}

LIXEIRAS = {
    "papel": "azul",
    "papelao": "azul",
    "plastico": "vermelha",
    "vidro": "verde",
    "metal": "amarela",
    "lixo": "cinza",
}

RECICLAVEL = {
    "papel": True,
    "papelao": True,
    "plastico": True,
    "vidro": True,
    "metal": True,
    "lixo": False,
}

PREPARO = {
    "papel": "Mantenha seco e sem restos de comida; papel engordurado ou molhado vai para o rejeito.",
    "papelao": "Desmonte as caixas para ocupar menos espaço e mantenha o material seco.",
    "plastico": "Esvazie e dê uma enxaguada rápida para tirar restos; amassar ajuda a reduzir o volume.",
    "vidro": "Esvazie e enxágue; vidro quebrado deve ir embrulhado em papel para proteger os coletores.",
    "metal": "Esvazie e enxágue latas; amassar latinhas reduz o volume.",
    "lixo": "Não há preparo: descarte no lixo comum, bem fechado.",
}


def _rotulo_lixeira(classe: str) -> str:
    return f"{LIXEIRAS[classe]} ({NOMES[classe]})"


def decidir(visao: ResultadoVisao, pln: ResultadoPLN) -> Decisao:
    """Combina o que foi visto na imagem com o que foi perguntado."""
    if pln.intencao == "fora_do_escopo" or pln.confianca < LIMIAR_CONFIANCA_PLN:
        return Decisao(
            resposta=(
                "Não entendi a pergunta. Você pode perguntar, por exemplo: "
                "\"onde descarto isso?\", \"isso é reciclável?\" ou \"preciso lavar antes?\""
            ),
            acao="reformular_pergunta",
        )

    if visao.confianca < LIMIAR_CONFIANCA_VISAO:
        palpites = ", ".join(f"{NOMES[c]} ({p:.0%})" for c, p in visao.top3) or NOMES[visao.classe]
        return Decisao(
            resposta=(
                f"Não tenho certeza sobre o material da foto (palpites: {palpites}). "
                "Tente outra foto, com o objeto sozinho, bem iluminado e com fundo liso."
            ),
            acao="pedir_nova_foto",
        )

    classe = visao.classe
    nome = NOMES[classe]
    lixeira = _rotulo_lixeira(classe)

    match pln.intencao:
        case "identificar":
            resposta = f"Parece ser {nome} ({visao.confianca:.0%} de confiança)."
        case "como_descartar":
            resposta = f"Isso parece ser {nome}: descarte na lixeira {lixeira}."
        case "e_reciclavel":
            if RECICLAVEL[classe]:
                resposta = f"Sim, {nome} é reciclável. Use a lixeira {lixeira}."
            else:
                resposta = f"Não, isso parece ser {nome}. Descarte na lixeira {lixeira}."
        case "cor_lixeira":
            resposta = f"A lixeira para {nome} é a {LIXEIRAS[classe]}."
        case "como_preparar":
            resposta = PREPARO[classe]
        case "ponto_coleta":
            # TODO(decisão & busca): buscar o ponto de coleta mais próximo (ex.: A*).
            resposta = f"A busca do ponto de coleta ainda não foi implementada. Por enquanto, use a lixeira {lixeira}."
        case _:
            raise ValueError(f"intenção sem regra: {pln.intencao!r}")

    return Decisao(resposta=resposta, acao="responder", lixeira=lixeira)
