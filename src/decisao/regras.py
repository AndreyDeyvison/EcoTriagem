"""Agente de decisão baseado em regras: combina Visão e PLN numa resposta.

Responsável: Agente de decisão & busca (Caio).
Documentação das regras (para o artigo): src/decisao/REGRAS.md

O agente percebe (ResultadoVisao, ResultadoPLN) e escolhe uma ação aplicando
as regras abaixo na ordem; a primeira que se aplica vence.

  R1  pergunta fora do escopo ou PLN pouco confiante  -> reformular_pergunta
  R2  visão confiante no material                     -> responder
  R3  visão insegura no material, mas confiante na
      lixeira (soma dos palpites da mesma lixeira)    -> responder
  R4  visão insegura também na lixeira                -> pedir_nova_foto

Cores das lixeiras segundo a Resolução CONAMA nº 275/2001.
"""

from collections import defaultdict

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

REGRAS = {
    "R1": "Pergunta fora do escopo ou PLN pouco confiante: pedir para reformular",
    "R2": "Visão confiante no material: responder sobre o material identificado",
    "R3": "Visão insegura no material, mas confiante na lixeira: responder sobre a lixeira",
    "R4": "Visão insegura também na lixeira: pedir nova foto",
}


def probabilidade_por_lixeira(visao: ResultadoVisao) -> dict[str, float]:
    """P(lixeira) = soma das probabilidades dos materiais que vão nela.

    Usa o top 3 da visão; a probabilidade fora do top 3 é ignorada, o que só
    pode subestimar a lixeira (decisão conservadora).
    """
    palpites = visao.top3 or [(visao.classe, visao.confianca)]
    soma: dict[str, float] = defaultdict(float)
    for classe, prob in palpites:
        soma[LIXEIRAS[classe]] += prob
    return dict(soma)


def _materiais_candidatos(visao: ResultadoVisao) -> tuple[str, list[str], float] | None:
    """Aplica R2/R3. Devolve (regra, materiais, confiança) ou None se nenhuma se aplica."""
    if visao.confianca >= LIMIAR_CONFIANCA_VISAO:
        return "R2", [visao.classe], visao.confianca

    por_lixeira = probabilidade_por_lixeira(visao)
    cor, prob = max(por_lixeira.items(), key=lambda item: item[1])
    if prob >= LIMIAR_CONFIANCA_VISAO:
        palpites = visao.top3 or [(visao.classe, visao.confianca)]
        materiais = [c for c, _ in palpites if LIXEIRAS[c] == cor]
        return "R3", materiais, prob
    return None


def _juntar(nomes: list[str]) -> str:
    return nomes[0] if len(nomes) == 1 else ", ".join(nomes[:-1]) + " ou " + nomes[-1]


def _responder(intencao: str, materiais: list[str], confianca: float) -> str:
    """Monta o texto da resposta para a intenção, dado(s) o(s) material(is) candidato(s)."""
    nome = _juntar([NOMES[m] for m in materiais])
    cor = LIXEIRAS[materiais[0]]
    rotulo = f"{cor} ({nome})"
    ambiguo = len(materiais) > 1

    match intencao:
        case "identificar":
            if ambiguo:
                return (
                    f"Parece ser {nome}; não consigo distinguir com certeza, mas os dois "
                    f"vão na mesma lixeira ({confianca:.0%} de confiança na lixeira {cor})."
                )
            return f"Parece ser {nome} ({confianca:.0%} de confiança)."
        case "como_descartar":
            return f"Isso parece ser {nome}: descarte na lixeira {rotulo}."
        case "e_reciclavel":
            if all(RECICLAVEL[m] for m in materiais):
                return f"Sim, {nome} é reciclável. Use a lixeira {rotulo}."
            return f"Não, isso parece ser {nome}. Descarte na lixeira {rotulo}."
        case "cor_lixeira":
            return f"A lixeira para {nome} é a {cor}."
        case "como_preparar":
            if ambiguo:
                return " ".join(f"Se for {NOMES[m]}: {PREPARO[m]}" for m in materiais)
            return PREPARO[materiais[0]]
        case "ponto_coleta":
            # TODO(Caio): buscar o ponto de coleta mais próximo (A*), ver PLANO_DE_TRABALHO.md.
            return f"A busca do ponto de coleta ainda não foi implementada. Por enquanto, use a lixeira {rotulo}."
        case _:
            raise ValueError(f"intenção sem regra: {intencao!r}")


def decidir(visao: ResultadoVisao, pln: ResultadoPLN) -> Decisao:
    """Combina o que foi visto na imagem com o que foi perguntado."""
    # R1
    if pln.intencao == "fora_do_escopo" or pln.confianca < LIMIAR_CONFIANCA_PLN:
        return Decisao(
            resposta=(
                "Não entendi a pergunta. Você pode perguntar, por exemplo: "
                "\"onde descarto isso?\", \"isso é reciclável?\" ou \"preciso lavar antes?\""
            ),
            acao="reformular_pergunta",
            regra="R1",
        )

    # R2 / R3
    candidatos = _materiais_candidatos(visao)
    if candidatos is not None:
        regra, materiais, confianca = candidatos
        nome = _juntar([NOMES[m] for m in materiais])
        return Decisao(
            resposta=_responder(pln.intencao, materiais, confianca),
            acao="responder",
            lixeira=f"{LIXEIRAS[materiais[0]]} ({nome})",
            regra=regra,
        )

    # R4
    palpites = ", ".join(f"{NOMES[c]} ({p:.0%})" for c, p in visao.top3) or NOMES[visao.classe]
    return Decisao(
        resposta=(
            f"Não tenho certeza sobre o material da foto (palpites: {palpites}). "
            "Tente outra foto, com o objeto sozinho, bem iluminado e com fundo liso."
        ),
        acao="pedir_nova_foto",
        regra="R4",
    )
