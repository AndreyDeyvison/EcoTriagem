"""Reconhecimento da intenção da pergunta do usuário.

Classificador treinado (TF-IDF + modelo linear) sobre o dataset de frases em
`dados/pln/intencoes.csv`. Treino e comparação de modelos em
`src.pln.treinar`; avaliação em `src.pln.avaliar`.

O modelo é carregado **uma única vez**, na primeira chamada, e fica em cache
no módulo — carregar o arquivo a cada pergunta custaria mais que a própria
classificação.

Enquanto `modelos/pln_intencoes.joblib` não existir (quem clona o repositório
não recebe a pasta `modelos/`), o módulo cai no baseline de palavras-chave e
avisa uma vez no console, em vez de quebrar o sistema. Use
`modelo_carregado()` para a interface mostrar esse estado ao usuário.
"""

from src.configuracao import LIMIAR_CONFIANCA_PLN
from src.integracao.contratos import ResultadoPLN
from src.pln import baseline
from src.pln.modelos import ARQUIVO_MODELO, ClassificadorIntencoes, carregar

_classificador: ClassificadorIntencoes | None = None
_tentou_carregar = False

MENSAGEM_SEM_MODELO = (
    "O modelo de PLN ainda não foi treinado: as intenções estão sendo "
    "reconhecidas pelo baseline de palavras-chave, que erra mais. "
    "Treine com: python -m src.pln.treinar"
)


def _obter_classificador() -> ClassificadorIntencoes | None:
    """Carrega o modelo na primeira chamada e reaproveita nas seguintes."""
    global _classificador, _tentou_carregar
    if not _tentou_carregar:
        _tentou_carregar = True
        try:
            _classificador = carregar()
        except FileNotFoundError:
            print(f"[pln] {MENSAGEM_SEM_MODELO}")
    return _classificador


def recarregar() -> None:
    """Esquece o modelo em cache (usado depois de um novo treino e nos testes)."""
    global _classificador, _tentou_carregar
    _classificador = None
    _tentou_carregar = False


def modelo_carregado() -> bool:
    """True se o classificador treinado está em uso; False se é o baseline."""
    return _obter_classificador() is not None


def descricao_do_modelo() -> str:
    """Texto curto sobre o que está respondendo, para a interface exibir."""
    classificador = _obter_classificador()
    if classificador is None:
        return "baseline de palavras-chave (modelo não treinado)"
    return f"{classificador.nome} — {classificador.descricao}"


def interpretar_pergunta(texto: str) -> ResultadoPLN:
    """Recebe a pergunta em linguagem natural e devolve a intenção reconhecida.

    A confiança é a probabilidade da intenção prevista. O módulo de decisão
    compara essa confiança com `LIMIAR_CONFIANCA_PLN` para decidir entre
    responder e pedir uma reformulação; aqui a intenção é devolvida como está,
    sem aplicar o limiar, para que a interface possa mostrar o número real.
    """
    classificador = _obter_classificador()
    if classificador is None:
        intencao, confianca = baseline.classificar(texto)
    else:
        intencao, confianca = classificador.classificar(texto)
    return ResultadoPLN(intencao=intencao, confianca=confianca, texto_original=texto)


def aceita(resultado: ResultadoPLN) -> bool:
    """Se a intenção passa o limiar de confiança do sistema.

    Conveniência para a interface e para a avaliação; a decisão final continua
    sendo do módulo de decisão.
    """
    return (
        resultado.intencao != "fora_do_escopo"
        and resultado.confianca >= LIMIAR_CONFIANCA_PLN
    )


__all__ = [
    "ARQUIVO_MODELO",
    "MENSAGEM_SEM_MODELO",
    "aceita",
    "descricao_do_modelo",
    "interpretar_pergunta",
    "modelo_carregado",
    "recarregar",
]
