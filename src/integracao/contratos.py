"""Contratos entre os módulos: o que cada um recebe e devolve.

Fluxo: imagem -> Visão -> ResultadoVisao
       pergunta -> PLN -> ResultadoPLN
       (ResultadoVisao, ResultadoPLN) -> Decisão -> Decisao

Qualquer mudança aqui afeta todos os módulos: combinar com a equipe antes.
Os __post_init__ validam os valores para que um erro de contrato apareça
logo no módulo que o causou, e não lá na interface.
"""

from dataclasses import dataclass, field

from src.configuracao import ACOES, CLASSES, INTENCOES


def _validar_confianca(valor: float) -> None:
    if not 0.0 <= valor <= 1.0:
        raise ValueError(f"confianca deve estar entre 0 e 1, recebido {valor}")


@dataclass(frozen=True)
class ResultadoVisao:
    """Saída do módulo de Visão Computacional / Machine Learning."""

    classe: str                                   # uma de CLASSES
    confianca: float                              # probabilidade da classe prevista
    top3: list[tuple[str, float]] = field(default_factory=list)  # [(classe, prob), ...]

    def __post_init__(self) -> None:
        if self.classe not in CLASSES:
            raise ValueError(f"classe desconhecida: {self.classe!r}; use uma de {CLASSES}")
        _validar_confianca(self.confianca)
        for classe, prob in self.top3:
            if classe not in CLASSES:
                raise ValueError(f"classe desconhecida no top3: {classe!r}")
            _validar_confianca(prob)


@dataclass(frozen=True)
class ResultadoPLN:
    """Saída do módulo de Processamento de Linguagem Natural."""

    intencao: str        # uma de INTENCOES
    confianca: float
    texto_original: str

    def __post_init__(self) -> None:
        if self.intencao not in INTENCOES:
            raise ValueError(f"intenção desconhecida: {self.intencao!r}; use uma de {INTENCOES}")
        _validar_confianca(self.confianca)


@dataclass(frozen=True)
class Decisao:
    """Saída do módulo de decisão: o que o sistema responde ao usuário."""

    resposta: str             # texto mostrado ao usuário
    acao: str                 # uma de ACOES
    lixeira: str | None = None  # ex.: "vermelha (plástico)"; None quando não se aplica

    def __post_init__(self) -> None:
        if self.acao not in ACOES:
            raise ValueError(f"ação desconhecida: {self.acao!r}; use uma de {ACOES}")


@dataclass(frozen=True)
class ResultadoPipeline:
    """Tudo que o fluxo produziu, para a interface exibir cada etapa."""

    visao: ResultadoVisao
    pln: ResultadoPLN
    decisao: Decisao
