"""Extração do objeto citado na pergunta (item extra do PLN).

Para que serve: a Visão só conhece as seis classes do TrashNet
(`src.configuracao.CLASSES`). Quando o usuário escreve "onde jogo essa
pilha?", a foto pode até ser classificada como "metal", mas a resposta certa
não é a lixeira amarela — pilha é resíduo perigoso e vai para logística
reversa. O texto carrega uma informação que a imagem não tem.

O que entrega:

- `classe_sugerida`: a classe de resíduo quando o próprio objeto já determina
  o material ("latinha" -> metal). Fica `None` quando o termo é ambíguo
  ("garrafa" pode ser vidro ou plástico) — nesse caso quem decide é a Visão.
- `motivo`: preenchido quando o resíduo exige tratamento fora das seis
  classes (pilha, lâmpada, eletrônico, óleo de cozinha...). É o gancho para os
  casos especiais do módulo de decisão.

Abordagem: léxico com casamento por limites de palavra sobre o texto
normalizado (sem acento, minúsculas), com os termos compostos testados antes
dos simples. É reconhecimento de entidades por dicionário — adequado a um
domínio fechado e pequeno, e sem necessidade de dados anotados por posição.

Isto **não** altera o contrato `ResultadoPLN`: adicionar um campo lá precisa
ser combinado com a equipe (regra 2 do plano de trabalho). Até lá, o módulo de
decisão pode chamar `extrair_objeto` diretamente.
"""

import re
from dataclasses import dataclass

from src.configuracao import CLASSES
from src.pln.preprocessamento import normalizar

# Motivos de descarte especial, reaproveitados por vários termos.
_PERIGOSO = "resíduo perigoso: logística reversa (não vai na coleta comum)"
_ELETRONICO = "lixo eletrônico: logística reversa em ponto de coleta ou loja"
_SAUDE = "resíduo de serviço de saúde: devolver em farmácia ou posto de saúde"
_MULTICAMADA = "embalagem multicamada (papel + plástico + alumínio): depende da cooperativa"
_NAO_RECICLA_COM_VIDRO = "não é reciclado junto com o vidro (composição diferente)"

# termo normalizado -> (nome exibido, classe sugerida ou None, motivo ou None)
LEXICO: dict[str, tuple[str, str | None, str | None]] = {
    # Material determinado pelo próprio objeto
    "latinha": ("latinha", "metal", None),
    "lata": ("lata", "metal", None),
    "lata de aluminio": ("lata de alumínio", "metal", None),
    "tampinha de metal": ("tampinha de metal", "metal", None),
    "garrafa pet": ("garrafa PET", "plastico", None),
    "garrafa de vidro": ("garrafa de vidro", "vidro", None),
    "pote de vidro": ("pote de vidro", "vidro", None),
    "copo de vidro": ("copo de vidro", "vidro", None),
    "sacola plastica": ("sacola plástica", "plastico", None),
    "saco plastico": ("saco plástico", "plastico", None),
    "caixa de papelao": ("caixa de papelão", "papelao", None),
    "papelao": ("papelão", "papelao", None),
    "jornal": ("jornal", "papel", None),
    "revista": ("revista", "papel", None),
    "folha de papel": ("folha de papel", "papel", None),
    "caderno": ("caderno", "papel", None),
    # Rejeito: recicláveis na aparência, rejeito na prática
    "fralda": ("fralda", "lixo", "material contaminado: vai para o rejeito"),
    "papel higienico": ("papel higiênico", "lixo", "material contaminado: vai para o rejeito"),
    "guardanapo": ("guardanapo usado", "lixo", "material contaminado: vai para o rejeito"),
    "caixa de pizza": ("caixa de pizza", "lixo", "papelão engordurado não é reciclável"),
    "papel engordurado": ("papel engordurado", "lixo", "gordura inviabiliza a reciclagem do papel"),
    "absorvente": ("absorvente", "lixo", "material contaminado: vai para o rejeito"),
    "bituca": ("bituca de cigarro", "lixo", "vai para o rejeito"),
    # Descarte especial (fora das seis classes)
    "pilha": ("pilha", None, _PERIGOSO),
    "bateria": ("bateria", None, _PERIGOSO),
    "lampada": ("lâmpada", None, "pode conter mercúrio: ponto de coleta específico"),
    "celular": ("celular", None, _ELETRONICO),
    "eletronico": ("eletrônico", None, _ELETRONICO),
    "computador": ("computador", None, _ELETRONICO),
    "carregador": ("carregador", None, _ELETRONICO),
    "cabo": ("cabo", None, _ELETRONICO),
    "remedio": ("remédio", None, _SAUDE),
    "medicamento": ("medicamento", None, _SAUDE),
    "seringa": ("seringa", None, _SAUDE),
    "agulha": ("agulha", None, _SAUDE),
    "oleo de cozinha": ("óleo de cozinha", None, "óleo de cozinha: ponto de coleta específico; nunca na pia"),
    "pneu": ("pneu", None, "logística reversa: borracheiras e ecopontos"),
    "tinta": ("tinta", None, "resíduo químico: ponto de coleta específico"),
    "aerossol": ("aerossol", None, "embalagem pressurizada: ponto de coleta específico"),
    "caixa de leite": ("embalagem longa vida", None, _MULTICAMADA),
    "longa vida": ("embalagem longa vida", None, _MULTICAMADA),
    "isopor": ("isopor", "plastico", "isopor é reciclável mas poucas cooperativas aceitam"),
    "espelho": ("espelho", None, _NAO_RECICLA_COM_VIDRO),
    "ceramica": ("cerâmica", None, _NAO_RECICLA_COM_VIDRO),
    "louca": ("louça", None, _NAO_RECICLA_COM_VIDRO),
    "vidro quebrado": ("vidro quebrado", "vidro", "embrulhe em papel grosso para proteger os coletores"),
    # Ambíguos: o material quem diz é a Visão
    "garrafa": ("garrafa", None, None),
    "embalagem": ("embalagem", None, None),
    "pote": ("pote", None, None),
    "copo": ("copo", None, None),
    "caixa": ("caixa", None, None),
    "tampa": ("tampa", None, None),
    "sacola": ("sacola", None, None),
}


@dataclass(frozen=True)
class ObjetoCitado:
    """Objeto reconhecido no texto da pergunta."""

    termo: str                     # como apareceu no léxico
    objeto: str                    # nome para exibir ao usuário
    classe_sugerida: str | None    # uma de CLASSES, ou None se o termo é ambíguo
    motivo: str | None             # por que exige tratamento especial, se exigir

    def __post_init__(self) -> None:
        if self.classe_sugerida is not None and self.classe_sugerida not in CLASSES:
            raise ValueError(f"classe desconhecida: {self.classe_sugerida!r}; use uma de {CLASSES}")

    @property
    def especial(self) -> bool:
        """True quando o objeto pede tratamento fora da regra da lixeira."""
        return self.motivo is not None


# Termos compostos antes dos simples: "garrafa de vidro" vence "garrafa".
_TERMOS_ORDENADOS = sorted(LEXICO, key=lambda termo: (-len(termo.split()), -len(termo)))
_PADROES = {termo: re.compile(rf"\b{re.escape(termo)}s?\b") for termo in _TERMOS_ORDENADOS}


def extrair_objetos(texto: str) -> list[ObjetoCitado]:
    """Todos os objetos citados no texto, dos termos mais específicos aos gerais.

    Trechos já consumidos por um termo composto não são reaproveitados, para
    "onde jogo essa garrafa de vidro?" não devolver "garrafa de vidro" e
    "garrafa" ao mesmo tempo.
    """
    normalizado = normalizar(texto)
    encontrados: list[ObjetoCitado] = []
    intervalos: list[tuple[int, int]] = []

    for termo in _TERMOS_ORDENADOS:
        for casamento in _PADROES[termo].finditer(normalizado):
            inicio, fim = casamento.span()
            if any(inicio < f and i < fim for i, f in intervalos):
                continue
            intervalos.append((inicio, fim))
            objeto, classe, motivo = LEXICO[termo]
            encontrados.append(
                ObjetoCitado(termo=termo, objeto=objeto, classe_sugerida=classe, motivo=motivo)
            )
    return encontrados


def extrair_objeto(texto: str) -> ObjetoCitado | None:
    """O objeto mais informativo citado no texto (ou None se nenhum).

    Prioridade: objeto com descarte especial > objeto que determina o material
    > objeto ambíguo. Assim "onde jogo a pilha dessa lanterna de plástico?"
    responde sobre a pilha, que é o caso que o usuário precisa saber.
    """
    objetos = extrair_objetos(texto)
    if not objetos:
        return None
    return max(
        objetos,
        key=lambda o: (o.motivo is not None, o.classe_sugerida is not None),
    )
