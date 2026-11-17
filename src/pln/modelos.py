"""Modelos de classificação de intenção comparados no projeto.

Todos seguem o mesmo esqueleto — vetorizar o texto com TF-IDF e classificar
com um modelo linear — e mudam só na representação do texto:

- `palavras`: n-gramas de palavras (1 e 2). Representação clássica; depende de
  a palavra do uso real ter aparecido no treino.
- `caracteres`: n-gramas de caracteres (3 a 5) dentro das palavras. Mais
  robusto a erro de digitação e a falta de acento ("recilavel" ainda
  compartilha os trechos "recicl"/"ciclav" com "reciclavel"), o que importa num
  dataset pequeno escrito em linguagem informal.
- `palavras_caracteres`: as duas representações concatenadas.

As duas últimas entradas são experimentos de ablação (não concorrem à escolha
do modelo final): servem para medir no artigo o efeito de remover stopwords e,
em seguida, o efeito de remover também as negações — o erro que a seção de
pré-processamento adverte para não cometer.

`ClassificadorIntencoes` é o objeto que vai para `modelos/`: é ele que a
avaliação mede e que a integração usa, então o que foi medido é exatamente o
que roda em produção.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.pln.preprocessamento import (
    normalizar,
    normalizar_sem_negacoes,
    normalizar_sem_stopwords,
    tem_conteudo,
)

SEED = 42

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_MODELO = RAIZ / "modelos" / "pln_intencoes.joblib"

# Menos regularização que o padrão (C=1): com TF-IDF esparso e poucas centenas
# de frases, o modelo precisa de liberdade para usar os termos raros. Valor
# fixo para todos os modelos, para que a comparação isole a representação do
# texto. A escolha foi feita pela validação cruzada no treino, nunca no teste.
C_REGULARIZACAO = 10.0


def _tfidf_palavras(preprocessador=normalizar) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=preprocessador,
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )


def _tfidf_caracteres() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalizar,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
    )


def _regressao_logistica() -> LogisticRegression:
    # class_weight="balanced" porque o dataset é quase equilibrado, mas a
    # divisão estratificada ainda deixa pequenas diferenças entre as classes.
    return LogisticRegression(
        C=C_REGULARIZACAO,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )


def _svc_calibrado() -> CalibratedClassifierCV:
    # LinearSVC não produz probabilidade; a calibração sigmoide (Platt) gera a
    # confiança que o limiar de 0,5 do sistema precisa.
    return CalibratedClassifierCV(
        LinearSVC(C=C_REGULARIZACAO, class_weight="balanced", random_state=SEED),
        method="sigmoid",
        cv=3,
    )


def _palavras_e_caracteres() -> FeatureUnion:
    return FeatureUnion(
        [("palavras", _tfidf_palavras()), ("caracteres", _tfidf_caracteres())]
    )


# nome -> (descrição para a tabela do artigo, construtor, é_ablação)
CATALOGO: dict[str, tuple[str, object, bool]] = {
    "tfidf_palavras_lr": (
        "TF-IDF de palavras (1-2) + Regressão Logística",
        lambda: Pipeline([("tfidf", _tfidf_palavras()), ("clf", _regressao_logistica())]),
        False,
    ),
    "tfidf_caracteres_lr": (
        "TF-IDF de caracteres (3-5) + Regressão Logística",
        lambda: Pipeline([("tfidf", _tfidf_caracteres()), ("clf", _regressao_logistica())]),
        False,
    ),
    "tfidf_palavras_caracteres_svc": (
        "TF-IDF de palavras + caracteres e LinearSVC calibrado",
        lambda: Pipeline([("tfidf", _palavras_e_caracteres()), ("clf", _svc_calibrado())]),
        False,
    ),
    "tfidf_palavras_caracteres_lr": (
        "TF-IDF de palavras + caracteres e Regressão Logística",
        lambda: Pipeline([("tfidf", _palavras_e_caracteres()), ("clf", _regressao_logistica())]),
        False,
    ),
    "ablacao_sem_stopwords": (
        "Ablação: TF-IDF de palavras sem stopwords (negações preservadas)",
        lambda: Pipeline(
            [
                ("tfidf", _tfidf_palavras(normalizar_sem_stopwords)),
                ("clf", _regressao_logistica()),
            ]
        ),
        True,
    ),
    "ablacao_sem_negacoes": (
        "Ablação: TF-IDF de palavras sem stopwords e sem negações",
        lambda: Pipeline(
            [
                ("tfidf", _tfidf_palavras(normalizar_sem_negacoes)),
                ("clf", _regressao_logistica()),
            ]
        ),
        True,
    ),
}


def construir(nome: str) -> Pipeline:
    """Cria um pipeline novo (não treinado) a partir do catálogo."""
    if nome not in CATALOGO:
        raise KeyError(f"modelo desconhecido: {nome!r}; use um de {sorted(CATALOGO)}")
    _, construtor, _ = CATALOGO[nome]
    return construtor()


def candidatos() -> list[str]:
    """Modelos que concorrem à escolha do modelo final (sem as ablações)."""
    return [nome for nome, (_, _, ablacao) in CATALOGO.items() if not ablacao]


def descricao(nome: str) -> str:
    return CATALOGO[nome][0]


@dataclass(frozen=True)
class ClassificadorIntencoes:
    """Pipeline treinado + a regra de entrada degenerada.

    A regra: texto sem nenhuma letra ou número ("?????", "...", só emoji) é
    `fora_do_escopo` com confiança 1,0. Esse texto não gera nenhuma
    característica de TF-IDF, então o modelo decidiria só pelo intercepto —
    uma resposta arbitrária. A regra mora aqui, e não na integração, para que a
    avaliação meça o mesmo comportamento que o sistema entrega ao usuário.
    """

    nome: str
    descricao: str
    pipeline: Pipeline

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.classes_)

    def classificar(self, texto: str) -> tuple[str, float]:
        """Devolve (intenção, confiança) para uma pergunta."""
        if not tem_conteudo(texto):
            return "fora_do_escopo", 1.0
        probabilidades = self.pipeline.predict_proba([texto])[0]
        indice = int(np.argmax(probabilidades))
        return self.classes[indice], float(probabilidades[indice])

    def prever(self, textos: list[str]) -> list[str]:
        return [self.classificar(t)[0] for t in textos]

    def prever_proba(self, textos: list[str]) -> np.ndarray:
        """Matriz (n_textos x n_classes) na ordem de `self.classes`."""
        classes = self.classes
        saida = np.zeros((len(textos), len(classes)))
        com_conteudo = [i for i, t in enumerate(textos) if tem_conteudo(t)]
        if com_conteudo:
            saida[com_conteudo] = self.pipeline.predict_proba([textos[i] for i in com_conteudo])
        for i in set(range(len(textos))) - set(com_conteudo):
            saida[i, classes.index("fora_do_escopo")] = 1.0
        return saida


def salvar(classificador: ClassificadorIntencoes, caminho: Path = ARQUIVO_MODELO) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classificador, caminho)
    return caminho


def carregar(caminho: Path = ARQUIVO_MODELO) -> ClassificadorIntencoes:
    if not caminho.exists():
        raise FileNotFoundError(
            f"modelo de PLN não encontrado em {caminho}. "
            "Treine com: python -m src.pln.treinar"
        )
    return joblib.load(caminho)
