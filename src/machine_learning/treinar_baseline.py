"""Baseline clássico de visão: características manuais + SVM.

Existe para medir o quanto a CNN realmente ganha. Sem esse ponto de
comparação, "87% de acurácia" não diz se o problema é fácil ou se o modelo é
bom — é a mesma função do baseline de palavras-chave no módulo de PLN.

Características (sem rede neural, tudo calculado à mão):

- **Histograma HSV** (16 bins de matiz, 8 de saturação, 8 de valor): cor é o
  sinal mais óbvio para separar papelão de vidro, e o HSV separa matiz de
  iluminação melhor que o RGB.
- **HOG** — histograma de gradientes orientados (células de 16x16, 9 bins de
  orientação, normalização L2 por célula): descreve forma e textura. Implementado
  com numpy para não acrescentar dependência; é a formulação clássica de Dalal e
  Triggs sem a normalização por blocos sobrepostos.

Classificador: SVM com núcleo RBF, `class_weight="balanced"` por causa da
classe `lixo`. O C é escolhido no conjunto de **validação** e o modelo final é
medido uma única vez no **teste**.

Saídas:
    modelos/baseline_visao.joblib
    resultados/metricas/metricas_baseline_visao.json

Executar a partir da raiz do repositório:
    python -m src.machine_learning.treinar_baseline
"""

import json
import random
import time
from datetime import date

import joblib
import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.configuracao import PASTA_MODELOS
from src.machine_learning.dados_visao import (
    CLASSES_ORDENADAS,
    RAIZ,
    carregar_divisao,
    por_conjunto,
)

SEED = 42

LADO = 128          # imagens reduzidas antes de extrair as características
BINS_MATIZ = 16
BINS_SATURACAO = 8
BINS_VALOR = 8
LADO_CELULA = 16    # células de 16x16 -> grade de 8x8
BINS_ORIENTACAO = 9
GRADE = LADO // LADO_CELULA

VALORES_C = (1.0, 10.0, 100.0)

ARQUIVO_MODELO = PASTA_MODELOS / "baseline_visao.joblib"
ARQUIVO_METRICAS = RAIZ / "resultados" / "metricas" / "metricas_baseline_visao.json"


def _fixar_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def histograma_hsv(imagem: Image.Image) -> np.ndarray:
    """Histogramas normalizados de matiz, saturação e valor, concatenados."""
    hsv = np.asarray(imagem.convert("HSV"), dtype=np.float32) / 255.0
    partes = []
    for canal, bins in enumerate((BINS_MATIZ, BINS_SATURACAO, BINS_VALOR)):
        histograma, _ = np.histogram(hsv[:, :, canal], bins=bins, range=(0.0, 1.0))
        partes.append(histograma / histograma.sum())
    return np.concatenate(partes).astype(np.float32)


def hog(imagem: Image.Image) -> np.ndarray:
    """Histograma de gradientes orientados, normalizado por célula.

    Gradientes por diferenças centrais, orientação sem sinal (0 a 180 graus)
    para que bordas claro/escuro e escuro/claro contem como a mesma borda, e
    voto distribuído entre os dois bins vizinhos (interpolação linear), que é o
    que evita o serrilhado do histograma.
    """
    cinza = np.asarray(imagem.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(cinza)
    magnitude = np.hypot(gx, gy)
    orientacao = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    posicao = orientacao / (180.0 / BINS_ORIENTACAO)
    bin_baixo = np.floor(posicao).astype(np.int32) % BINS_ORIENTACAO
    bin_alto = (bin_baixo + 1) % BINS_ORIENTACAO
    peso_alto = posicao - np.floor(posicao)

    linhas, colunas = np.indices(cinza.shape)
    celula = (linhas // LADO_CELULA) * GRADE + (colunas // LADO_CELULA)

    histograma = np.zeros(GRADE * GRADE * BINS_ORIENTACAO, dtype=np.float32)
    np.add.at(histograma, celula * BINS_ORIENTACAO + bin_baixo, magnitude * (1.0 - peso_alto))
    np.add.at(histograma, celula * BINS_ORIENTACAO + bin_alto, magnitude * peso_alto)

    por_celula = histograma.reshape(GRADE * GRADE, BINS_ORIENTACAO)
    normas = np.linalg.norm(por_celula, axis=1, keepdims=True) + 1e-6
    return (por_celula / normas).ravel()


def extrair(caminho) -> np.ndarray:
    with Image.open(caminho) as imagem:
        imagem = imagem.convert("RGB").resize((LADO, LADO), Image.BILINEAR)
        return np.concatenate([histograma_hsv(imagem), hog(imagem)])


def matriz_de_caracteristicas(registros, rotulo: str) -> tuple[np.ndarray, np.ndarray]:
    inicio = time.perf_counter()
    X = np.stack([extrair(r.caminho) for r in registros])
    y = np.array([r.classe for r in registros])
    print(f"  {rotulo:<10} {X.shape[0]:>5} imagens -> {X.shape[1]} características "
          f"({time.perf_counter() - inicio:.1f}s)")
    return X, y


def _construir(c: float, probabilidade: bool) -> Pipeline:
    """SVM RBF; com `probabilidade`, envolvido numa calibração sigmoide.

    O SVM não produz probabilidade por construção. `SVC(probability=True)` foi
    depreciado no scikit-learn 1.9, então a calibração é feita explicitamente
    com `CalibratedClassifierCV`, que é o que aquele parâmetro fazia por dentro.
    Só o modelo final é calibrado: na busca do C basta a classe prevista.
    """
    svm = SVC(
        C=c,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        random_state=SEED,
    )
    classificador = (
        CalibratedClassifierCV(svm, method="sigmoid", cv=3, ensemble=False)
        if probabilidade
        else svm
    )
    return Pipeline([("escala", StandardScaler()), ("svm", classificador)])


def _metricas(y_verdadeiro, y_previsto) -> dict:
    relatorio = classification_report(
        y_verdadeiro, y_previsto, labels=list(CLASSES_ORDENADAS), output_dict=True, zero_division=0
    )
    return {
        "acuracia": float(accuracy_score(y_verdadeiro, y_previsto)),
        "f1_macro": float(f1_score(y_verdadeiro, y_previsto, average="macro", zero_division=0)),
        "por_classe": {
            classe: {
                "precisao": round(relatorio[classe]["precision"], 4),
                "revocacao": round(relatorio[classe]["recall"], 4),
                "f1": round(relatorio[classe]["f1-score"], 4),
                "suporte": int(relatorio[classe]["support"]),
            }
            for classe in CLASSES_ORDENADAS
        },
    }


def main() -> None:
    _fixar_seeds()
    ARQUIVO_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

    grupos = por_conjunto(carregar_divisao())
    print("Extraindo características (histograma HSV + HOG):")
    X_treino, y_treino = matriz_de_caracteristicas(grupos["treino"], "treino")
    X_validacao, y_validacao = matriz_de_caracteristicas(grupos["validacao"], "validação")
    X_teste, y_teste = matriz_de_caracteristicas(grupos["teste"], "teste")

    print("\nEscolhendo o C na validação (F1-macro):")
    resultados_c = []
    for c in VALORES_C:
        modelo = _construir(c, probabilidade=False)
        modelo.fit(X_treino, y_treino)
        f1 = f1_score(y_validacao, modelo.predict(X_validacao), average="macro", zero_division=0)
        resultados_c.append({"C": c, "f1_macro_validacao": round(float(f1), 4)})
        print(f"  C={c:<6} F1-macro validação = {f1:.4f}")

    melhor = max(resultados_c, key=lambda r: r["f1_macro_validacao"])
    print(f"\nMelhor C: {melhor['C']} — treinando o modelo final com probabilidades...")

    modelo = _construir(melhor["C"], probabilidade=True)
    modelo.fit(X_treino, y_treino)

    metricas_validacao = _metricas(y_validacao, modelo.predict(X_validacao))
    metricas_teste = _metricas(y_teste, modelo.predict(X_teste))

    print(f"\n{'conjunto':<12}{'acurácia':>10}{'F1-macro':>10}")
    print(f"{'validação':<12}{metricas_validacao['acuracia']:>10.4f}{metricas_validacao['f1_macro']:>10.4f}")
    print(f"{'teste':<12}{metricas_teste['acuracia']:>10.4f}{metricas_teste['f1_macro']:>10.4f}")

    print(f"\n{'classe':<9}{'precisão':>10}{'revocação':>11}{'F1':>8}{'n':>5}  (teste)")
    for classe, m in metricas_teste["por_classe"].items():
        print(f"{classe:<9}{m['precisao']:>10.3f}{m['revocacao']:>11.3f}{m['f1']:>8.3f}{m['suporte']:>5}")

    joblib.dump(
        {
            "modelo": modelo,
            "classes": list(CLASSES_ORDENADAS),
            "lado": LADO,
            "caracteristicas": "histograma HSV (32) + HOG (576)",
            "seed": SEED,
        },
        ARQUIVO_MODELO,
    )

    ARQUIVO_METRICAS.write_text(
        json.dumps(
            {
                "gerado_em": date.today().isoformat(),
                "modelo": "SVM RBF sobre histograma HSV + HOG",
                "seed": SEED,
                "n_caracteristicas": int(X_treino.shape[1]),
                "busca_de_C": resultados_c,
                "C_escolhido": melhor["C"],
                "imagens": {
                    "treino": len(grupos["treino"]),
                    "validacao": len(grupos["validacao"]),
                    "teste": len(grupos["teste"]),
                },
                "validacao": metricas_validacao,
                "teste": metricas_teste,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nModelo salvo em {ARQUIVO_MODELO.relative_to(RAIZ)}")
    print(f"Métricas salvas em {ARQUIVO_METRICAS.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
