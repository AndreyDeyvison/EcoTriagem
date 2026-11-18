"""Avaliação final do modelo de visão no conjunto de teste.

Roda uma única vez, depois do treino: o conjunto de teste não participou nem
da escolha da época (feita pelo F1-macro na validação) nem da escolha do C do
baseline, então os números aqui são a estimativa honesta de desempenho.

A inferência usa o **mesmo objeto que a interface usa**
(`src.visao_computacional.classificador`), e não uma cópia do pré-processamento:
assim o que é medido é exatamente o que o usuário recebe.

O que produz:

resultados/metricas/
    comparacao_modelos_visao.csv/.json   baseline clássico x CNN
    analise_limiar_visao.csv             varredura do limiar de confiança
    avaliacao_mundo_real.json            fotos da equipe (quando existirem)
resultados/graficos/
    matriz_confusao_visao.png            absoluta e normalizada
    curva_limiar_visao.png

Executar a partir da raiz do repositório:
    python -m src.machine_learning.avaliar
"""

import csv
import json
import re
from datetime import date

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.configuracao import CLASSES, LIMIAR_CONFIANCA_VISAO
from src.machine_learning.dados_visao import RAIZ, carregar_divisao, por_conjunto
from src.visao_computacional import classificador

PASTA_METRICAS = RAIZ / "resultados" / "metricas"
PASTA_GRAFICOS = RAIZ / "resultados" / "graficos"
PASTA_FOTOS_EQUIPE = RAIZ / "dados" / "fotos_equipe"

ARQUIVO_BASELINE = PASTA_METRICAS / "metricas_baseline_visao.json"
ARQUIVO_COMPARACAO_CSV = PASTA_METRICAS / "comparacao_modelos_visao.csv"
ARQUIVO_COMPARACAO_JSON = PASTA_METRICAS / "comparacao_modelos_visao.json"
ARQUIVO_LIMIAR = PASTA_METRICAS / "analise_limiar_visao.csv"
ARQUIVO_MUNDO_REAL = PASTA_METRICAS / "avaliacao_mundo_real.json"
ARQUIVO_MATRIZ = PASTA_GRAFICOS / "matriz_confusao_visao.png"
ARQUIVO_CURVA_LIMIAR = PASTA_GRAFICOS / "curva_limiar_visao.png"

LIMIARES = np.round(np.arange(0.30, 1.00, 0.05), 2)
LOTE = 32

# classe_nome_numero.jpg -> a classe é o primeiro campo
PADRAO_FOTO_EQUIPE = re.compile(r"^(?P<classe>[a-z]+)_[a-z0-9]+_\d+\.(jpg|jpeg|png)$", re.IGNORECASE)


def prever(modelo, caminhos: list) -> tuple[np.ndarray, list[str]]:
    """Probabilidades (n x classes) pelo mesmo caminho que a interface usa."""
    tensores = []
    for caminho in caminhos:
        with Image.open(caminho) as imagem:
            imagem = imagem.convert("RGB")
            tensores.append(modelo.transformar(imagem))

    saidas = []
    with torch.no_grad():
        for inicio in range(0, len(tensores), LOTE):
            lote = torch.stack(tensores[inicio : inicio + LOTE])
            saidas.append(torch.softmax(modelo.rede(lote), dim=1))
    return torch.cat(saidas).numpy(), modelo.classes


def metricas(verdadeiros: list[str], previstos: list[str]) -> dict:
    relatorio = classification_report(
        verdadeiros, previstos, labels=list(CLASSES), output_dict=True, zero_division=0
    )
    return {
        "acuracia": float(accuracy_score(verdadeiros, previstos)),
        "f1_macro": float(f1_score(verdadeiros, previstos, average="macro", zero_division=0)),
        "por_classe": {
            classe: {
                "precisao": round(relatorio[classe]["precision"], 4),
                "revocacao": round(relatorio[classe]["recall"], 4),
                "f1": round(relatorio[classe]["f1-score"], 4),
                "suporte": int(relatorio[classe]["support"]),
            }
            for classe in CLASSES
        },
    }


def varrer_limiar(verdadeiros, previstos, confiancas) -> list[dict]:
    """Cobertura e acurácia das aceitas para cada limiar de confiança.

    Abaixo do limiar o sistema não responde: pede outra foto. Subir o limiar
    aumenta a acurácia do que é respondido e diminui quanto é respondido — a
    tabela mostra o preço exato dessa troca.
    """
    verdadeiros = np.array(verdadeiros)
    previstos = np.array(previstos)
    correto = verdadeiros == previstos
    total = len(verdadeiros)

    linhas = []
    for limiar in LIMIARES:
        aceitas = confiancas >= limiar
        n_aceitas = int(aceitas.sum())
        linhas.append(
            {
                "limiar": float(limiar),
                "cobertura": n_aceitas / total,
                "aceitas": n_aceitas,
                "acuracia_aceitas": float(correto[aceitas].mean()) if n_aceitas else 0.0,
                "acuracia_geral": float(correto.mean()),
                "pedidos_de_nova_foto": total - n_aceitas,
            }
        )
    return linhas


def _grafico_matriz(verdadeiros, previstos) -> None:
    rotulos = list(CLASSES)
    absoluta = confusion_matrix(verdadeiros, previstos, labels=rotulos)
    normalizada = confusion_matrix(verdadeiros, previstos, labels=rotulos, normalize="true")

    figura, eixos = plt.subplots(1, 2, figsize=(15, 6.5))
    ConfusionMatrixDisplay(absoluta, display_labels=rotulos).plot(
        ax=eixos[0], cmap="Blues", colorbar=False, values_format="d"
    )
    eixos[0].set_title("Contagens absolutas")
    ConfusionMatrixDisplay(normalizada, display_labels=rotulos).plot(
        ax=eixos[1], cmap="Blues", colorbar=False, values_format=".2f"
    )
    eixos[1].set_title("Normalizada por classe correta (revocação)")

    for eixo in eixos:
        eixo.set_xlabel("classe prevista")
        eixo.set_ylabel("classe correta")
        plt.setp(eixo.get_xticklabels(), rotation=35, ha="right")
    figura.suptitle("Visão — matriz de confusão (conjunto de teste)")
    figura.tight_layout()
    figura.savefig(ARQUIVO_MATRIZ, dpi=160)
    plt.close(figura)


def _grafico_limiar(linhas: list[dict]) -> None:
    limiares = [linha["limiar"] for linha in linhas]
    figura, eixo = plt.subplots(figsize=(8, 5))
    eixo.plot(limiares, [l["cobertura"] for l in linhas], marker="o", ms=3,
              label="Cobertura (previsões aceitas)")
    eixo.plot(limiares, [l["acuracia_aceitas"] for l in linhas], marker="o", ms=3,
              label="Acurácia entre as aceitas")
    eixo.axhline(linhas[0]["acuracia_geral"], color="gray", linestyle=":",
                 label=f"acurácia sem limiar ({linhas[0]['acuracia_geral']:.3f})")
    eixo.axvline(LIMIAR_CONFIANCA_VISAO, color="crimson", linestyle="--",
                 label=f"limiar adotado ({LIMIAR_CONFIANCA_VISAO})")
    eixo.set_xlabel("limiar de confiança da visão")
    eixo.set_ylabel("proporção")
    eixo.set_ylim(0, 1.02)
    eixo.set_title("Visão — efeito do limiar de confiança (conjunto de teste)")
    eixo.grid(alpha=0.3)
    eixo.legend(loc="lower left", fontsize=9)
    figura.tight_layout()
    figura.savefig(ARQUIVO_CURVA_LIMIAR, dpi=160)
    plt.close(figura)


def _caminho_curto(caminho) -> str:
    """Caminho relativo à raiz quando possível; absoluto caso contrário."""
    try:
        return str(caminho.relative_to(RAIZ))
    except ValueError:
        return str(caminho)


def avaliar_fotos_da_equipe(modelo) -> dict | None:
    """Testa o modelo nas fotos reais da equipe, se a pasta existir.

    Espera arquivos `classe_nome_numero.jpg` em dados/fotos_equipe/. O TrashNet
    tem fundo branco e iluminação controlada; fotos de casa não têm. É esperado
    que o desempenho caia, e o quanto ele cai é o resultado que interessa.
    """
    if not PASTA_FOTOS_EQUIPE.exists():
        print(f"\n[fotos da equipe] pasta {_caminho_curto(PASTA_FOTOS_EQUIPE)} não existe; "
              "etapa pulada (crie a pasta e nomeie como classe_nome_numero.jpg).")
        return None

    caminhos, rotulos, ignorados = [], [], []
    for caminho in sorted(PASTA_FOTOS_EQUIPE.iterdir()):
        if caminho.name.startswith("."):
            continue
        casamento = PADRAO_FOTO_EQUIPE.match(caminho.name)
        if not casamento:
            ignorados.append(f"{caminho.name}: fora do padrão classe_nome_numero.jpg")
            continue
        classe = casamento.group("classe").lower()
        if classe not in CLASSES:
            ignorados.append(f"{caminho.name}: classe {classe!r} desconhecida")
            continue
        caminhos.append(caminho)
        rotulos.append(classe)

    if not caminhos:
        print(f"\n[fotos da equipe] nenhuma foto válida em {_caminho_curto(PASTA_FOTOS_EQUIPE)}; "
              "etapa pulada.")
        for aviso in ignorados:
            print(f"  ignorado - {aviso}")
        return None

    probabilidades, classes = prever(modelo, caminhos)
    previstos = [classes[i] for i in probabilidades.argmax(axis=1)]
    confiancas = probabilidades.max(axis=1)

    resultado = metricas(rotulos, previstos)
    resultado["n_fotos"] = len(caminhos)
    resultado["confianca_media"] = round(float(confiancas.mean()), 4)
    resultado["ignorados"] = ignorados
    resultado["erros"] = [
        {"arquivo": c.name, "esperado": v, "previsto": p, "confianca": round(float(conf), 4)}
        for c, v, p, conf in zip(caminhos, rotulos, previstos, confiancas)
        if v != p
    ]
    print(f"\n[fotos da equipe] {len(caminhos)} fotos | acurácia {resultado['acuracia']:.4f} | "
          f"F1-macro {resultado['f1_macro']:.4f}")
    return resultado


def main() -> None:
    PASTA_METRICAS.mkdir(parents=True, exist_ok=True)
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

    modelo = classificador.carregar_modelo_obrigatorio()
    checkpoint = torch.load(classificador.CAMINHO_MODELO, map_location="cpu", weights_only=True)
    print(f"Modelo: {checkpoint['arquitetura']} | melhor época {checkpoint['epoca']} | "
          f"F1-macro validação {checkpoint['f1_macro_validacao']:.4f}")

    teste = por_conjunto(carregar_divisao())["teste"]
    print(f"Teste: {len(teste)} imagens\n")

    probabilidades, classes = prever(modelo, [r.caminho for r in teste])
    verdadeiros = [r.classe for r in teste]
    previstos = [classes[i] for i in probabilidades.argmax(axis=1)]
    confiancas = probabilidades.max(axis=1)

    metricas_cnn = metricas(verdadeiros, previstos)
    baseline = json.loads(ARQUIVO_BASELINE.read_text(encoding="utf-8"))
    metricas_baseline = baseline["teste"]

    print(f"{'modelo':<28}{'acurácia':>10}{'F1-macro':>10}")
    print(f"{'baseline (HSV+HOG e SVM)':<28}{metricas_baseline['acuracia']:>10.4f}"
          f"{metricas_baseline['f1_macro']:>10.4f}")
    print(f"{'CNN (MobileNetV2)':<28}{metricas_cnn['acuracia']:>10.4f}{metricas_cnn['f1_macro']:>10.4f}")
    ganho_f1 = metricas_cnn["f1_macro"] - metricas_baseline["f1_macro"]
    print(f"{'ganho da CNN':<28}{metricas_cnn['acuracia'] - metricas_baseline['acuracia']:>+10.4f}"
          f"{ganho_f1:>+10.4f}")

    print(f"\n{'classe':<9}{'precisão':>10}{'revocação':>11}{'F1':>8}{'F1 base':>9}{'n':>5}")
    linhas_comparacao = []
    for classe in CLASSES:
        cnn = metricas_cnn["por_classe"][classe]
        base = metricas_baseline["por_classe"][classe]
        linhas_comparacao.append(
            {
                "classe": classe,
                "precisao_cnn": cnn["precisao"], "revocacao_cnn": cnn["revocacao"], "f1_cnn": cnn["f1"],
                "precisao_baseline": base["precisao"], "revocacao_baseline": base["revocacao"],
                "f1_baseline": base["f1"], "suporte": cnn["suporte"],
            }
        )
        print(f"{classe:<9}{cnn['precisao']:>10.3f}{cnn['revocacao']:>11.3f}"
              f"{cnn['f1']:>8.3f}{base['f1']:>9.3f}{cnn['suporte']:>5}")

    with ARQUIVO_COMPARACAO_CSV.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas_comparacao[0]))
        escritor.writeheader()
        escritor.writerows(linhas_comparacao)

    linhas_limiar = varrer_limiar(verdadeiros, previstos, confiancas)
    with ARQUIVO_LIMIAR.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas_limiar[0]))
        escritor.writeheader()
        escritor.writerows([{c: (round(v, 4) if isinstance(v, float) else v)
                             for c, v in linha.items()} for linha in linhas_limiar])

    no_limiar = min(linhas_limiar, key=lambda l: abs(l["limiar"] - LIMIAR_CONFIANCA_VISAO))
    print(f"\nLimiar adotado ({LIMIAR_CONFIANCA_VISAO}): aceita {no_limiar['cobertura']:.1%} das "
          f"imagens com {no_limiar['acuracia_aceitas']:.1%} de acerto "
          f"({no_limiar['pedidos_de_nova_foto']} pedidos de nova foto)")

    _grafico_matriz(verdadeiros, previstos)
    _grafico_limiar(linhas_limiar)

    mundo_real = avaliar_fotos_da_equipe(modelo)
    if mundo_real is not None:
        mundo_real["queda_em_relacao_ao_teste"] = {
            "acuracia": round(mundo_real["acuracia"] - metricas_cnn["acuracia"], 4),
            "f1_macro": round(mundo_real["f1_macro"] - metricas_cnn["f1_macro"], 4),
        }
        ARQUIVO_MUNDO_REAL.write_text(
            json.dumps({"gerado_em": date.today().isoformat(),
                        "referencia_teste_trashnet": {
                            "acuracia": metricas_cnn["acuracia"],
                            "f1_macro": metricas_cnn["f1_macro"]},
                        **mundo_real}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Relatório salvo em {ARQUIVO_MUNDO_REAL.relative_to(RAIZ)}")

    ARQUIVO_COMPARACAO_JSON.write_text(
        json.dumps(
            {
                "gerado_em": date.today().isoformat(),
                "conjunto": "teste",
                "imagens": len(teste),
                "baseline": {
                    "descricao": baseline["modelo"],
                    "acuracia": metricas_baseline["acuracia"],
                    "f1_macro": metricas_baseline["f1_macro"],
                    "por_classe": metricas_baseline["por_classe"],
                },
                "cnn": {
                    "descricao": f"{checkpoint['arquitetura']} (transferência de aprendizado)",
                    "melhor_epoca": checkpoint["epoca"],
                    "f1_macro_validacao": checkpoint["f1_macro_validacao"],
                    "acuracia": metricas_cnn["acuracia"],
                    "f1_macro": metricas_cnn["f1_macro"],
                    "por_classe": metricas_cnn["por_classe"],
                },
                "ganho_da_cnn": {
                    "acuracia": round(metricas_cnn["acuracia"] - metricas_baseline["acuracia"], 4),
                    "f1_macro": round(ganho_f1, 4),
                },
                "limiar": {
                    "adotado": LIMIAR_CONFIANCA_VISAO,
                    "no_limiar_adotado": no_limiar,
                    "varredura": linhas_limiar,
                },
                "fotos_da_equipe": mundo_real,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nComparação em {ARQUIVO_COMPARACAO_CSV.relative_to(RAIZ)}")
    print(f"Limiar em {ARQUIVO_LIMIAR.relative_to(RAIZ)}")
    print(f"Gráficos em {ARQUIVO_MATRIZ.relative_to(RAIZ)} e {ARQUIVO_CURVA_LIMIAR.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
