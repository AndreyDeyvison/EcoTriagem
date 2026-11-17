"""Divide o TrashNet em treino, validação e teste (estratificado, seed fixa).

Antes de dividir, trata imagens idênticas (mesmo conteúdo de arquivo) para
que a mesma imagem não caia em treino e teste ao mesmo tempo; ver
listar_imagens().

Saída: dados/processados/divisao_trashnet.csv com as colunas
    caminho  (relativo à raiz do repositório)
    classe   (nome usado no projeto: papel, plastico, ...)
    conjunto (treino | validacao | teste)

Executar a partir da raiz do repositório (depois de baixar_dados):
    python -m src.machine_learning.dividir_dados
"""

import csv
import hashlib
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.configuracao import CLASSES_TRASHNET

SEED = 42
PROPORCAO_VALIDACAO = 0.15
PROPORCAO_TESTE = 0.15

RAIZ = Path(__file__).resolve().parents[2]
ORIGEM = RAIZ / "dados" / "brutos" / "trashnet"
SAIDA = RAIZ / "dados" / "processados" / "divisao_trashnet.csv"


def listar_imagens() -> tuple[list[str], list[str], list[str]]:
    """Devolve (caminhos, classes, descartes).

    Arquivos com conteúdo idêntico são agrupados. Se todas as cópias têm a
    mesma classe, fica só uma. Se as classes divergem (o TrashNet tem a mesma
    foto em "glass" e em "plastic", por exemplo), todas são descartadas,
    porque não há como saber qual rótulo está certo.
    """
    grupos: dict[str, list[tuple[str, str]]] = {}
    for pasta_en, classe in sorted(CLASSES_TRASHNET.items()):
        for imagem in sorted((ORIGEM / pasta_en).glob("*.jpg")):
            digest = hashlib.sha256(imagem.read_bytes()).hexdigest()
            grupos.setdefault(digest, []).append((imagem.relative_to(RAIZ).as_posix(), classe))

    caminhos, classes, descartes = [], [], []
    for copias in grupos.values():
        rotulos = {classe for _, classe in copias}
        if len(rotulos) > 1:
            descartes.append("rótulos conflitantes: " + ", ".join(f"{c} ({y})" for c, y in copias))
            continue
        if len(copias) > 1:
            descartes.append(f"duplicata: mantido {copias[0][0]}, removido(s) " + ", ".join(c for c, _ in copias[1:]))
        caminhos.append(copias[0][0])
        classes.append(copias[0][1])
    return caminhos, classes, descartes


def main() -> None:
    caminhos, classes, descartes = listar_imagens()
    if not caminhos:
        raise SystemExit("Nenhuma imagem encontrada. Rode antes: python -m src.machine_learning.baixar_dados")

    if descartes:
        print(f"{len(descartes)} grupo(s) de imagens idênticas tratados:")
        for d in descartes:
            print(f"  {d}")

    resto_x, teste_x, resto_y, teste_y = train_test_split(
        caminhos, classes, test_size=PROPORCAO_TESTE, stratify=classes, random_state=SEED
    )
    treino_x, val_x, treino_y, val_y = train_test_split(
        resto_x,
        resto_y,
        test_size=PROPORCAO_VALIDACAO / (1 - PROPORCAO_TESTE),
        stratify=resto_y,
        random_state=SEED,
    )

    linhas = (
        [(c, y, "treino") for c, y in zip(treino_x, treino_y)]
        + [(c, y, "validacao") for c, y in zip(val_x, val_y)]
        + [(c, y, "teste") for c, y in zip(teste_x, teste_y)]
    )
    linhas.sort()

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    with SAIDA.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["caminho", "classe", "conjunto"])
        escritor.writerows(linhas)

    contagem = Counter((y, conj) for _, y, conj in linhas)
    print(f"\n{'classe':<9} {'treino':>7} {'valid.':>7} {'teste':>7} {'total':>7}")
    for classe in sorted(set(classes)):
        t, v, s = (contagem[(classe, c)] for c in ("treino", "validacao", "teste"))
        print(f"{classe:<9} {t:>7} {v:>7} {s:>7} {t + v + s:>7}")
    t, v, s = len(treino_x), len(val_x), len(teste_x)
    print(f"{'TOTAL':<9} {t:>7} {v:>7} {s:>7} {t + v + s:>7}")
    print(f"\nDivisão salva em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
