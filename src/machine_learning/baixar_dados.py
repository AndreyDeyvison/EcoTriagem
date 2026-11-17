"""Baixa o dataset TrashNet para dados/brutos/trashnet/.

Fonte: https://github.com/garythung/trashnet (licença MIT)
Thung, G.; Yang, M. "Classification of Trash for Recyclability Status", 2016.
Versão usada: dataset-resized.zip (imagens 512x384), 2.527 imagens em 6 classes.

Executar a partir da raiz do repositório:
    python -m src.machine_learning.baixar_dados
"""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

from src.configuracao import CLASSES_TRASHNET

URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
SHA256 = "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
CONTAGEM_ESPERADA = {
    "cardboard": 403,
    "glass": 501,
    "metal": 410,
    "paper": 594,
    "plastic": 482,
    "trash": 137,
}

RAIZ = Path(__file__).resolve().parents[2]
DESTINO = RAIZ / "dados" / "brutos" / "trashnet"


def contar_imagens(pasta: Path) -> dict[str, int]:
    return {c: len(list((pasta / c).glob("*.jpg"))) for c in CONTAGEM_ESPERADA}


def baixar() -> bytes:
    print(f"Baixando {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as resposta:
        conteudo = resposta.read()
    obtido = hashlib.sha256(conteudo).hexdigest()
    if obtido != SHA256:
        raise RuntimeError(f"SHA-256 não confere (esperado {SHA256}, obtido {obtido}); arquivo alterado ou corrompido.")
    return conteudo


def extrair(conteudo: bytes, destino: Path) -> None:
    """Extrai só as imagens dataset-resized/<classe>/<arquivo>.jpg, ignorando o resto."""
    with zipfile.ZipFile(io.BytesIO(conteudo)) as arquivo_zip:
        for nome in arquivo_zip.namelist():
            partes = nome.split("/")
            if (
                len(partes) != 3
                or partes[0] != "dataset-resized"
                or partes[1] not in CLASSES_TRASHNET
                or not partes[2].endswith(".jpg")
                or partes[2] != Path(partes[2]).name
            ):
                continue
            alvo = destino / partes[1] / partes[2]
            alvo.parent.mkdir(parents=True, exist_ok=True)
            alvo.write_bytes(arquivo_zip.read(nome))


def main() -> None:
    if DESTINO.exists() and contar_imagens(DESTINO) == CONTAGEM_ESPERADA:
        print(f"Dataset já está completo em {DESTINO.relative_to(RAIZ)}; nada a fazer.")
        return

    extrair(baixar(), DESTINO)

    contagem = contar_imagens(DESTINO)
    if contagem != CONTAGEM_ESPERADA:
        raise RuntimeError(f"Contagem inesperada: {contagem}")
    for classe, n in contagem.items():
        print(f"  {CLASSES_TRASHNET[classe]:<9} {n:>4} imagens")
    print(f"Total: {sum(contagem.values())} imagens em {DESTINO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
