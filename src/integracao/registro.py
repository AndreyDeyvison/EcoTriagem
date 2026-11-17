"""Registro de exemplos de entrada e saída (imagem, pergunta, cada etapa, resposta).

Cada exemplo vira uma imagem .jpg e uma linha em exemplos.jsonl na pasta
escolhida. Usado pela avaliação de ponta a ponta e pela interface
(python app.py --registrar).
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from src.integracao.contratos import ResultadoPipeline

ARQUIVO_REGISTRO = "exemplos.jsonl"


def salvar_exemplo(
    pasta: Path,
    imagem: Image.Image,
    resultado: ResultadoPipeline,
    extras: dict | None = None,
) -> dict:
    """Salva a imagem e acrescenta o registro em pasta/exemplos.jsonl. Devolve o registro."""
    pasta.mkdir(parents=True, exist_ok=True)
    numero = sum(1 for _ in pasta.glob("exemplo_*.jpg")) + 1
    nome_imagem = f"exemplo_{numero:03d}.jpg"
    imagem.convert("RGB").save(pasta / nome_imagem, quality=90)

    registro = {
        "imagem": nome_imagem,
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "pergunta": resultado.pln.texto_original,
        "visao": asdict(resultado.visao),
        "pln": {"intencao": resultado.pln.intencao, "confianca": resultado.pln.confianca},
        "decisao": asdict(resultado.decisao),
        **(extras or {}),
    }
    with (pasta / ARQUIVO_REGISTRO).open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return registro
