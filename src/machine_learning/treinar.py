"""Treino do classificador de resíduos por transferência de aprendizado.

MobileNetV2 pré-treinada na ImageNet, ajustada ao TrashNet em duas fases:

1. **Extração de características** — o backbone fica congelado e só a cabeça
   classificadora é treinada, com taxa de aprendizado alta. A cabeça começa
   aleatória; treiná-la junto com o backbone destruiria os pesos da ImageNet
   logo nos primeiros gradientes.
2. **Ajuste fino** — os blocos finais do backbone são descongelados e treinados
   com taxa 10x menor e agendador cosseno. Só os blocos finais porque são os
   que carregam as características específicas do domínio; os iniciais
   detectam bordas e texturas que servem para qualquer imagem.

Escolha da arquitetura: MobileNetV2 (3,5 M de parâmetros) em vez da
EfficientNet-B0 por ser mais leve e estável no acelerador MPS do Mac, que é o
hardware disponível. O script detecta sozinho cuda, mps ou cpu.

Desbalanceamento: pesos inversamente proporcionais à frequência da classe na
função de perda (a classe `lixo` tem 95 imagens de treino contra 416 de papel).

O melhor modelo é escolhido pelo **F1-macro na validação** — não pela acurácia,
que numa base desbalanceada premia acertar as classes grandes.

Saídas:
    modelos/classificador_residuos.pt
    resultados/metricas/historico_treino_visao.csv
    resultados/graficos/curvas_treino_visao.png

Executar a partir da raiz do repositório:
    python -m src.machine_learning.treinar
"""

import argparse
import csv
import random
import time
from datetime import date

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from src.configuracao import PASTA_MODELOS
from src.machine_learning.dados_visao import (
    CLASSES_ORDENADAS,
    RAIZ,
    carregar_divisao,
    pesos_por_classe,
    por_conjunto,
    resumo,
)

SEED = 42

TAMANHO_ENTRADA = 224
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)

LOTE = 32
EPOCAS_FASE1 = 6
EPOCAS_FASE2 = 12
LR_FASE1 = 1e-3
LR_FASE2 = 1e-4
DECAIMENTO_PESOS = 1e-4
BLOCO_DESCONGELADO = 14  # features[14:] da MobileNetV2 entram no ajuste fino

ARQUIVO_MODELO = PASTA_MODELOS / "classificador_residuos.pt"
ARQUIVO_HISTORICO = RAIZ / "resultados" / "metricas" / "historico_treino_visao.csv"
ARQUIVO_CURVAS = RAIZ / "resultados" / "graficos" / "curvas_treino_visao.png"

COLUNAS_HISTORICO = (
    "epoca", "fase", "lr", "perda_treino", "acuracia_treino",
    "perda_validacao", "acuracia_validacao", "f1_macro_validacao", "segundos",
)


def fixar_seeds(seed: int = SEED) -> None:
    """Toda fonte de aleatoriedade do treino, num só lugar."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def escolher_dispositivo() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def transformacoes(treino: bool) -> transforms.Compose:
    """Aumento de dados só no treino; validação e teste só redimensionam.

    Aplicar aumento na avaliação daria uma métrica que muda a cada execução e
    não corresponde ao que o usuário final recebe.
    """
    etapas = [transforms.Resize((TAMANHO_ENTRADA, TAMANHO_ENTRADA))]
    if treino:
        etapas += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        ]
    etapas += [transforms.ToTensor(), transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET)]
    return transforms.Compose(etapas)


class ConjuntoTrashNet(Dataset):
    """Imagens de um conjunto da divisão fixa, com o rótulo como índice."""

    def __init__(self, registros, treino: bool):
        self.registros = registros
        self.transformar = transformacoes(treino)

    def __len__(self) -> int:
        return len(self.registros)

    def __getitem__(self, indice: int):
        registro = self.registros[indice]
        with Image.open(registro.caminho) as imagem:
            imagem = imagem.convert("RGB")
            return self.transformar(imagem), registro.indice


def criar_carregador(registros, treino: bool, lote: int = LOTE) -> DataLoader:
    gerador = torch.Generator()
    gerador.manual_seed(SEED)
    return DataLoader(
        ConjuntoTrashNet(registros, treino),
        batch_size=lote,
        shuffle=treino,
        num_workers=0,  # determinístico e sem risco de travar no macOS
        generator=gerador if treino else None,
        drop_last=False,
    )


def construir_modelo() -> nn.Module:
    """MobileNetV2 da ImageNet com a última camada trocada pelas 6 classes."""
    modelo = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    entradas = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(entradas, len(CLASSES_ORDENADAS))
    return modelo


def congelar_backbone(modelo: nn.Module) -> None:
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False


def descongelar_blocos_finais(modelo: nn.Module, a_partir_de: int = BLOCO_DESCONGELADO) -> int:
    liberados = 0
    for indice, bloco in enumerate(modelo.features):
        if indice >= a_partir_de:
            for parametro in bloco.parameters():
                parametro.requires_grad = True
                liberados += parametro.numel()
    return liberados


def _passar_epoca(modelo, carregador, dispositivo, criterio, otimizador=None):
    """Uma passada completa. Com otimizador treina; sem otimizador, avalia."""
    treinando = otimizador is not None
    modelo.train(treinando)

    perda_total, verdadeiros, previstos = 0.0, [], []
    contexto = torch.enable_grad() if treinando else torch.no_grad()
    with contexto:
        for imagens, rotulos in carregador:
            imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)
            if treinando:
                otimizador.zero_grad(set_to_none=True)
            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)
            if treinando:
                perda.backward()
                otimizador.step()
            perda_total += perda.item() * imagens.size(0)
            previstos.extend(saidas.argmax(1).cpu().tolist())
            verdadeiros.extend(rotulos.cpu().tolist())

    return (
        perda_total / len(verdadeiros),
        accuracy_score(verdadeiros, previstos),
        f1_score(verdadeiros, previstos, average="macro", zero_division=0),
    )


def _salvar_checkpoint(modelo, epoca: int, f1: float, acuracia: float) -> None:
    """Guarda tudo que a inferência precisa para reconstruir o modelo sozinha."""
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arquitetura": "mobilenet_v2",
            "state_dict": {c: t.cpu() for c, t in modelo.state_dict().items()},
            "classes": list(CLASSES_ORDENADAS),  # índice da saída -> nome da classe
            "normalizacao": {"media": list(MEDIA_IMAGENET), "desvio": list(DESVIO_IMAGENET)},
            "tamanho_entrada": TAMANHO_ENTRADA,
            "epoca": epoca,
            "f1_macro_validacao": f1,
            "acuracia_validacao": acuracia,
            "seed": SEED,
            "gerado_em": date.today().isoformat(),
        },
        ARQUIVO_MODELO,
    )


def _graficos(historico: list[dict], epocas_fase1: int) -> None:
    epocas = [linha["epoca"] for linha in historico]
    figura, (perda, metrica) = plt.subplots(1, 2, figsize=(12, 4.5))

    perda.plot(epocas, [l["perda_treino"] for l in historico], marker="o", ms=3, label="treino")
    perda.plot(epocas, [l["perda_validacao"] for l in historico], marker="o", ms=3, label="validação")
    perda.set_ylabel("perda (entropia cruzada ponderada)")
    perda.set_title("Perda por época")

    metrica.plot(epocas, [l["acuracia_treino"] for l in historico], marker="o", ms=3, label="acurácia treino")
    metrica.plot(epocas, [l["acuracia_validacao"] for l in historico], marker="o", ms=3, label="acurácia validação")
    metrica.plot(epocas, [l["f1_macro_validacao"] for l in historico], marker="o", ms=3, label="F1-macro validação")
    metrica.set_ylabel("proporção")
    metrica.set_title("Acurácia e F1-macro por época")

    melhor = max(historico, key=lambda l: l["f1_macro_validacao"])
    for eixo in (perda, metrica):
        eixo.axvline(epocas_fase1 + 0.5, color="gray", linestyle=":", label="início do ajuste fino")
        eixo.axvline(melhor["epoca"], color="crimson", linestyle="--", label="melhor F1-macro")
        eixo.set_xlabel("época")
        eixo.grid(alpha=0.3)
        eixo.legend(fontsize=8)

    figura.suptitle("Visão — treino da MobileNetV2 no TrashNet")
    figura.tight_layout()
    figura.savefig(ARQUIVO_CURVAS, dpi=160)
    plt.close(figura)


def main() -> None:
    analisador = argparse.ArgumentParser(description="Treina o classificador de resíduos.")
    analisador.add_argument("--epocas-fase1", type=int, default=EPOCAS_FASE1)
    analisador.add_argument("--epocas-fase2", type=int, default=EPOCAS_FASE2)
    analisador.add_argument("--lote", type=int, default=LOTE)
    argumentos = analisador.parse_args()

    fixar_seeds()
    ARQUIVO_HISTORICO.parent.mkdir(parents=True, exist_ok=True)
    ARQUIVO_CURVAS.parent.mkdir(parents=True, exist_ok=True)

    dispositivo = escolher_dispositivo()
    grupos = por_conjunto(carregar_divisao())
    print(f"Dispositivo: {dispositivo}\n")
    print(resumo(carregar_divisao()))

    carregador_treino = criar_carregador(grupos["treino"], treino=True, lote=argumentos.lote)
    carregador_validacao = criar_carregador(grupos["validacao"], treino=False, lote=argumentos.lote)

    pesos = pesos_por_classe(grupos["treino"])
    print("\nPesos por classe na função de perda:")
    for classe, peso in zip(CLASSES_ORDENADAS, pesos):
        print(f"  {classe:<9} {peso:.3f}")

    modelo = construir_modelo().to(dispositivo)
    criterio = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32, device=dispositivo))

    historico: list[dict] = []
    melhor_f1 = -1.0
    epoca = 0

    # ---- Fase 1: só a cabeça classificadora ----
    congelar_backbone(modelo)
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"\n=== Fase 1: extração de características ({treinaveis:,} parâmetros treináveis) ===")
    otimizador = torch.optim.AdamW(
        [p for p in modelo.parameters() if p.requires_grad], lr=LR_FASE1, weight_decay=DECAIMENTO_PESOS
    )

    for _ in range(argumentos.epocas_fase1):
        epoca += 1
        inicio = time.perf_counter()
        perda_treino, acuracia_treino, _ = _passar_epoca(
            modelo, carregador_treino, dispositivo, criterio, otimizador
        )
        perda_validacao, acuracia_validacao, f1_validacao = _passar_epoca(
            modelo, carregador_validacao, dispositivo, criterio
        )
        duracao = time.perf_counter() - inicio
        historico.append(
            {
                "epoca": epoca, "fase": "extracao", "lr": LR_FASE1,
                "perda_treino": round(perda_treino, 4), "acuracia_treino": round(acuracia_treino, 4),
                "perda_validacao": round(perda_validacao, 4),
                "acuracia_validacao": round(acuracia_validacao, 4),
                "f1_macro_validacao": round(f1_validacao, 4), "segundos": round(duracao, 1),
            }
        )
        marca = ""
        if f1_validacao > melhor_f1:
            melhor_f1 = f1_validacao
            _salvar_checkpoint(modelo, epoca, f1_validacao, acuracia_validacao)
            marca = "  <- melhor"
        print(
            f"época {epoca:>2} | perda {perda_treino:.4f} / {perda_validacao:.4f} | "
            f"acurácia val {acuracia_validacao:.4f} | F1-macro val {f1_validacao:.4f} | "
            f"{duracao:.0f}s{marca}"
        )

    # ---- Fase 2: ajuste fino dos blocos finais ----
    liberados = descongelar_blocos_finais(modelo)
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"\n=== Fase 2: ajuste fino a partir de features[{BLOCO_DESCONGELADO}] "
          f"({liberados:,} parâmetros liberados, {treinaveis:,} treináveis) ===")
    otimizador = torch.optim.AdamW(
        [p for p in modelo.parameters() if p.requires_grad], lr=LR_FASE2, weight_decay=DECAIMENTO_PESOS
    )
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=argumentos.epocas_fase2)

    for _ in range(argumentos.epocas_fase2):
        epoca += 1
        lr_atual = otimizador.param_groups[0]["lr"]
        inicio = time.perf_counter()
        perda_treino, acuracia_treino, _ = _passar_epoca(
            modelo, carregador_treino, dispositivo, criterio, otimizador
        )
        perda_validacao, acuracia_validacao, f1_validacao = _passar_epoca(
            modelo, carregador_validacao, dispositivo, criterio
        )
        agendador.step()
        duracao = time.perf_counter() - inicio
        historico.append(
            {
                "epoca": epoca, "fase": "ajuste_fino", "lr": round(lr_atual, 8),
                "perda_treino": round(perda_treino, 4), "acuracia_treino": round(acuracia_treino, 4),
                "perda_validacao": round(perda_validacao, 4),
                "acuracia_validacao": round(acuracia_validacao, 4),
                "f1_macro_validacao": round(f1_validacao, 4), "segundos": round(duracao, 1),
            }
        )
        marca = ""
        if f1_validacao > melhor_f1:
            melhor_f1 = f1_validacao
            _salvar_checkpoint(modelo, epoca, f1_validacao, acuracia_validacao)
            marca = "  <- melhor"
        print(
            f"época {epoca:>2} | perda {perda_treino:.4f} / {perda_validacao:.4f} | "
            f"acurácia val {acuracia_validacao:.4f} | F1-macro val {f1_validacao:.4f} | "
            f"{duracao:.0f}s{marca}"
        )

    with ARQUIVO_HISTORICO.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_HISTORICO)
        escritor.writeheader()
        escritor.writerows(historico)
    _graficos(historico, argumentos.epocas_fase1)

    melhor = max(historico, key=lambda linha: linha["f1_macro_validacao"])
    print(f"\nMelhor época: {melhor['epoca']} ({melhor['fase']}) — "
          f"F1-macro validação {melhor['f1_macro_validacao']:.4f}")
    print(f"Modelo salvo em {ARQUIVO_MODELO.relative_to(RAIZ)}")
    print(f"Histórico em {ARQUIVO_HISTORICO.relative_to(RAIZ)}")
    print(f"Curvas em {ARQUIVO_CURVAS.relative_to(RAIZ)}")
    print("\nPróximo passo: python -m src.machine_learning.avaliar")


if __name__ == "__main__":
    main()
