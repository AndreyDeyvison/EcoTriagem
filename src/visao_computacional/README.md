# Módulo de Visão Computacional & ML — classificação de resíduos

Decide **de que material é** o resíduo fotografado. Recebe a imagem e devolve a
classe prevista com a confiança e o top 3, no contrato `ResultadoVisao`.

```python
from PIL import Image
from src.visao_computacional.classificador import classificar_imagem

classificar_imagem(Image.open("foto.jpg"))
# ResultadoVisao(classe='vidro', confianca=0.93, top3=[('vidro', 0.93), ('plastico', 0.04), ...])
```

## Como rodar

```bash
python -m src.machine_learning.baixar_dados          # TrashNet (~40 MB)
python -m src.machine_learning.analise_exploratoria  # distribuição das classes
python -m src.machine_learning.treinar_baseline      # baseline clássico (~1 min)
python -m src.machine_learning.treinar               # MobileNetV2 (~5 min no MPS)
python -m src.machine_learning.avaliar               # métricas finais no teste
pytest testes/test_visao.py
```

`treinar.py` precisa rodar antes de `avaliar.py`. O dataset e os modelos são
ignorados pelo git: quem clona precisa baixar e treinar. Até lá a classificação
usa um stub e avisa no console, sem quebrar o sistema.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `src/machine_learning/dados_visao.py` | lê a divisão fixa e calcula os pesos por classe |
| `src/machine_learning/analise_exploratoria.py` | distribuição, dimensões e modos de cor |
| `src/machine_learning/treinar_baseline.py` | histograma HSV + HOG e SVM RBF |
| `src/machine_learning/treinar.py` | MobileNetV2 em duas fases, escolha pelo F1-macro |
| `src/machine_learning/avaliar.py` | teste, matriz de confusão, limiar e fotos da equipe |
| `src/visao_computacional/classificador.py` | `classificar_imagem` — o que o resto do sistema chama |

## Resultados

Conjunto de teste de 379 imagens, usado uma única vez, depois de o modelo já
estar escolhido pelo F1-macro na validação:

| | acurácia | F1-macro |
|---|---|---|
| baseline clássico (HSV + HOG e SVM RBF) | 0,710 | 0,699 |
| **MobileNetV2 (transferência de aprendizado)** | **0,894** | **0,881** |
| ganho da CNN | +18,5 pp | +18,1 pp |

Por classe (CNN): metal 0,934 · papelão 0,941 · papel 0,925 · vidro 0,877 ·
plástico 0,841 · **lixo 0,766** — a classe minoritária continua sendo a pior,
mesmo com os pesos na função de perda.

Números completos em `resultados/metricas/` e gráficos em `resultados/graficos/`.

## Três pontos de atenção para a equipe

**O limiar de 0,6 está bem escolhido** (para o módulo de decisão). A varredura
confirma a escolha: a 0,6 o sistema aceita 91,6% das fotos e acerta 93,4% das
aceitas, contra 89,4% sem limiar nenhum. Subir para 0,75 levaria a acurácia a
96,5%, mas pediria nova foto em 17,7% dos casos. Diferente do limiar do PLN,
este se paga. Dados em `resultados/metricas/analise_limiar_visao.csv`.

**As fotos da equipe ainda não foram avaliadas.** `avaliar.py` já tem o suporte
pronto: basta criar `dados/fotos_equipe/` com arquivos no padrão
`classe_nome_numero.jpg` (ex.: `plastico_andrey_03.jpg`) e rodar de novo —
ele gera `resultados/metricas/avaliacao_mundo_real.json` com a queda de
desempenho em relação ao TrashNet. Sem a pasta, a etapa é pulada com aviso.

**O stub mudou de comportamento na prática.** Antes do modelo treinado,
`classificar_imagem` devolvia sempre "plástico, 0,87". Agora devolve a previsão
real, que para imagens sintéticas tem confiança baixa. Um teste da interface
dependia disso sem querer (ver `testes/test_app.py`) e foi ajustado para fixar o
resultado de visão em vez de depender do modelo.
