# Plano de trabalho — EcoTriagem

Divisão de tarefas da equipe, com o que cada integrante precisa entregar, onde no
repositório e como isso pesa na nota (enunciado, seção 10).

| Integrante | Área | Critérios do enunciado | Pontos diretos |
|---|---|---|---|
| **Matteus** | Visão Computacional & Machine Learning | 4.2 ML, 4.3 Visão | 30 |
| **Andrey** | Processamento de Linguagem Natural | 4.4 PLN (+ reforça 4.2 ML) | 15 |
| **Caio** | Agente de decisão & busca | 4.1 Agentes, busca, heurísticas | 10 |
| **Frederico** | Integração e Interface Web | 4.5 Integração, reprodutibilidade | 10 + 5 |
| **Edvaldo** | Documentação, artigo e README | Problema e arquitetura, artigo | 10 + 10 |

Experimentos e avaliação (10 pts) é responsabilidade de todos: cada módulo gera suas
métricas e gráficos em `resultados/`.

---

## O que já está pronto

- Estrutura de pastas do enunciado (seção 7).
- **Vocabulário comum** em `src/configuracao.py`: classes, intenções e limiares de confiança.
- **Contratos entre módulos** em `src/integracao/contratos.py`: o formato exato que cada módulo recebe e devolve.
- **Protótipo integrado** (`python app.py`): imagem + pergunta → resposta, rodando com *stubs* (versões provisórias) de Visão e PLN.
- **Dataset TrashNet** baixado e dividido: `python -m src.machine_learning.baixar_dados` e
  `python -m src.machine_learning.dividir_dados` → `dados/processados/divisao_trashnet.csv`
  (2.521 imagens; 1.764 treino / 378 validação / 379 teste).

### Decisões já tomadas

| Item | Definição |
|---|---|
| Classes de resíduo | `papel`, `papelao`, `plastico`, `vidro`, `metal`, `lixo` (rejeito) |
| Intenções do PLN | `identificar`, `como_descartar`, `e_reciclavel`, `cor_lixeira`, `como_preparar`, `ponto_coleta`, `fora_do_escopo` |
| Limiar da visão | abaixo de 0,6 → pedir outra foto |
| Limiar do PLN | abaixo de 0,5 → pedir para reformular |
| Cores das lixeiras | Resolução CONAMA nº 275/2001 |
| Python | 3.13.5 |

---

## Regras para todos

1. **Não mudar a assinatura nem o tipo de retorno das funções dos stubs.** Cada um troca o
   *conteúdo* da sua função; o resto do sistema continua funcionando.
2. **Mudança em `configuracao.py` ou `contratos.py` só combinando com a equipe** (e avisando o Frederico).
3. **Uma branch por tarefa** (ex.: `feat/treino-visao`) e Pull Request para a `main`. Nada direto na `main`.
4. **Rodar `pytest` antes de abrir o PR.** Teste quebrado não entra.
5. **Resultados reais.** Todo número que for para o artigo precisa sair de um script do repositório
   e estar salvo em `resultados/`. O enunciado proíbe resultados inventados.
6. **Seed fixa (42)** em tudo que tem aleatoriedade.
7. **Nada de senhas, tokens ou fotos com pessoas** no repositório.

---

## Matteus — Visão Computacional & Machine Learning

**Objetivo:** substituir o stub de `src/visao_computacional/classificador.py` por um modelo
treinado no TrashNet, com avaliação completa. É o **caminho crítico** do projeto (a etapa mais
longa), então deve começar primeiro.

**Contrato a respeitar:**
```python
def classificar_imagem(imagem: PIL.Image.Image) -> ResultadoVisao
# ResultadoVisao(classe="plastico", confianca=0.87, top3=[("plastico", 0.87), ...])
```

### Tarefas

1. **Análise exploratória** (`notebooks/`): amostras por classe, tamanhos, distribuição das classes.
   A classe `lixo` tem só 137 imagens — o desbalanceamento precisa aparecer no artigo.
2. **Baseline simples** (para comparação): extrair características (histograma de cor, HOG) e treinar
   um SVM ou Regressão Logística (scikit-learn). Serve para mostrar que a CNN ganha *de quanto*.
3. **Modelo principal — transfer learning:** MobileNetV2 ou EfficientNet-B0 pré-treinado no ImageNet
   (PyTorch/torchvision). Roda em CPU ou no Colab gratuito.
   - Ler os conjuntos do `divisao_trashnet.csv` (nunca refazer a divisão).
   - Pré-processamento: redimensionar para 224×224 e normalizar no padrão do ImageNet.
   - *Data augmentation* **só no treino** (rotação, *flip*, brilho/contraste).
   - Tratar o desbalanceamento com pesos por classe na função de perda.
   - Treinar primeiro só a última camada; depois fazer *fine-tuning* das camadas finais.
   - Escolher o melhor modelo pelo **F1-macro na validação**.
   - Salvar em `modelos/` (fora do git) — o script de treino deve conseguir regenerá-lo.
4. **Avaliação no conjunto de teste** (uma única vez, no final):
   - acurácia, precisão, recall e F1 por classe, **F1-macro**;
   - matriz de confusão;
   - curvas de perda/acurácia por época;
   - salvar em `resultados/metricas/` (JSON/CSV) e `resultados/graficos/` (PNG).
5. **Análise do limiar de confiança:** curva de "% de previsões aceitas × acurácia das aceitas"
   para diferentes limiares. Isso justifica (ou corrige) o 0,6 usado pelo Caio.
6. **Teste no mundo real:** avaliar o modelo nas fotos tiradas pela equipe (ver seção *Fotos da equipe*).
   É esperado que o desempenho caia — medir e discutir isso vale nota em "erros e limitações".
7. **Integrar:** implementar `classificar_imagem` carregando o modelo **uma única vez** (não a cada chamada).
8. Adicionar as dependências usadas (`torch`, `torchvision`, `matplotlib`…) ao `requirements.txt`.

### Entregáveis
- `src/machine_learning/treinar.py` e `src/machine_learning/avaliar.py` (ou nomes equivalentes).
- Classificador real em `src/visao_computacional/classificador.py`.
- Tabela comparativa baseline × CNN, matriz de confusão, curvas e análise de limiar.
- Texto para o artigo: dataset, preparação, modelo, treino e resultados de visão.

---

## Andrey — Processamento de Linguagem Natural

**Objetivo:** substituir o stub de `src/pln/intencoes.py` (que hoje só procura palavras-chave e
**não conta como PLN** para o professor) por um classificador de intenções treinado e avaliado.

**Contrato a respeitar:**
```python
def interpretar_pergunta(texto: str) -> ResultadoPLN
# ResultadoPLN(intencao="como_descartar", confianca=0.92, texto_original="onde jogo isso?")
```

### Tarefas

1. **Criar o dataset de frases** em `dados/pln/intencoes.csv` (colunas `texto,intencao`):
   - 40 a 60 frases por intenção, cobrindo as 7 intenções;
   - variar o jeito de falar: formal, informal, gírias, sem acento, com erro de digitação,
     frases curtas ("e isso?", "reciclável?");
   - para `fora_do_escopo`: saudações, perguntas de outros assuntos, frases sem sentido.
   - **Dica:** pedir para outros integrantes escreverem as frases do conjunto de *teste*. Assim o
     teste não fica viciado no jeito de escrever de quem montou o treino.
2. **Pré-processamento:** minúsculas, remoção de acentos e pontuação, tokenização. Cuidado com
   *stopwords*: "não" muda o sentido e não pode ser removida. Opcional: radicalização (NLTK RSLP)
   ou lematização (spaCy `pt_core_news_sm`).
3. **Comparar pelo menos dois modelos:**
   - baseline: as palavras-chave do stub atual;
   - TF-IDF + Regressão Logística ou LinearSVC (scikit-learn) — aprendizado supervisionado;
   - opcional: *embeddings* de sentenças multilíngues (sentence-transformers) + classificador.
4. **Avaliação:** acurácia, F1-macro, matriz de confusão e exemplos de erros. Avaliar também o
   limiar de 0,5 para detectar `fora_do_escopo`.
5. **Integrar:** implementar `interpretar_pergunta` carregando o modelo de `modelos/` uma única vez.
6. **Extra (se sobrar tempo):** extrair o objeto citado na frase ("garrafa", "pilha", "lâmpada").
   Isso permite ao Caio tratar resíduos especiais que a visão não reconhece (pilhas, eletrônicos).

### Entregáveis
- `dados/pln/intencoes.csv` (versionado — é pequeno e foi criado pela equipe).
- Script de treino/avaliação em `src/pln/` e modelo salvo em `modelos/`.
- Tabela comparativa dos modelos, matriz de confusão e exemplos de entrada/saída.
- Texto para o artigo: dataset de intenções, pré-processamento, modelos e resultados de PLN.

---

## Caio — Agente de decisão & busca

**Objetivo:** transformar `src/decisao/regras.py` (versão inicial que já funciona) num
componente de decisão completo e implementar **busca** com função real no sistema. O enunciado
exige que o conceito esteja *funcionando*, não só citado.

**Contrato a respeitar:**
```python
def decidir(visao: ResultadoVisao, pln: ResultadoPLN) -> Decisao
# Decisao(resposta="...", acao="responder", lixeira="vermelha (plástico)")
```

### Tarefas

1. **Documentar o sistema de regras** como tabela (condição → ação) para o artigo, e
   enquadrar teoricamente: agente reativo baseado em regras que percebe (visão + PLN) e age (resposta).
2. **Heurísticas para confiança intermediária**, usando o `top3`. Exemplo: se a confiança está
   abaixo de 0,6 mas as duas classes mais prováveis vão para a **mesma lixeira** (papel e papelão →
   azul), dá para responder mesmo assim. Isso reduz pedidos de nova foto sem errar a lixeira.
3. **Casos especiais:** papel engordurado, vidro quebrado, embalagens multicamada (tipo longa vida),
   resíduos que o modelo não conhece (pilhas, eletrônicos) — em parceria com o Andrey, se ele
   implementar a extração do objeto citado.
4. **Busca do ponto de coleta** (intenção `ponto_coleta`, hoje com um `TODO` no código):
   - montar um grafo de Maceió (bairros ou pontos de referência como nós, distâncias como arestas)
     com ecopontos/pontos de entrega. Pode ser simulado ou, se acharem, com endereços reais de
     ecopontos (citar a fonte);
   - implementar **A\*** com heurística de distância em linha reta;
   - comparar com BFS e busca gulosa: custo do caminho, nós expandidos e tempo → tabela para o artigo;
   - considerar o tipo de resíduo (nem todo ponto aceita tudo).
5. **Combinar com o Frederico o contrato da busca**. Sugestão:
   ```python
   def buscar_ponto_coleta(origem: str, classe: str) -> RotaColeta
   # RotaColeta(destino="Ecoponto X", caminho=["Farol", "Pinheiro", ...], distancia_km=3.2, nos_expandidos=7)
   ```
6. **Testes** para as regras novas e para a busca (casos com caminho conhecido).

### Entregáveis
- `src/decisao/regras.py` expandido e `src/decisao/busca.py` (A\*, BFS, gulosa).
- Dados do grafo (ex.: `dados/mapa/` em JSON/CSV, versionado).
- Tabela de regras, tabela comparativa dos algoritmos de busca.
- Texto para o artigo: fundamentação (agentes, A\*, heurísticas) e resultados da busca.

---

## Frederico — Integração e Interface Web

**Objetivo:** garantir que Visão, PLN e Decisão funcionem como **um único fluxo** (requisito 4.5 —
"três códigos isolados não serão aceitos"), com uma interface que demonstre isso, e avaliar o
sistema de ponta a ponta.

### Tarefas

1. **Publicar os contratos** (push + PR na `main`) e manter `contratos.py` e `pipeline.py`
   como referência para a equipe.
2. **Preparar a pipeline para os modelos reais:**
   - carregar todos os modelos uma vez, na inicialização do app;
   - mensagem clara quando um modelo ainda não foi treinado (em vez de quebrar);
   - normalizar imagens antes da visão: PNG com transparência → RGB, rotação EXIF de fotos de
     celular, imagens muito grandes.
3. **Interface em formato de conversa** (Gradio):
   - envio de foto ou captura pela webcam/câmera do celular;
   - chat com várias perguntas sobre a mesma foto, sem reprocessar a imagem;
   - painel com cada etapa: material e barras de confiança, intenção reconhecida, regra aplicada;
   - galeria de exemplos prontos para clicar (demonstração reproduzível);
   - quando a busca do Caio ficar pronta: entrada da localização e exibição da rota.
4. **Avaliação de ponta a ponta** (`src/integracao/avaliar_sistema.py`):
   - conjunto de casos (imagem de teste + pergunta + resposta esperada);
   - **% de recomendações de descarte corretas** no fluxo completo;
   - **em qual módulo cada erro nasceu** (visão, PLN ou decisão);
   - **tempo de resposta** de cada etapa.
5. **Registro de exemplos:** modo que salva interações (imagem, pergunta, saída de cada etapa,
   resposta) em `resultados/exemplos/` — o enunciado pede exemplos de entrada e saída.
6. **Reprodutibilidade:** garantir que qualquer um clone o repositório e rode tudo com poucos comandos.
7. **Extra:** publicar a demo no Hugging Face Spaces (gratuito) para ter um link funcionando.

### Entregáveis
- `app.py` com a interface final e `src/integracao/` completo.
- Script e tabela da avaliação de ponta a ponta.
- Exemplos de interações em `resultados/exemplos/`.
- Texto para o artigo: integração, interface e avaliação do sistema; diagrama de arquitetura
  (junto com o Edvaldo).

---

## Edvaldo — Documentação, artigo e README

**Objetivo:** transformar o trabalho da equipe no artigo (até 6 páginas, IEEE) e num README
completo, e conferir se tudo que o enunciado pede foi entregue.

### Tarefas

1. **Etapa 1 do enunciado — definição do problema:** problema, público-alvo, objetivos e
   contribuições. Preencher as seções correspondentes do README.
2. **Diagrama de arquitetura** (com o Frederico): imagem → Visão → Decisão ← PLN ← pergunta.
   Vai no README e no artigo.
3. **Overleaf:** criar o projeto no template IEEE Conference (duas colunas), com as seções do
   enunciado, e compartilhar com a equipe. Fonte do LaTeX também em `artigo/`.
4. **Trabalhos relacionados e referências** (verificáveis e citadas no texto). Pontos de partida:
   artigo original do TrashNet (Thung & Yang, 2016), datasets RealWaste e TACO, MobileNetV2,
   Resolução CONAMA 275/2001, Política Nacional de Resíduos Sólidos (Lei 12.305/2010).
5. **Coletar os resultados de cada módulo** e conferir se os números do artigo batem com os
   arquivos em `resultados/`.
6. **Coordenar as fotos da equipe** (ver abaixo).
7. **Teste de reprodutibilidade:** seguir o README do zero, num computador "limpo", e apontar o que falhar.
8. **Checklist final da entrega** (seção 12 do enunciado): link do GitHub, link do Overleaf com
   permissão de visualização, PDF final de até 6 páginas, demonstração ou vídeo, **nomes
   completos e responsabilidades de cada integrante**.
9. Apoiar o Matteus nos gráficos e métricas, que é a área com mais volume de trabalho.

### Entregáveis
- README completo, diagrama de arquitetura, artigo no Overleaf + PDF.

---

## Fotos da equipe (todos)

Cada integrante tira **10 a 20 fotos** de resíduos reais de casa, distribuídas entre as 6 classes.
Esse conjunto testa o modelo fora do TrashNet (que tem fundo branco e iluminação controlada).

- Uma foto por objeto, com fundos e iluminação variados.
- **Sem pessoas, rostos, documentos ou endereços** nas fotos.
- Nomear como `classe_nome_numero.jpg` (ex.: `plastico_andrey_03.jpg`) e enviar ao Matteus.
- Usar **somente para teste**, nunca para treino.

---

## Dependências entre as partes

```text
Matteus (modelo de visão) ──┐
                            ├──> Frederico (pipeline + interface + avaliação ponta a ponta)
Andrey  (modelo de PLN)  ───┤                         │
                            │                         v
Caio    (regras + busca) ───┘          Edvaldo (README + artigo)
```

Por causa dos stubs, **ninguém precisa esperar ninguém para começar**. A avaliação de ponta a
ponta e os resultados do artigo dependem dos modelos reais estarem prontos.

## Cronograma (enunciado, seção 11)

| Semana | Matteus | Andrey | Caio | Frederico | Edvaldo |
|---|---|---|---|---|---|
| 1 | Análise exploratória | Lista de frases | Tabela de regras | Contratos publicados | Problema, objetivos, diagrama |
| 2 | Baseline | Dataset de frases completo | Grafo do mapa | Pipeline robusta | Overleaf + trabalhos relacionados |
| 3 | Treino da CNN | Treino dos classificadores | A\*, BFS, gulosa | Interface em conversa | Metodologia (rascunho) |
| 4 | Integração do modelo | Integração do modelo | Integração da busca | Integração de tudo | README em andamento |
| 5 | Avaliação + fotos da equipe | Avaliação | Comparação das buscas | Avaliação ponta a ponta | Resultados e discussão |
| 6 | Texto do artigo | Texto do artigo | Texto do artigo | Texto do artigo + demo | Revisão final e entrega |
