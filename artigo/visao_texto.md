# Visão Computacional & ML — texto para o artigo

Rascunho das seções de visão, para o Edvaldo adaptar ao template IEEE. Todo
número sai de um script do repositório e está salvo em `resultados/` — a fonte
de cada um está indicada entre colchetes.

---

## Metodologia — Visão Computacional

### Conjunto de dados

Usamos o **TrashNet** [Thung e Yang, 2016], com 2.527 fotografias de resíduos
em seis classes, tiradas sobre fundo branco com iluminação controlada. Após
remover arquivos de conteúdo idêntico e os pares com rótulos conflitantes (a
mesma fotografia aparece em mais de uma pasta no dataset original), restaram
**2.521 imagens**, todas 512×384 em RGB [`resultados/metricas/distribuicao_trashnet.json`].

A divisão em treino, validação e teste foi feita **uma única vez**, de forma
estratificada e com semente fixa 42, e gravada em
`dados/processados/divisao_trashnet.csv`: 1.764 imagens de treino, 378 de
validação e 379 de teste. Todos os scripts leem essa divisão em vez de
refazê-la, o que garante que baseline e CNN sejam comparados exatamente sobre
as mesmas imagens.

**Tabela I — Distribuição das classes.**
[`resultados/metricas/distribuicao_trashnet.csv`, figura
`resultados/graficos/distribuicao_classes_trashnet.png`]

| Classe | Treino | Validação | Teste | Total | % |
|---|---|---|---|---|---|
| papel | 416 | 89 | 89 | 594 | 23,6 |
| vidro | 348 | 75 | 75 | 498 | 19,8 |
| plástico | 336 | 72 | 72 | 480 | 19,0 |
| metal | 287 | 61 | 61 | 409 | 16,2 |
| papelão | 282 | 60 | 61 | 403 | 16,0 |
| **lixo (rejeito)** | **95** | **21** | **21** | **137** | **5,4** |

O conjunto é desbalanceado: a classe `lixo` tem 137 imagens contra 594 de
`papel`, uma razão de 4,3×, e fica com apenas 21 imagens no teste. É a classe
semanticamente mais difícil — "rejeito" não é um material, é a ausência de
reciclabilidade — e a que tem menos exemplos, combinação que antecipa o pior
desempenho do sistema.

### Pré-processamento e aumento de dados

As imagens são redimensionadas para 224×224 e normalizadas com a média e o
desvio-padrão da ImageNet, que é o pré-processamento que os pesos
pré-treinados esperam. O **aumento de dados é aplicado somente ao treino**
(espelhamento horizontal e vertical, rotação de até 20°, e variação de brilho,
contraste e saturação); validação e teste passam apenas por redimensionamento e
normalização. Aplicar aumento na avaliação produziria uma métrica que muda a
cada execução e que não corresponde ao que o usuário recebe.

O espelhamento vertical é admissível aqui porque não existe "em pé" num resíduo
fotografado de cima — diferente de tarefas como reconhecimento de texto.

### Baseline clássico

Para medir o quanto a rede neural realmente contribui, implementamos um
baseline sem aprendizado profundo: **histograma HSV** (16 bins de matiz, 8 de
saturação, 8 de valor) concatenado com **HOG** — histograma de gradientes
orientados, em células de 16×16 com 9 bins de orientação e normalização L2 por
célula —, totalizando 608 características por imagem. O classificador é uma
**SVM com núcleo RBF** e `class_weight="balanced"`; o parâmetro C foi escolhido
no conjunto de validação (C = 10).

### Modelo principal: transferência de aprendizado

Adotamos a **MobileNetV2** pré-treinada na ImageNet, escolhida em vez da
EfficientNet-B0 por ser mais leve (3,5 M de parâmetros) e estável no acelerador
MPS do hardware disponível. A última camada densa foi substituída por uma de
seis saídas.

O treino tem duas fases, com um motivo: a cabeça classificadora começa
aleatória, e treiná-la junto com o backbone faria os primeiros gradientes, muito
grandes, destruírem os pesos aprendidos na ImageNet.

1. **Extração de características** (6 épocas, taxa 10⁻³): backbone congelado,
   apenas 7.686 parâmetros treináveis.
2. **Ajuste fino** (12 épocas, taxa 10⁻⁴ com agendador cosseno): os blocos
   finais do backbone são descongelados (1,68 M de parâmetros). Apenas os
   finais, porque são os que carregam características específicas do domínio;
   os iniciais detectam bordas e texturas que servem a qualquer imagem.

O desbalanceamento é tratado com **pesos inversamente proporcionais à
frequência da classe** na entropia cruzada: a classe `lixo` recebe peso 2,45
contra 0,56 de `papel`. O melhor modelo é escolhido pelo **F1-macro na
validação**, e não pela acurácia, que numa base desbalanceada premia acertar as
classes grandes. Toda fonte de aleatoriedade usa semente 42.

---

## Resultados — Visão Computacional

O treino convergiu na época 17, com F1-macro de 0,9037 na validação
[`resultados/metricas/historico_treino_visao.csv`, figura
`resultados/graficos/curvas_treino_visao.png`]. O efeito das duas fases é
visível na curva: a extração de características estaciona em torno de 0,79 de
F1-macro, e o ajuste fino leva o modelo a 0,90 — ou seja, **a maior parte do
ganho vem de adaptar o backbone, não de treinar a cabeça**.

**Tabela II — Baseline clássico contra CNN, conjunto de teste (379 imagens).**
[`resultados/metricas/comparacao_modelos_visao.csv`]

| | Acurácia | F1-macro |
|---|---|---|
| Baseline: HSV + HOG e SVM RBF | 0,710 | 0,699 |
| **MobileNetV2 (transferência de aprendizado)** | **0,894** | **0,881** |
| Ganho da CNN | +18,5 pp | +18,1 pp |

| Classe | Precisão | Revocação | F1 (CNN) | F1 (baseline) | n |
|---|---|---|---|---|---|
| papel | 0,887 | 0,966 | 0,925 | 0,831 | 89 |
| papelão | 0,966 | 0,918 | 0,941 | 0,855 | 61 |
| plástico | 0,879 | 0,806 | 0,841 | 0,594 | 72 |
| vidro | 0,901 | 0,853 | 0,877 | 0,627 | 75 |
| metal | 0,934 | 0,934 | 0,934 | 0,603 | 61 |
| lixo | 0,692 | 0,857 | 0,766 | 0,686 | 21 |

O ganho da CNN não é uniforme, e a distribuição é informativa. Onde a cor já
separava bem as classes — papel e papelão, de tonalidade característica — o
baseline vai bem e a CNN acrescenta pouco (+9 pontos de F1). Onde a cor engana
— plástico, vidro e metal são todos claros, translúcidos ou reflexivos — o
baseline fracassa e a CNN ganha de 25 a 33 pontos. É a evidência de que o
problema exige representação de forma e textura, não de cor.

**Figura — matriz de confusão** [`resultados/graficos/matriz_confusao_visao.png`].
Os erros se concentram em três pares fisicamente plausíveis: plástico e vidro
trocados entre si (11 casos), papelão lido como papel (5 casos) e plástico lido
como rejeito (5 casos). Nenhum desses erros é arbitrário: garrafas de plástico
transparente e de vidro têm a mesma aparência numa fotografia, e papelão fino
é visualmente papel. A classe `lixo` tem a menor precisão (0,692), o que
significa que o modelo a usa como "categoria de descarte" para itens dos quais
não tem certeza — comportamento conservador e, do ponto de vista do sistema,
preferível ao inverso.

### Análise do limiar de confiança

O sistema adota um limiar de 0,6: abaixo dele, em vez de arriscar uma
recomendação de descarte errada, pede outra fotografia. Varremos o limiar de
0,30 a 0,95 [`resultados/metricas/analise_limiar_visao.csv`, figura
`resultados/graficos/curva_limiar_visao.png`].

| Limiar | Cobertura | Acurácia entre as aceitas | Pedidos de nova foto |
|---|---|---|---|
| sem limiar | 1,000 | 0,894 | 0 |
| 0,50 | 0,953 | 0,920 | 18 |
| **0,60 (adotado)** | **0,916** | **0,934** | **32** |
| 0,75 | 0,823 | 0,965 | 67 |
| 0,90 | 0,691 | 0,985 | 117 |

A varredura **confirma a escolha de 0,6**: o limiar converte 8,4% de cobertura
em 4 pontos de acurácia entre as respostas efetivamente dadas, e fica no joelho
da curva. Acima de 0,75 o custo cresce rápido — pedir nova foto a cada cinco ou
seis fotografias prejudica a experiência mais do que o erro que evita.

---

## Limitações — Visão Computacional

**O TrashNet não é o mundo real.** As imagens têm fundo branco e iluminação
controlada; fotografias de resíduos em casa têm fundo variado, sombras, vários
objetos no enquadramento e sujeira. É esperado que o desempenho caia fora do
dataset, e a estimativa de 0,894 de acurácia deve ser lida como o limite
superior. A avaliação com fotografias tiradas pela equipe está implementada
(`dados/fotos_equipe/`) mas ainda não foi executada; é o próximo passo e o
resultado deve entrar nesta seção.

**A classe minoritária continua fraca.** Mesmo com pesos na função de perda, a
classe `lixo` fica em 0,766 de F1, bem abaixo das demais, e suas métricas são
instáveis: com 21 imagens de teste, um acerto a mais desloca a revocação em
quase 5 pontos.

**Uma imagem, um objeto.** O modelo classifica a fotografia inteira numa única
classe. Resíduos compostos — uma garrafa de plástico com tampa de metal e
rótulo de papel — não são decompostos, e a resposta reflete o material
dominante na imagem.

**Classes fora do vocabulário.** Pilhas, lâmpadas e eletrônicos não existem no
TrashNet e serão classificados como alguma das seis classes, com confiança
possivelmente alta. É por isso que o módulo de PLN extrai o objeto citado na
pergunta: o texto cobre o que a imagem não tem como saber.
