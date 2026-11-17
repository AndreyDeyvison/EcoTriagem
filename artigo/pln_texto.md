# PLN — texto para o artigo

Rascunho das seções de Processamento de Linguagem Natural, para o Edvaldo
adaptar ao template IEEE. Todo número aqui sai de um script do repositório e
está salvo em `resultados/` — a fonte de cada um está indicada entre colchetes.

---

## Metodologia — Processamento de Linguagem Natural

### Formulação do problema

O EcoTriagem precisa distinguir o que o usuário quer saber a respeito do
resíduo fotografado: identificar o material, saber como descartá-lo, saber se
é reciclável, saber a cor da lixeira, saber como prepará-lo, ou localizar um
ponto de coleta. A tarefa foi modelada como **classificação de intenção**, um
problema de classificação supervisionada de texto curto com sete classes
mutuamente exclusivas, sendo a sétima (`fora_do_escopo`) reservada a saudações,
perguntas de outros domínios e texto sem sentido. Essa última classe é o que
permite ao agente pedir uma reformulação em vez de responder qualquer coisa.

### Construção do conjunto de dados

Não há, até onde verificamos, um conjunto público de perguntas em português
sobre descarte de resíduos. O conjunto foi então escrito pela equipe: **399
frases, de 52 a 58 por intenção** [`dados/pln/intencoes.csv`]. As frases foram
escritas deliberadamente em registros variados, porque o usuário real não
escreve como um formulário: linguagem formal ("qual a forma correta de
descarte?") e informal ("e ai onde jogo?"), abreviações ("vc", "pra"), ausência
de acentuação ("reciclavel"), erros de digitação ("recilavel", "lixiera") e
frases muito curtas que só fazem sentido junto com a foto ("e isso?", "cor?").

Os critérios de rotulagem foram fixados antes da escrita e estão documentados
em `dados/pln/README.md`, porque as fronteiras entre intenções próximas são a
principal fonte de ruído: "onde jogo isso?" é `como_descartar` (pergunta pelo
destino), enquanto "onde levo isso?" é `ponto_coleta` (pede um local físico).

Uma segunda versão do conjunto acrescentou 36 frases que **nomeiam o objeto**
("onde jogo essa latinha?", "a caixa de leite é reciclável?"). A primeira
versão falava sempre em "isso" e "esse material", e testes manuais mostraram
que o modelo errava justamente as perguntas escritas como as pessoas escrevem.
A correção foi feita a partir de frases novas, nunca do conjunto de teste.

A divisão é estratificada, 80% para treino (319 frases) e 20% para teste (80
frases), com semente fixa 42. O pipeline aceita um arquivo
`dados/pln/intencoes_teste.csv` escrito por outros integrantes da equipe; quando
presente, ele substitui a divisão aleatória, de modo que o teste não fique
viciado no estilo de escrita de quem montou o treino.

### Pré-processamento

A normalização converte para minúsculas, remove acentos e pontuação, colapsa
espaços e encurta letras repetidas de ênfase ("ooooi" → "ooi"). A mesma função
é usada no treino e na inferência, o que elimina a diferença entre o texto que
o modelo aprendeu e o texto que recebe em produção.

A lista de stopwords **exclui as negações** ("não", "nem", "sem", "nunca"):
descartar "não" é um erro frequente em português, porque inverte o sentido da
frase. Mediu-se o efeito dessa decisão (Tabela I): remover stopwords preservando
as negações já custa 9 pontos de F1-macro, e remover também as negações não
melhora nada — ou seja, para este domínio a remoção de stopwords é prejudicial,
e o modelo final não a utiliza.

Entradas sem nenhuma letra ou número ("?????") são classificadas como
`fora_do_escopo` por regra, antes do modelo: elas não geram nenhuma
característica TF-IDF e a decisão sairia apenas do intercepto do classificador.

### Modelos comparados

Todos os modelos seguem o mesmo esqueleto — representação TF-IDF seguida de um
classificador linear — e diferem na representação do texto:

- **palavras**: n-gramas de palavras (1 e 2);
- **caracteres**: n-gramas de caracteres de 3 a 5 dentro das palavras,
  escolhido por tolerar erro de digitação e falta de acento ("recilavel" ainda
  compartilha os trechos "recicl" e "ciclav" com "reciclavel");
- **palavras + caracteres**: as duas representações concatenadas.

O ponto de comparação é o **baseline de casamento de palavras-chave**, a versão
que o protótipo usava antes e que não aprende nada dos dados.

A escolha do modelo usou **validação cruzada estratificada de 5 partes sobre o
conjunto de treino**, e não um único conjunto de validação: com poucas centenas
de frases, uma validação única teria cerca de 50 exemplos e a diferença entre
dois modelos ficaria dentro do ruído amostral. O conjunto de teste foi usado uma
única vez, depois de o modelo já estar escolhido.

---

## Resultados — PLN

**Tabela I — Comparação dos modelos (F1-macro, validação cruzada de 5 partes
no treino).** [`resultados/metricas/pln_comparacao_modelos.csv`]

| Modelo | F1-macro | Desvio |
|---|---|---|
| Baseline: palavras-chave | 0,673 | ±0,039 |
| TF-IDF palavras + Regressão Logística | 0,841 | ±0,037 |
| TF-IDF caracteres + Regressão Logística | 0,890 | ±0,029 |
| TF-IDF palavras+caracteres + LinearSVC calibrado | 0,897 | ±0,033 |
| **TF-IDF palavras+caracteres + Regressão Logística** | **0,900** | **±0,022** |
| Ablação: sem stopwords (negações preservadas) | 0,810 | ±0,057 |
| Ablação: sem stopwords e sem negações | 0,810 | ±0,063 |

A representação por caracteres sozinha já supera a representação por palavras em
5 pontos de F1-macro, o que é coerente com um conjunto pequeno escrito em
linguagem informal: ela aproveita a semelhança entre formas que o modelo de
palavras trataria como termos sem relação. O modelo final combina as duas.

**Tabela II — Desempenho no conjunto de teste (80 frases).**
[`resultados/metricas/pln_relatorio_teste.csv`]

| | Acurácia | F1-macro |
|---|---|---|
| Baseline: palavras-chave | 0,662 | 0,689 |
| Modelo final | **0,875** | **0,877** |

| Intenção | Precisão | Revocação | F1 | F1 baseline | n |
|---|---|---|---|---|---|
| identificar | 0,846 | 0,917 | 0,880 | 0,667 | 12 |
| como_descartar | 0,692 | 0,818 | 0,750 | 0,700 | 11 |
| e_reciclavel | 1,000 | 0,833 | 0,909 | 0,833 | 12 |
| cor_lixeira | 0,917 | 1,000 | 0,957 | 0,900 | 11 |
| como_preparar | 0,900 | 0,750 | 0,818 | 0,588 | 12 |
| ponto_coleta | 0,909 | 0,833 | 0,870 | 0,667 | 12 |
| fora_do_escopo | 0,909 | 1,000 | 0,952 | 0,465 | 10 |

O ganho sobre o baseline é de 19 pontos de F1-macro, e está concentrado onde o
casamento de palavras-chave é estruturalmente incapaz: `fora_do_escopo` passa de
0,465 para 0,952, porque essa classe não tem vocabulário próprio — é definida
por ausência, e só um modelo treinado consegue representá-la.

**Figura — matriz de confusão** [`resultados/graficos/pln_matriz_confusao.png`].
Os 10 erros do teste [`resultados/metricas/pln_erros.csv`] estão quase todos
entre intenções vizinhas, que compartilham vocabulário e cuja fronteira é fina
até para um anotador humano: "como deixo isso seguro pra descartar?" (preparo)
foi lida como `como_descartar`, e "onde posso levar esse residuo?" (ponto de
coleta) como `como_descartar`. Nenhum erro cruza para um grupo semanticamente
distante, e metade deles ocorre com confiança abaixo de 0,5 — ou seja, o sistema
pede uma reformulação em vez de responder errado.

### Análise do limiar de confiança

O sistema adota um limiar de 0,5: abaixo dele a resposta dá lugar a um pedido de
reformulação. Para avaliar essa escolha, varreu-se o limiar de 0 a 0,95
[`resultados/metricas/pln_limiar.csv`, figura
`resultados/graficos/pln_limiar_confianca.png`], medindo três quantidades:
a cobertura (quantas perguntas o sistema aceita responder), a acurácia entre as
respondidas, e o **acerto do sistema** — a fração de casos em que o sistema faz
a coisa certa, isto é, responde com a intenção correta quando a pergunta é
legítima e recusa quando ela está fora do escopo.

| Limiar | Cobertura | Acurácia entre as respondidas | Acerto do sistema |
|---|---|---|---|
| 0,00 | 1,000 | 0,875 | 0,875 |
| 0,35 | 0,963 | 0,883 | **0,875** |
| 0,50 (adotado) | 0,850 | 0,927 | 0,825 |
| 0,70 | 0,700 | 0,946 | 0,750 |

O resultado **contraria a escolha inicial**. A revocação da classe
`fora_do_escopo` é 1,000 em todos os limiares: o modelo reconhece sozinho as
perguntas fora de escopo, e o limiar não acrescenta nenhuma proteção — apenas
recusa perguntas legítimas. Elevar o limiar melhora a acurácia entre as
respondidas (de 0,875 para 0,927), mas é um ganho ilusório, obtido à custa de
não responder: o acerto do sistema cai de 0,875 para 0,825. O maior limiar sem
perda alguma é **0,35**, e é o valor que recomendamos para
`LIMIAR_CONFIANCA_PLN`.

### Extração do objeto citado

Como extensão, o módulo extrai o objeto mencionado na pergunta por casamento
léxico sobre o texto normalizado [`src/pln/objetos.py`]. A motivação é que o
texto carrega informação que a imagem não tem: em "onde jogo essa pilha?", a
foto pode ser classificada como metal, mas pilha é resíduo perigoso e segue
logística reversa, não a lixeira amarela. O léxico devolve a classe de resíduo
quando o termo já determina o material ("latinha" → metal), deixa a decisão com
a Visão quando o termo é ambíguo ("garrafa" pode ser vidro ou plástico) e
sinaliza os resíduos que exigem tratamento fora das seis classes do TrashNet
(pilha, bateria, lâmpada, eletrônico, óleo de cozinha, embalagem longa vida).

---

## Limitações — PLN

**O conjunto foi escrito pela equipe.** As 399 frases cobrem a variedade que
conseguimos antecipar, não a distribuição real de perguntas de usuários. Frases
curtas com objeto nomeado continuam sendo o ponto fraco: "e essa pilha velha?"
recebe confiança 0,26 e não é respondida, embora a extração léxica reconheça a
pilha. A correção natural é coletar perguntas reais em uso.

**O conjunto de teste é pequeno** (80 frases, 10 a 12 por intenção): uma
diferença de um acerto desloca o F1 de uma classe em cerca de 4 pontos, de modo
que as comparações por classe devem ser lidas como indicativas. A escolha do
modelo, essa sim, usou validação cruzada sobre 319 frases.

**O modelo não representa composição.** "a caixa de leite vai junto com o
papelão?" é uma pergunta de comparação entre dois materiais; o classificador
devolve `como_descartar` com confiança 0,27. Tratar esse tipo de pergunta exige
ir além da classificação de intenção isolada.

**Não há desambiguação por contexto.** Cada pergunta é classificada sozinha; o
sistema não usa a pergunta anterior da conversa para interpretar a seguinte.
