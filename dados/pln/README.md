# Dataset de intenções (PLN)

`intencoes.csv` — colunas `texto,intencao`. Cada linha é uma pergunta que um usuário
poderia escrever para o EcoTriagem, com a intenção correspondente (uma das sete de
`src/configuracao.py`).

`intencoes_teste.csv` (opcional, mesmas colunas) — frases escritas por **outros
integrantes da equipe**. Se o arquivo existir, ele é usado como conjunto de teste e
o `intencoes.csv` inteiro vira treino. Isso evita que o teste fique viciado no jeito
de escrever de quem montou o treino. Sem esse arquivo, o teste é uma divisão
estratificada de 20% do `intencoes.csv` com seed 42.

## Como as frases foram escritas

Meta de 40 a 60 frases por intenção, variando de propósito:

- registro formal ("qual a forma correta de descarte?") e informal ("e ai onde jogo?");
- gírias e abreviações ("vc", "pra", "troço");
- frases sem acento ("reciclavel") e com erro de digitação ("recilavel", "lixiera");
- frases muito curtas que só fazem sentido com a foto em contexto ("e isso?", "cor?");
- frases que **nomeiam o objeto** ("onde jogo essa latinha?", "a caixa de leite é
  reciclável?"), e não só "isso" / "esse material".

O último item foi acrescentado numa segunda rodada: a primeira versão do dataset falava
sempre em "isso" e "esse material" e o modelo errava perguntas com o objeto nomeado,
que é exatamente como as pessoas escrevem. Foram 36 frases novas (6 por intenção dentro
do escopo), identificadas em frases de teste manual — nunca no conjunto de teste.

## Critérios de rotulagem (as fronteiras ambíguas)

A maior fonte de ruído num dataset de intenções é a mesma frase receber rótulos
diferentes. As regras abaixo foram fixadas antes de escrever as frases:

| Dúvida | Regra adotada |
|---|---|
| "onde jogo isso?" | `como_descartar` — pergunta pelo destino/procedimento, não por um endereço |
| "onde levo isso?" / "onde entrego?" | `ponto_coleta` — pede um local físico para levar o resíduo |
| "em qual lixeira?" / "qual a cor?" | `cor_lixeira` — pergunta pelo recipiente da coleta seletiva |
| "isso é reciclável?" | `e_reciclavel` — resposta sim/não sobre reciclabilidade |
| "isso é lixo ou reciclável?" | `e_reciclavel` — continua sendo sim/não |
| "o que é isso?" / "é plástico?" | `identificar` — pergunta sobre o material do objeto |
| "preciso lavar/amassar/tirar a tampa?" | `como_preparar` — preparo antes do descarte |
| "como chego no aeroporto?" | `fora_do_escopo` — pergunta de rota que não é sobre resíduo |

`fora_do_escopo` reúne três tipos de entrada: saudações e despedidas ("oi", "valeu"),
perguntas de outros assuntos ("vai chover hoje?") e texto sem sentido ("asdasd", "?????").
Essa classe existe para o sistema pedir uma reformulação em vez de inventar uma resposta,
e é a mais difícil: ela não tem vocabulário próprio, é definida por ausência.
