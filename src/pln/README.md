# Módulo de PLN — classificação de intenções

Decide **o que o usuário quer saber** sobre o resíduo fotografado. Recebe a
pergunta em linguagem natural e devolve uma das sete intenções de
`src/configuracao.py` com uma confiança, no contrato `ResultadoPLN`.

```python
from src.pln.intencoes import interpretar_pergunta

interpretar_pergunta("e ai onde que eu jogo isso?")
# ResultadoPLN(intencao='como_descartar', confianca=0.88, texto_original='e ai onde que eu jogo isso?')
```

## Como rodar

```bash
python -m src.pln.treinar    # compara os modelos e salva o melhor em modelos/
python -m src.pln.avaliar    # mede no teste e gera métricas, gráficos e exemplos
pytest testes/test_pln.py
```

`treinar.py` precisa rodar antes de `avaliar.py`. O modelo treinado já vem no
repositório, então treinar só é necessário para reproduzir os números ou mudar o
modelo. Sem o arquivo, o sistema continua funcionando com o baseline de
palavras-chave e avisa no console.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `dados/pln/intencoes.csv` | dataset de 399 frases (52 a 58 por intenção) — versionado |
| `dados/pln/README.md` | critérios de rotulagem e como as frases foram escritas |
| `preprocessamento.py` | normalização, tokenização e stopwords (negações preservadas) |
| `dados.py` | leitura do CSV e divisão treino/teste (estratificada, seed 42) |
| `baseline.py` | casamento de palavras-chave: a versão anterior, usada como comparação |
| `modelos.py` | catálogo de modelos e o `ClassificadorIntencoes` salvo em `modelos/` |
| `treinar.py` | validação cruzada de 5 partes, escolha do modelo e treino final |
| `avaliar.py` | métricas no teste, matriz de confusão, erros e análise do limiar |
| `intencoes.py` | `interpretar_pergunta` — o que o resto do sistema chama |
| `objetos.py` | extra: extrai o objeto citado na frase (pilha, lâmpada, caixa de leite...) |

## Resultados

Conjunto de teste de 80 frases, nunca usado para escolher o modelo:

| | acurácia | F1-macro |
|---|---|---|
| baseline de palavras-chave | 0,662 | 0,689 |
| **TF-IDF palavras + caracteres e Regressão Logística** | **0,875** | **0,877** |

Números completos em `resultados/metricas/pln_*` e gráficos em
`resultados/graficos/pln_*`. Exemplos de entrada e saída em
`resultados/exemplos/pln_entrada_saida.md`.

## Dois pontos de atenção para a equipe

**O limiar de 0,5 custa mais do que rende** (para o módulo de decisão). O modelo
detecta sozinho todas as perguntas fora do escopo do teste — a classe
`fora_do_escopo` tem revocação 1,000 —, então o limiar não acrescenta proteção
e só recusa pergunta legítima: a 0,5 o sistema acerta 82,5%, e a 0,35 acerta
87,5% respondendo 96,3% das perguntas. Sugestão de baixar `LIMIAR_CONFIANCA_PLN`
para 0,35, a combinar com a equipe (regra 2 do plano de trabalho).

**O objeto citado no texto está disponível** (para o módulo de decisão).
`objetos.extrair_objeto("onde jogo essa pilha?")` devolve o objeto, a classe
sugerida quando o termo já determina o material, e o motivo quando o resíduo
exige descarte especial. Isso cobre os casos que a Visão não tem como resolver
(pilha, lâmpada, eletrônico, embalagem longa vida). Nada disso altera
`ResultadoPLN`: pôr o objeto no contrato precisa ser combinado com a equipe.
