# PLN — exemplos de entrada e saída

Modelo: `tfidf_palavras_caracteres_lr` — TF-IDF de palavras + caracteres e Regressão Logística  
Limiar de confiança do sistema: 0.5  
Gerado por `python -m src.pln.avaliar` em 2026-11-17

Frases escritas depois do dataset, nenhuma usada no treino ou no teste.
A última coluna é o item extra do PLN (`src/pln/objetos.py`): o objeto
citado no texto, que a Visão sozinha não tem como saber.

| pergunta | intenção | confiança | sistema responde? | objeto citado |
|---|---|---|---|---|
| o que é esse objeto da foto? | `identificar` | 0.96 | sim | — |
| e ai onde que eu jogo isso? | `como_descartar` | 0.88 | sim | — |
| isso eh recilavel? | `e_reciclavel` | 0.96 | sim | — |
| qual lixiera eu uso? | `cor_lixeira` | 0.86 | sim | — |
| preciso lavar antes de jogar fora? | `como_preparar` | 0.98 | sim | — |
| tem algum ecoponto perto da ufal? | `ponto_coleta` | 0.98 | sim | — |
| e essa pilha velha? | `identificar` | 0.26 | não — pede reformulação | pilha (descarte especial) |
| onde jogo esse pote de vidro vazio? | `como_descartar` | 0.68 | sim | pote de vidro (vidro) |
| a caixa de leite vai junto com o papelao? | `como_descartar` | 0.27 | não — pede reformulação | embalagem longa vida (descarte especial) |
| e esse? | `identificar` | 0.68 | sim | — |
| qual cor de lixeira? | `cor_lixeira` | 1.00 | sim | — |
| bom dia tudo certo? | `fora_do_escopo` | 0.89 | não — pede reformulação | — |
| qual o resultado do flamengo ontem? | `fora_do_escopo` | 0.88 | não — pede reformulação | — |
| ????? | `fora_do_escopo` | 1.00 | não — pede reformulação | — |
