# Agente de decisão do EcoTriagem

Documentação do componente de decisão (`src/decisao/regras.py`), base para a
seção de Metodologia do artigo. Atende ao requisito 4.1 do enunciado
(agentes inteligentes, heurísticas e sistemas baseados em regras).

## 1. Enquadramento

O componente é um **agente reativo baseado em modelo** com **regras
condição–ação** (Russell & Norvig, *Artificial Intelligence: A Modern Approach*,
cap. 2 — conferir edição e página antes de citar):

| Elemento | No EcoTriagem |
|---|---|
| Ambiente | Usuário com um resíduo nas mãos |
| Percepções | Saída da Visão (`ResultadoVisao`: material, confiança, top 3) e do PLN (`ResultadoPLN`: intenção, confiança) |
| Estado interno | A análise da foto é mantida durante a conversa; cada nova pergunta é decidida sobre ela sem reprocessar a imagem |
| Ações | `responder`, `pedir_nova_foto`, `reformular_pergunta` |
| Conhecimento | Tabelas de lixeiras (Resolução CONAMA nº 275/2001), reciclabilidade e preparo |
| Medida de desempenho | Indicar a lixeira correta; quando não houver segurança, pedir mais informação em vez de errar |

## 2. Regras de controle

Avaliadas **em ordem**; a primeira cuja condição é verdadeira define a ação
(resolução de conflito por prioridade). τᵥ = 0,6 e τₚ = 0,5 são os limiares de
confiança da Visão e do PLN (`src/configuracao.py`).

| Regra | Condição | Ação |
|---|---|---|
| **R1** | intenção = `fora_do_escopo` **ou** confiança do PLN < τₚ | `reformular_pergunta` |
| **R2** | confiança da Visão no material ≥ τᵥ | `responder` sobre o material |
| **R3** | confiança no material < τᵥ, **mas** max₍ₗ₎ P(lixeira = l) ≥ τᵥ | `responder` sobre a lixeira *l* |
| **R4** | nenhuma das anteriores | `pedir_nova_foto` |

R1 vem primeiro porque, se a pergunta não foi entendida, a qualidade da
imagem é irrelevante.

### Heurística da lixeira (R3)

O que o usuário precisa saber é **em qual lixeira** descartar, não o material
exato. Vários materiais compartilham a mesma lixeira (papel e papelão → azul).
Por isso a probabilidade de cada lixeira é a soma das probabilidades dos
materiais que vão nela:

    P(lixeira = l) = Σ  P(material = m)     para m no top 3 com lixeira(m) = l

Exemplo: papel 45%, papelão 30%, vidro 20%. Nenhum material passa de 60%,
mas P(azul) = 75%, então o sistema responde "lixeira azul (papel ou papelão)"
em vez de pedir outra foto.

Usar só o top 3 ignora a probabilidade dos demais materiais, o que pode
apenas **subestimar** P(lixeira): a heurística nunca fica mais confiante do
que o modelo de visão permite.

## 3. Regras de resposta (após R2 ou R3)

| Intenção | Resposta |
|---|---|
| `identificar` | Nome do material e confiança (em R3: os materiais candidatos) |
| `como_descartar` | Lixeira e cor |
| `e_reciclavel` | Sim/não, com a lixeira |
| `cor_lixeira` | Cor da lixeira |
| `como_preparar` | Dica de preparo (em R3: uma dica por material candidato) |
| `ponto_coleta` | Rota até o ponto de coleta mais próximo que aceita o material (busca A\*, seção 5) |

Cores (CONAMA 275/2001): azul = papel/papelão; vermelha = plástico;
verde = vidro; amarela = metal; cinza = rejeito não reciclável.

## 4. Rastreabilidade e avaliação

Toda `Decisao` informa a regra aplicada (`regra`). Ela aparece no painel da
interface e é contabilizada em `python -m src.integracao.avaliar_sistema`
(chave `regras` do JSON): quantas vezes cada regra foi aplicada e a taxa de
acerto quando aplicada — em especial, quanto a R3 evita pedidos de nova foto
sem introduzir erros.

## 5. Busca do ponto de coleta

Quando a intenção é `ponto_coleta`, o agente resolve um problema de busca
num grafo (`src/decisao/busca.py`, mapa em `dados/mapa/maceio_simulado.json`):

| Elemento | Definição |
|---|---|
| Estados | 26 bairros de Maceió (coordenadas **aproximadas**) |
| Estado inicial | Bairro informado pelo usuário na interface |
| Ações / custo | Ir a um bairro vizinho (46 ligações); custo = distância em linha reta × fator de tortuosidade ≥ 1 (km) |
| Objetivo | Qualquer bairro com ponto de coleta que aceite o(s) material(is) (6 pontos **fictícios**) |
| Heurística | h(n) = menor distância em linha reta (haversine) de n até um bairro objetivo |

A heurística é **admissível** e **consistente**: como cada aresta custa a
distância em linha reta multiplicada por um fator ≥ 1, e a distância em linha
reta obedece à desigualdade triangular, h nunca superestima o custo restante.
Por isso o A\* é ótimo. As duas propriedades são verificadas em
`testes/test_busca.py` para todos os bairros e materiais.

**Comparação** (`python -m src.decisao.comparar_buscas`; 130 consultas =
26 origens × 5 materiais recicláveis; referência de ótimo = custo uniforme):

| Algoritmo | Rotas ótimas | Excesso médio | Excesso máximo | Nós expandidos (média) |
|---|---|---|---|---|
| Busca em largura (BFS) | 83.1% | 2.6% | 36.3% | 4.7 |
| Custo uniforme | 100.0% | 0.0% | 0.0% | 4.3 |
| Busca gulosa | 86.2% | 2.2% | 28.4% | 2.2 |
| A* | 100.0% | 0.0% | 0.0% | 2.4 |

O A\* encontra sempre a rota de menor distância expandindo cerca de metade dos
nós do custo uniforme. A busca gulosa expande ainda menos, mas não garante a
menor rota. A BFS minimiza o número de ligações, não a distância. Os tempos
ficam em microssegundos para todos (mapa pequeno) e estão no JSON de resultados.
Gráficos: `resultados/graficos/comparacao_buscas_*.png` e `mapa_rota_exemplo.png`.

**Limitação:** o mapa é simulado. Para uso real, trocar
`dados/mapa/maceio_simulado.json` por dados de ruas e pontos de coleta oficiais
(o código não muda).

## 6. Pendências

- Textos de preparo: conferir com fonte oficial (prefeitura ou guia de
  coleta seletiva) e citar.
- Casos especiais: papel engordurado, vidro quebrado, longa vida, pilhas e
  eletrônicos (depende do PLN extrair o objeto citado).
