# Risco de febre amarela silvestre no Brasil

Previsão, com uma temporada de antecedência, de febre amarela silvestre nos 5.570
municípios brasileiros, e as análises ecológicas, de vacinação e de cenários
climáticos que a sustentam.

> **Painel online:** https://dias0202.github.io/yellow-fever-risk-brazil/ ·
> English version: [README.md](README.md) · Model card: [MODEL_CARD.md](MODEL_CARD.md)

## O problema

A febre amarela silvestre não é uma propriedade estática do habitat. O vírus circula
entre mosquitos *Haemagogus*/*Sabethes* e primatas não humanos e avança como **onda
epizoótica por floresta contígua**. O município é atingido porque a onda chegou
perto, não porque o seu clima mudou. Depois que a onda passa, a população local de
bugios é dizimada e a sobrevivente fica imune, e a área permanece refratária por
anos.

Daí três consequências, que organizam o projeto:

1. **Contágio e persistência carregam a maior parte do sinal.** Um município com foco
   a menos de 50 km na temporada anterior tem probabilidade de evento na temporada
   seguinte **16,8×** a taxa de base. Um município que teve evento ele mesmo, **19×**.
2. **A unidade de tempo é a temporada de vigilância julho–junho, não o ano civil.**
   97,6% dos casos humanos ocorrem entre dezembro e maio, e o ano civil corta cada onda
   ao meio. Os registros do Ministério da Saúde trazem o campo de temporada
   `MONITORAMENTO` em 100% dos casos.
3. **A validação precisa corresponder ao uso.** O desfecho é raro (0,74% dos
   municípios-ano). Anos adjacentes do mesmo município são quase duplicatas, de modo
   que uma partição aleatória por linha mede interpolação, e não previsão.

## Dois modelos, duas perguntas

| Modelo | Pergunta | Validação |
|---|---|---|
| **Previsão por temporada**: regressão logística com regularização L2 sobre 12 variáveis, sem peso de classe, calibração de Platt | Onde a onda em curso deve aparecer na temporada que começa em julho? | Janela expansiva por temporada: treina em tudo até T−1, prevê T, avança |
| **Adequabilidade ambiental**: floresta aleatória sobre 32 variáveis de clima, paisagem, imunidade e demografia | Quais determinantes ecológicos se sustentam fora da amostra, e como o clima projetado desloca o risco? | Blocos temporais, espaciais (UF fora) e espaço-temporais (macrorregião *e* ano fora) |

### Por que não uma partição aleatória

Mesma floresta aleatória, mesmo painel, mesmas variáveis ambientais. Muda apenas a
partição (`dashboard/dados/validacao.json`):

| Partição | ROC-AUC | PR-AUC | × acaso | Municípios investigados por evento |
|---|---:|---:|---:|---:|
| Linhas aleatórias | 0,965 | 0,421 | 57,4× | 2,9 |
| Espacial (UF não vista) | 0,864 | 0,102 | 13,8× | 10,4 |
| Temporal (ano não visto) | 0,746 | 0,033 | 3,4× | 23,1 |
| Espaço-temporal (macrorregião e ano não vistos) | 0,524 [0,505; 0,542] | 0,024 | 2,4× | 20,5 |

A ROC-AUC cai 0,44, enquanto a PR-AUC sobre o acaso cai por um fator de 24. O clima
defasado mostra o mecanismo com mais clareza. O clima do ano anterior é quase idêntico
entre linhas vizinhas do mesmo município, de modo que acrescentá-lo eleva a PR-AUC em
53% sob partição aleatória e em apenas 15% sob partição temporal. **Todo resultado
abaixo vem de validação em que o período avaliado nunca participa do ajuste.**

## Resultados

### Previsão por temporada

O modelo é avaliado prospectivamente em 21 temporadas, com as mesmas variáveis que
produzem a lista publicada (`results/metrics/f6_05_previsao_operacional.json`):

| Ordenação | ROC-AUC | PR-AUC / acaso | Recall @100 | Recall @500 | Recall @1.000 |
|---|---:|---:|---:|---:|---:|
| Modelo logístico final | 0,734 | **7,28×** | **0,108** | 0,360 | 0,515 |
| Força de infecção crua (sem aprendizado) | 0,775 | 8,14× | 0,083 | 0,362 | 0,541 |
| Persistência local | — | 1,98× | — | — | — |

- **Nenhuma ordenação domina a outra.** O modelo aprendido é melhor no topo da lista:
  em 100 municípios captura cerca de 30% mais surtos. A força de infecção ponderada por
  distância é melhor na cauda. A tabela de saída traz as duas ordenações, com a
  recomendação registrada no próprio arquivo: o modelo para listas curtas (≈100) e a
  força de infecção para varreduras largas (500+). As duas superam com folga a
  persistência local.
- **Curva de decisão.** Os modelos por temporada têm uma faixa de limiares com
  benefício líquido positivo: de 19 a 28, conforme o componente
  (`results/metrics/f6_0*`).
- **Calibração por decil** monotônica, com correlação de posto entre decis de +1,00. O
  decil mais baixo tem taxa observada de 0,026%. O decil de topo prevê 3,87% e observa
  3,90%.
- **Todo coeficiente tem o sinal que a epidemiologia prevê.** A distância ao foco
  entra negativa; força de infecção, cobertura florestal e altitude entram positivas;
  temporadas desde o último evento entram positivas, o que é refratariedade.
- **Por que parcimonioso.** Com dez formas de medir "quão perto esteve a onda", os
  preditores quase colineares se cancelavam, e o decil de menor risco previsto tinha
  taxa observada de 1,90%, quatro vezes a média. Um representante por mecanismo, com
  L2 forte, remove o cancelamento na origem.
- **Por que sem peso de classe.** Reponderar os positivos por ~140× melhora a
  calibração mas destrói a ordenação, e é a ordenação que a PR-AUC e uma lista de
  prioridade medem.

**Limitação declarada.** O modelo prevê o *avanço de uma onda em curso* e **não**
prevê a *emergência de uma onda nova*. Nas temporadas em que a onda avançava
(2018/19, 2019/20, 2020/21), chega a 23×, 16× e 8,6× o acaso. Em 2016/17, a temporada
em que a onda chegou à Mata Atlântica, cai a **1,2×**. Agregar em microrregiões foi
testado e não ajuda: eleva a prevalência e reduz a razão sobre ela. A previsão para a
temporada 2025/26, para os 5.570 municípios, está em
`results/tables/f6_05_previsao_proxima_temporada.csv`.

### Determinantes ecológicos (modelo ambiental)

- No treino, a importância de Gini (MDI) reproduz quase perfeitamente a importância
  por permutação (ρ = +0,81 e +0,91 nos cortes de 2014 e 2019). Em dado não visto, não
  tem relação estatística com ela (ρ = +0,10 e +0,30; p = 0,57 e 0,09).
- Só quatro das 32 variáveis têm IC 95% da importância por permutação acima de zero
  nos dois cortes temporais: `ppt_sazonalidade`, `populacao_log`, `tmin_medio` e
  `ppt_total_lag2`.
- Sob bloco espaço-temporal, o **Sul** é a única macrorregião em que o modelo
  transfere sem história local: ROC-AUC 0,737 [0,697; 0,777] e PR-AUC 8,0× o acaso.
  Todas as demais ficam entre 0,45 e 0,56.

### Vacinação

A cobertura vacinal registrada é **consequência do surto**, e não causa de proteção.
Num estudo de evento com dupla centragem, por ano e por município, a cobertura fica no
normal do município até o ano anterior ao surto, **salta 4,5× no ano do surto** e
permanece elevada por três anos: é o bloqueio vacinal que o próprio surto dispara.

| Defasagem da imunidade | ρ com o evento, dentro do município |
|---|---|
| 0 anos | +0,082 [+0,073; +0,091] |
| 1 ano | +0,017 |
| 2 anos | −0,005 (p = 0,35) |

A variável de modelagem é, por isso, a imunidade defasada em dois anos. Na ablação, a
imunidade contemporânea acrescentava +0,0199 de ROC-AUC [+0,0070; +0,0320] e nada de
PR-AUC, que é a assinatura de um vazamento. Com a defasagem, a imunidade não carrega
sinal generalizável, e por isso **nenhum cenário de vacinação é reportado**.

### Cenários climáticos (CMIP6, 2021–2040)

São cinco GCMs, escolhidos por cobrirem a faixa de sensibilidade climática de
equilíbrio do CMIP6 (INM-CM5-0 ≈1,9 K a UKESM1-0-LL ≈5,3 K), e quatro SSPs. O método
delta é aplicado a cada ano observado, e a incerteza vem de 30 reajustes por
reamostragem de municípios.

| Contraste | Média | IC 95% |
|---|---:|---|
| Futuro − linha de base | +0,00083 | [+0,00040; +0,00130] |
| SSP5-8.5 − SSP1-2.6 | +0,00010 | [−0,00006; +0,00023] |

- O sinal climático sobrevive, mas **a separação entre SSPs não**.
- A escolha do GCM move o resultado muito mais que a do SSP (dispersão entre GCMs de
  0,46 contra 0,10 entre SSPs, em `dashboard/dados/clima.json`). O UKESM1-0-LL no
  cenário mais brando (+17,0%) projeta aumento maior que qualquer outro GCM no mais
  severo (no máximo +9,2%).
- Em 83% dos municípios, a dispersão entre cenários é menor que a dispersão entre
  reajustes do modelo. Por isso **nenhum mapa municipal de cenário é publicado**.

## Dados

São dez fontes públicas, cada uma com procedência declarada em
[`data/SOURCES.yml`](data/SOURCES.yml).

| Fonte | Conteúdo | Obtenção |
|---|---|---|
| Ministério da Saúde (CGARB/DEDT/SVSA) | Casos humanos confirmados e epizootias em primatas, 1994–2025 | Exportação manual |
| TerraClimate (Abatzoglou et al., 2018) | Clima mensal, 1/24° | `ingest/baixar_terraclimate.py` |
| WorldClim 2.1 / CMIP6 | Linha de base e projeções 2021–2040, 5 GCMs × 4 SSPs; elevação | `ingest/baixar_worldclim.py` |
| Hansen GFC-2024-v1.12 | Cobertura arbórea em 2000 e perda anual, 30 m | `ingest/baixar_hansen.py` |
| Derivado do MapBiomas (Abdalla e Augusto, 2022) | Frações de uso do solo, 1985–2020 | Exportação manual |
| IBGE | Estimativas populacionais, malha municipal 2022, hierarquia territorial (API) | Manual / API |
| PNI / DATASUS | Cobertura vacinal contra febre amarela, 1994–2022 | Exportação manual |

## Reprodutibilidade

**O que está no repositório:** todo o código, os testes, todos os JSONs de métrica e
tabelas de resultado citados neste README e no painel, e os dados do painel.

**O que não está:** os dados brutos e o painel processado (`data/raw/`,
`data/processed/`). Várias fontes são exportações manuais (TabNet, planilhas do IBGE,
arquivos do Ministério da Saúde, derivado do MapBiomas) que não podem ser baixadas
automaticamente, e os rasters de clima e floresta somam perto de 200 GB. Os testes que
dependem do painel construído **são pulados com o motivo declarado** quando ele não
existe. A álgebra do módulo de cenários e as invariantes dos grupos de variáveis rodam
sempre.

O layout esperado de `data/raw/` e a sequência completa de comandos estão no
[README em inglês](README.md#reproducibility).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt
export PYTHONPATH=src
ruff check .
python -m pytest
cd dashboard && python -m http.server 8848   # painel
```

## Licença

- **Código:** MIT, ver [LICENSE](LICENSE).
- **Resultados derivados** (`results/`, `dashboard/dados/`): CC BY 4.0, ver
  [LICENSE-DATA.md](LICENSE-DATA.md).
- **Dados de origem** mantêm os termos de cada provedor. Cite as fontes originais
  listadas em `data/SOURCES.yml`.

## Citação

Ver [CITATION.cff](CITATION.cff). Desenvolvido no Laboratório de Vírus da Universidade
Federal de Minas Gerais, com financiamento da FAPEMIG.
