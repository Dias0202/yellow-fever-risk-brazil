# Portal público — risco de febre amarela silvestre

Portal estático do projeto: previsão da próxima temporada, validação, determinantes,
vacinação e cenários climáticos.

Linguagem visual sóbria — barra escura no topo, fundo cinza-claro, painéis brancos
com borda fina, azul institucional como único acento, tipografia de sistema.

## Regra central

**Os valores das figuras não são digitados no HTML.** Vêm de arquivos JSON em
`dados/`, gerados por `yfvrisk.viz.exportar_painel` a partir das saídas da análise.
Número escrito à mão num HTML não tem como ser reproduzido nem conferido, e diverge
em silêncio dos arquivos de métrica.

## Estrutura

```
index.html          abertura: série observada, por que a validação é prospectiva
previsao.html       modelo por temporada: desempenho, calibração, coeficientes, lista
validacao.html      quatro esquemas, transferência regional, calibração, curva de decisão
determinantes.html  enxame SHAP, confronto Gini × permutação, curvas ALE
vacinacao.html      estudo de evento da cobertura em torno do surto
cenarios.html       matriz GCM × SSP, contrastes pareados, recorte regional
metodos.html        os dois modelos, fontes, decisões, controles de qualidade, pendências

css/painel.css      sistema visual
js/nucleo.js        escalas, eixos, cores, animação, dica flutuante
js/enxame.js        enxame SHAP (a peça central de determinantes.html)
js/pagina-*.js      uma por página
dados/*.json        gerados; não editar à mão
```

## Regerar os dados

```
python -m yfvrisk.viz.exportar_painel
```

O passo do SHAP reajusta o modelo do corte de 2019 e recalcula as contribuições
por ponto, o que leva alguns minutos. Os demais arquivos são leitura direta de
`results/`.

## Servir localmente

Os módulos ES exigem servidor HTTP; abrir os arquivos por `file://` não funciona.

```
cd dashboard && python -m http.server 8848
```

## O que o portal deliberadamente não mostra

Cada omissão decorre de uma medição, e está declarada na própria página:

| Não apresentado | Motivo |
|---|---|
| Mapa municipal de risco do modelo ambiental | Sob bloco espaço-temporal, a calibração superestima o decil de topo por 2,4× a 4,7×; nenhum limiar tem benefício líquido positivo |
| Mapa de cenário climático | Em 83% dos municípios a dispersão entre cenários é menor que a dispersão entre reajustes do modelo |
| Alavanca de vacinação | A variável não carrega sinal generalizável nem depois de defasada em dois anos |
| Comparação entre SSPs | O contraste tem intervalo que inclui zero, e a escolha do GCM altera o resultado por um fator de sete |

## Notas de implementação

**Sem biblioteca de gráficos.** Cada figura responde a uma pergunta específica —
a ordenação que se desfaz, o enxame que se separa, o salto no ano do surto — e
nenhuma é um tipo de gráfico pronto. O SVG escrito à mão dá a forma exata e mantém
o portal sem dependência externa.

**As animações são de transição, não de entrada.** Existem onde há dois estados a
comparar: a métrica que muda de escala, a ordenação que se reorganiza, o corte de
treino que se alterna. `prefers-reduced-motion` desliga todas.

**Reamostragem por município, nunca por linha**, em todo intervalo exibido. As
trinta e duas linhas de um município não são independentes.

## Pendências

- Versão em inglês do portal.
- Camada de mapa com a **história observada** — onde houve surto, e quando.
