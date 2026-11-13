/* Enxame de valores SHAP.

   Cada ponto é um município-ano do período 2020–2025, que o modelo nunca viu no
   treino. A posição horizontal é quanto aquela variável empurrou a predição
   daquela linha, para cima ou para baixo; a cor é onde o valor da variável cai na
   distribuição, do menor ao maior.

   Um gráfico de barras de importância diz que uma variável importa; não diz em que direção, nem que a
   direção pode inverter conforme o valor. O enxame mostra as três coisas de uma
   vez, e mostra também quando uma variável só age numa cauda — nuvem concentrada
   em zero com um punhado de pontos longe.

   A animação tem função. Os pontos partem de zero, que é o estado em que nada se
   sabe da contribuição de nenhuma variável, e se separam para as posições
   medidas. E a ordenação alterna entre a contribuição medida fora da amostra e a
   importância de Gini medida no treino: a reordenação é o resultado. */

import {
  cria, svg, linear, eixoX, rampa, aoEntrar, anima, dica, apelido,
  CORES_GRUPO, NOME_GRUPO, fmt, variavelCss, semMovimento,
} from "./nucleo.js";

const MARGEM = { topo: 34, direita: 26, baixo: 52, esquerda: 214 };
const RAIO = 2.1;
const ALTURA_LINHA = 34;

/* Metade da altura útil de uma linha. Nenhum ponto passa disso, senão o enxame
   de uma variável invade a faixa da vizinha e as duas ficam ilegíveis. */
const DESVIO_MAXIMO = ALTURA_LINHA / 2 - RAIO - 1;

/* Empacotamento em abelha: agrupa por faixa horizontal e empilha alternando para
   cima e para baixo a partir do centro da linha.

   O empilhamento é limitado a DESVIO_MAXIMO. Onde a coluna estoura o limite — o
   que acontece justamente em torno de zero, onde se concentra a maioria dos
   municípios-ano — os pontos excedentes voltam para dentro numa segunda camada,
   em vez de continuarem subindo. O resultado é uma banda de topo achatado, que é
   a leitura correta: ali há mais pontos do que a linha comporta desenhar
   separados. */
function empacota(xs, raio) {
  const largura = raio * 1.9;
  const baldes = new Map();
  const ys = new Array(xs.length);
  const ordem = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]);
  const salto = raio * 1.7;
  const camadas = Math.max(1, Math.floor(DESVIO_MAXIMO / salto));

  for (const i of ordem) {
    const chave = Math.round(xs[i] / largura);
    const n = baldes.get(chave) || 0;
    baldes.set(chave, n + 1);
    // 0, +1, −1, +2, −2, … e depois recomeça, deslocado meio passo.
    const nivel = Math.ceil(n / 2);
    const sinal = n % 2 === 0 ? 1 : -1;
    const volta = Math.floor(nivel / (camadas + 1));
    const dentro = nivel % (camadas + 1);
    ys[i] = sinal * (dentro * salto + (volta % 2) * salto * 0.5);
  }
  return ys;
}

/* Quantil de um vetor já ordenado. */
function quantil(ordenado, q) {
  if (!ordenado.length) return 0;
  const i = Math.min(ordenado.length - 1,
                     Math.max(0, Math.round(q * (ordenado.length - 1))));
  return ordenado[i];
}

export async function enxame(hospedeiro, dados, importancia, opcoes = {}) {
  const { aoTrocarOrdem = null } = opcoes;
  const cores = CORES_GRUPO();
  const cor = rampa([variavelCss("--baixo"), variavelCss("--meio"),
                     variavelCss("--alto")]);
  const t = dica();

  const mdi = new Map();
  for (const v of importancia.cortes["2019"].variaveis) {
    mdi.set(v.nome, { mdi: v.mdi, posto: v.posto_mdi });
  }

  const vars = dados.variaveis.map((v) => ({
    ...v, mdi: mdi.get(v.nome)?.mdi ?? 0, posto_mdi: mdi.get(v.nome)?.posto ?? 99,
  }));

  const porShap = [...vars].sort((a, b) => b.medio_absoluto - a.medio_absoluto);
  const porGini = [...vars].sort((a, b) => b.mdi - a.mdi);

  const largura = 940;
  const altura = MARGEM.topo + vars.length * ALTURA_LINHA + MARGEM.baixo;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label",
    "Enxame de contribuições SHAP por variável, medidas fora da amostra");

  /* O domínio vem do percentil 99,5 do módulo da contribuição, e não do máximo.
     Com o máximo, uma dezena de municípios-ano extremos comprime toda a estrutura
     restante contra a linha do zero e o gráfico deixa de mostrar o que tem a
     mostrar. Os pontos além do limite ficam desenhados sobre a borda, marcados
     com opacidade menor — são visíveis como acúmulo na extremidade, e o número
     deles entra na dica. */
  const todos = vars.flatMap((v) => v.contribuicao.map(Math.abs))
    .sort((a, b) => a - b);
  const limite = quantil(todos, 0.995) || Math.max(...todos);
  const x = linear([-limite, limite], [MARGEM.esquerda, largura - MARGEM.direita]);
  const recorta = (c) => Math.max(-limite, Math.min(limite, c));

  /* Faixa alternada, quase imperceptível, só para ancorar a leitura da linha. */
  const fundo = cria("g", {}, s);

  /* Linha do zero: a referência de "não empurrou para lado nenhum". */
  cria("line", {
    x1: x(0), x2: x(0), y1: MARGEM.topo - 10,
    y2: altura - MARGEM.baixo + 4,
    stroke: "#bbb", "stroke-width": 1,
  }, s);

  eixoX(s, x, altura - MARGEM.baixo + 12, {
    formato: (v) => (v === 0 ? "0" : fmt.sinal(v, 2)),
    titulo: "contribuição para a probabilidade prevista",
    tamanho: 6,
  });

  const camadaLinhas = cria("g", {}, s);
  const linhas = new Map();

  for (const v of vars) {
    const g = cria("g", { class: "linha-enxame" }, camadaLinhas);
    const faixa = cria("rect", {
      x: MARGEM.esquerda - 200, y: -ALTURA_LINHA / 2,
      width: largura - MARGEM.esquerda - MARGEM.direita + 200,
      height: ALTURA_LINHA, fill: "transparent",
    }, g);

    const rotulo = cria("text", {
      x: MARGEM.esquerda - 14, y: 4, "text-anchor": "end",
      class: "rotulo-serie",
    }, g);
    rotulo.textContent = apelido(v.nome);

    const marca = cria("rect", {
      x: MARGEM.esquerda - 8, y: -5, width: 3, height: 10,
      fill: cores[v.grupo] || "#999",
    }, g);

    const nuvem = cria("g", { opacity: 0 }, g);
    const ys = empacota(v.contribuicao.map((c) => x(recorta(c))), RAIO);
    const pontos = v.contribuicao.map((c, i) => {
      const fora = Math.abs(c) > limite;
      const p = cria("circle", {
        cx: x(0), cy: 0, r: RAIO,
        fill: cor(v.valor_relativo[i]),
        "fill-opacity": fora ? 0.95 : 0.6,
        "stroke": fora ? "#fff" : null,
        "stroke-width": fora ? 0.6 : null,
      }, nuvem);
      p.__alvo = { cx: x(recorta(c)), cy: ys[i] };
      p.__dado = { c, rel: v.valor_relativo[i], fora };
      p.__base = fora ? 0.95 : 0.6;
      return p;
    });

    /* Interação na faixa inteira, e não ponto a ponto: com 1.200 alvos de
       2,5 px, mirar num deles seria hostil. A faixa devolve a leitura da
       variável, que é o que interessa. */
    faixa.style.cursor = "crosshair";
    faixa.addEventListener("mousemove", (e) => {
      const alvo = x.inversa(e.offsetX * (largura / hospedeiro.clientWidth));
      const perto = pontos.reduce((a, b) =>
        Math.abs(b.__dado.c - alvo) < Math.abs(a.__dado.c - alvo) ? b : a);
      for (const p of pontos) p.setAttribute("fill-opacity", 0.2);
      perto.setAttribute("fill-opacity", 1);
      perto.setAttribute("r", RAIO * 2);
      t.mostra(
        `<b>${apelido(v.nome)}</b><br>` +
        `contribuição <span class="destaque-dica">${fmt.sinal(perto.__dado.c, 4)}</span><br>` +
        `valor da variável no percentil <span class="destaque-dica">${
          fmt.n(perto.__dado.rel * 100, 0)}</span><br>` +
        (perto.__dado.fora
          ? '<span style="opacity:.75">fora da escala, desenhado na borda</span><br>'
          : "") +
        `<span style="opacity:.7">${NOME_GRUPO[v.grupo] || v.grupo}</span>`,
        e);
      perto.addEventListener("mouseleave", () => {
        perto.setAttribute("r", RAIO);
      }, { once: true });
    });
    faixa.addEventListener("mouseleave", () => {
      t.esconde();
      for (const p of pontos) {
        p.setAttribute("fill-opacity", p.__base);
        p.setAttribute("r", RAIO);
      }
    });

    linhas.set(v.nome, { g, nuvem, pontos, faixa, rotulo, marca });
  }

  /* ---------------------------------------------------------- posições */
  let ordemAtual = porShap;

  function posiciona(ordem, duracao = 0) {
    const destino = new Map(
      ordem.map((v, i) => [v.nome, MARGEM.topo + i * ALTURA_LINHA + ALTURA_LINHA / 2]));
    for (const [nome, l] of linhas) {
      const y = destino.get(nome);
      if (duracao === 0 || semMovimento()) {
        l.g.setAttribute("transform", `translate(0,${y})`);
        l.__y = y;
        continue;
      }
      const de = l.__y ?? y;
      anima(duracao, (t) => {
        const atual = de + (y - de) * t;
        l.g.setAttribute("transform", `translate(0,${atual})`);
      });
      l.__y = y;
    }
    // Faixa de fundo alternada acompanha a ordem.
    fundo.innerHTML = "";
    ordem.forEach((v, i) => {
      if (i % 2) return;
      cria("rect", {
        x: MARGEM.esquerda - 206, y: MARGEM.topo + i * ALTURA_LINHA,
        width: largura - MARGEM.esquerda - MARGEM.direita + 206,
        height: ALTURA_LINHA,
        fill: "#f4f6f8", opacity: 1,
      }, fundo);
    });
  }

  posiciona(porShap);

  /* ---------------------------------------------------------- animação */
  function solta() {
    ordemAtual.forEach((v, i) => {
      const l = linhas.get(v.nome);
      l.nuvem.setAttribute("opacity", 1);
      anima(760, (t) => {
        for (const p of l.pontos) {
          p.setAttribute("cx", x(0) + (p.__alvo.cx - x(0)) * t);
          p.setAttribute("cy", p.__alvo.cy * t);
        }
      }, { atraso: i * 55 });
    });
  }

  aoEntrar(hospedeiro, solta);

  /* ------------------------------------------------------------- troca */
  return {
    ordenaPor(chave) {
      ordemAtual = chave === "gini" ? porGini : porShap;
      posiciona(ordemAtual, 900);
      if (aoTrocarOrdem) aoTrocarOrdem(chave, ordemAtual);
    },
    ordens: { shap: porShap, gini: porGini },
  };
}

/* Legenda da rampa de cor. Desenhada em SVG para casar com o gráfico e
   acompanhar o tema claro/escuro. */
export function legendaCor(hospedeiro) {
  const cor = rampa([variavelCss("--baixo"), variavelCss("--meio"),
                     variavelCss("--alto")]);
  const s = svg(hospedeiro, 260, 34);
  const grad = cria("linearGradient", { id: "rampa-shap" },
                    cria("defs", {}, s));
  for (let i = 0; i <= 10; i++) {
    cria("stop", { offset: `${i * 10}%`, "stop-color": cor(i / 10) }, grad);
  }
  cria("rect", { x: 0, y: 6, width: 150, height: 9, fill: "url(#rampa-shap)" }, s);
  const a = cria("text", { x: 0, y: 29, class: "rotulo-eixo" }, s);
  a.textContent = "menor";
  const b = cria("text", { x: 150, y: 29, "text-anchor": "end",
                           class: "rotulo-eixo" }, s);
  b.textContent = "maior";
  const c = cria("text", { x: 162, y: 14, class: "rotulo-eixo" }, s);
  c.textContent = "valor da variável";
  return s;
}
