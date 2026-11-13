/* Página de vacinação: estudo de evento, dupla centragem e saturação. */

import {
  carrega, cria, svg, linear, banda, eixoX, eixoY, aoEntrar, anima, dica,
  fmt, variavelCss, tracaCaminho, linhaReta, linhaSuave,
} from "./nucleo.js";

/* ----------------------------------------------------------------------
   Figura 1 — estudo de evento.
   Colunas, e não linha: cada k é uma medida separada, e a ligação contínua
   sugeriria uma trajetória de um mesmo município que não existe — são
   milhares de surtos alinhados.
   ---------------------------------------------------------------------- */
function evento(hospedeiro, controles, d) {
  const M = { topo: 44, direita: 40, baixo: 58, esquerda: 78 };
  const largura = 860, altura = 400;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label", "Cobertura vacinal centrada, em torno do ano do surto");
  const t = dica();

  const todas = [...d.todos, ...d.primeiro_surto];
  const ks = d.todos.map((p) => p.k);
  const teto = Math.max(...todas.map((p) => p.residuo_medio)) * 1.18;
  const piso = Math.min(0, ...todas.map((p) => p.residuo_medio)) - 1.2;

  const x = banda(ks.length, [M.esquerda, largura - M.direita], 0.34);
  const y = linear([piso, teto], [altura - M.baixo, M.topo]);

  eixoY(s, y, M.esquerda, {
    formato: (v) => fmt.sinal(v, 0), grade: largura - M.direita,
    titulo: "desvio da cobertura, em pontos percentuais",
  });
  cria("line", {
    x1: M.esquerda, x2: largura - M.direita, y1: y(0), y2: y(0),
    stroke: "#999", "stroke-width": 1,
  }, s);

  /* Sombreia o "depois", incluindo o próprio ano do surto. */
  const i0 = ks.indexOf(0);
  cria("rect", {
    x: x(i0) - x.largura * 0.26, y: M.topo,
    width: largura - M.direita - (x(i0) - x.largura * 0.26),
    height: altura - M.baixo - M.topo,
    fill: variavelCss("--alto"), opacity: 0.05,
  }, s);
  const marcaDepois = cria("text", {
    x: x(i0) + 4, y: M.topo - 10, class: "rotulo-eixo",
  }, s);
  marcaDepois.textContent = "surto e anos seguintes";

  const azul = variavelCss("--azul");
  const alto = variavelCss("--alto");

  const colunas = ks.map((k, i) => {
    const g = cria("g", {}, s);
    const ret = cria("rect", {
      x: x(i), width: x.largura, y: y(0), height: 0,
      fill: k >= 0 ? alto : azul, "fill-opacity": k >= 0 ? 0.85 : 0.6,
    }, g);
    const val = cria("text", {
      x: x(i) + x.largura / 2, y: 0, "text-anchor": "middle",
      class: "rotulo-valor", "font-size": 11,
    }, g);
    const alvo = cria("rect", {
      x: x(i) - x.largura * 0.2, y: M.topo,
      width: x.largura * 1.4, height: altura - M.baixo - M.topo,
      fill: "transparent", cursor: "help",
    }, g);
    return { k, ret, val, alvo, i };
  });

  eixoX(s, linear([0, ks.length - 1],
                  [x(0) + x.largura / 2, x(ks.length - 1) + x.largura / 2]),
        altura - M.baixo, {
    valores: ks.map((_, i) => i),
    formato: (i) => {
      const k = ks[Math.round(i)];
      return k === 0 ? "ano do surto" : (k > 0 ? `+${k}` : String(k));
    },
    titulo: "anos em relação ao surto",
  });

  let atual = "todos";

  function desenha(chave, duracao) {
    const serie = d[chave];
    const mapa = new Map(serie.map((p) => [p.k, p]));
    for (const c of colunas) {
      const p = mapa.get(c.k);
      if (!p) continue;
      const de = c.__v ?? 0;
      const para = p.residuo_medio;
      anima(duracao, (tt) => {
        const v = de + (para - de) * tt;
        c.ret.setAttribute("y", Math.min(y(0), y(v)));
        c.ret.setAttribute("height", Math.abs(y(v) - y(0)));
        c.val.setAttribute("y", (v >= 0 ? y(v) - 7 : y(v) + 15));
        c.val.textContent = fmt.sinal(v, 1);
      }, { atraso: c.i * 40 });
      c.__v = para;
      c.alvo.onmousemove = (e) => t.mostra(
        `<b>${c.k === 0 ? "ano do surto" : `${c.k > 0 ? "+" : ""}${c.k} ano(s)`}</b><br>` +
        `desvio médio <span class="destaque-dica">${fmt.sinal(p.residuo_medio, 2)}</span> p.p.<br>` +
        `mediana <span class="destaque-dica">${fmt.sinal(p.residuo_mediano, 2)}</span> p.p.<br>` +
        `${fmt.inteiro(p.n)} observações`, e);
      c.alvo.onmouseleave = () => t.esconde();
    }
  }

  aoEntrar(hospedeiro, () => desenha("todos", 700));

  controles.addEventListener("click", (e) => {
    const b = e.target.closest("[data-serie]");
    if (!b) return;
    for (const o of controles.querySelectorAll("[data-serie]")) {
      o.setAttribute("aria-pressed", String(o === b));
    }
    atual = b.dataset.serie;
    desenha(atual, 650);
  });
}

/* ----------------------------------------------------------------------
   Figura 2 — as três medidas de associação.
   ---------------------------------------------------------------------- */
function centragem(hospedeiro, d) {
  const itens = [
    { rotulo: "Bruta", sub: "todos os municípios-ano contra todos",
      v: d.bruta, ic: null },
    { rotulo: "Centrada no ano", sub: "remove a tendência nacional comum",
      v: d.centrada_no_ano, ic: null },
    { rotulo: "Dupla centragem", sub: "dentro do próprio município",
      v: d.dupla_centragem, ic: d.ic },
  ];
  const M = { topo: 34, direita: 80, baixo: 56, esquerda: 190 };
  const largura = 800, altura = M.topo + itens.length * 62 + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  const t = dica();

  const hi = Math.max(...itens.map((i) => i.ic ? i.ic[1] : i.v)) * 1.2;
  const x = linear([0, hi], [M.esquerda, largura - M.direita]);
  const y = banda(itens.length, [M.topo, altura - M.baixo], 0.45);

  eixoX(s, x, altura - M.baixo + 8, {
    formato: (v) => fmt.n(v, 2),
    titulo: "correlação de posto com a ocorrência de evento",
  });
  cria("line", {
    x1: x(0), x2: x(0), y1: M.topo - 8, y2: altura - M.baixo,
    stroke: "#999", "stroke-width": 1,
  }, s);

  const alto = variavelCss("--alto");
  const azul = variavelCss("--azul");

  itens.forEach((it, i) => {
    const g = cria("g", { transform: `translate(0,${y(i)})` }, s);
    const destaque = i === itens.length - 1;
    const rot = cria("text", {
      x: M.esquerda - 14, y: y.largura / 2 - 2, "text-anchor": "end",
      class: "rotulo-serie", "font-weight": destaque ? 600 : 400,
    }, g);
    rot.textContent = it.rotulo;
    const sub = cria("text", {
      x: M.esquerda - 14, y: y.largura / 2 + 14, "text-anchor": "end",
      class: "rotulo-eixo",
    }, g);
    sub.textContent = it.sub;

    const barra = cria("rect", {
      x: x(0), y: 2, width: 0, height: y.largura - 4,
      fill: destaque ? alto : azul, "fill-opacity": destaque ? 0.9 : 0.5,
    }, g);
    let traco = null;
    if (it.ic) {
      traco = cria("line", {
        x1: x(it.ic[0]), x2: x(it.ic[1]),
        y1: y.largura / 2, y2: y.largura / 2,
        stroke: "#3c464e", "stroke-width": 1.4, opacity: 0,
      }, g);
    }
    const val = cria("text", {
      x: x(0), y: y.largura / 2 + 4, class: "rotulo-valor",
    }, g);

    const alvo = cria("rect", {
      x: 0, y: -4, width: largura, height: y.largura + 8,
      fill: "transparent", cursor: "help",
    }, g);
    alvo.addEventListener("mousemove", (e) => t.mostra(
      `<b>${it.rotulo}</b><br>ρ = <span class="destaque-dica">${
        fmt.sinal(it.v, 4)}</span>` +
      (it.ic ? `<br>IC 95% [${fmt.sinal(it.ic[0], 4)}, ${
        fmt.sinal(it.ic[1], 4)}]<br><span style="opacity:.75">reamostragem por município</span>` : ""), e));
    alvo.addEventListener("mouseleave", () => t.esconde());

    aoEntrar(hospedeiro, () => {
      anima(800, (tt) => {
        const v = it.v * tt;
        barra.setAttribute("width", Math.max(0, x(v) - x(0)));
        val.setAttribute("x", x(v) + 10);
        val.textContent = fmt.sinal(v, 3);
      }, { atraso: i * 140 });
      if (traco) anima(400, (tt) => traco.setAttribute("opacity", tt),
                       { atraso: 900 });
    });
  });
}

/* ----------------------------------------------------------------------
   Figura 3 — saturação da imunidade.
   ---------------------------------------------------------------------- */
function saturacao(hospedeiro, serie) {
  const M = { topo: 30, direita: 40, baixo: 56, esquerda: 62 };
  const largura = 860, altura = 360;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label", "Imunidade acumulada entre municípios, por ano");
  const t = dica();

  const x = linear([serie[0].ano, serie[serie.length - 1].ano],
                   [M.esquerda, largura - M.direita]);
  const y = linear([0, 1], [altura - M.baixo, M.topo]);

  eixoY(s, y, M.esquerda, {
    formato: (v) => fmt.pct(v, 0), grade: largura - M.direita,
    titulo: "imunidade acumulada",
  });
  eixoX(s, x, altura - M.baixo, { formato: (v) => String(Math.round(v)), tamanho: 7 });

  const azul = variavelCss("--azul");

  /* Faixa interdecil. */
  const area = cria("path", {
    d: [...serie.map((p) => `${p === serie[0] ? "M" : "L"}${x(p.ano)},${y(p.p90)}`),
        ...serie.slice().reverse().map((p) => `L${x(p.ano)},${y(p.p10)}`), "Z"].join(""),
    fill: azul, "fill-opacity": 0.16, opacity: 0,
  }, s);

  const media = cria("path", {
    d: linhaSuave(serie.map((p) => [x(p.ano), y(p.media)])),
    fill: "none", stroke: azul, "stroke-width": 2.4,
  }, s);

  /* Marca onde a fonte do PNI termina. */
  const fim = 2022;
  cria("line", {
    x1: x(fim), x2: x(fim), y1: M.topo, y2: altura - M.baixo,
    stroke: "#999", "stroke-width": 1, "stroke-dasharray": "4 3",
  }, s);
  const rf = cria("text", {
    x: x(fim) + 6, y: M.topo + 12, class: "rotulo-eixo",
  }, s);
  rf.textContent = "fim da fonte do PNI";

  const captura = cria("rect", {
    x: M.esquerda, y: M.topo, width: largura - M.direita - M.esquerda,
    height: altura - M.baixo - M.topo, fill: "transparent", cursor: "crosshair",
  }, s);
  const regua = cria("line", {
    y1: M.topo, y2: altura - M.baixo, stroke: "#666", "stroke-width": 1, opacity: 0,
  }, s);
  captura.addEventListener("mousemove", (e) => {
    const alvo = x.inversa(e.offsetX * (largura / hospedeiro.clientWidth));
    const p = serie.reduce((a, b) =>
      Math.abs(b.ano - alvo) < Math.abs(a.ano - alvo) ? b : a);
    regua.setAttribute("x1", x(p.ano));
    regua.setAttribute("x2", x(p.ano));
    regua.setAttribute("opacity", 0.35);
    t.mostra(
      `<b>${p.ano}</b><br>média <span class="destaque-dica">${
        fmt.pct(p.media, 1)}</span><br>` +
      `decil 1 <span class="destaque-dica">${fmt.pct(p.p10, 1)}</span> · ` +
      `decil 9 <span class="destaque-dica">${fmt.pct(p.p90, 1)}</span><br>` +
      `amplitude <span class="destaque-dica">${fmt.n(p.amplitude, 3)}</span>`, e);
  });
  captura.addEventListener("mouseleave", () => {
    regua.setAttribute("opacity", 0);
    t.esconde();
  });

  aoEntrar(hospedeiro, () => {
    anima(700, (tt) => area.setAttribute("opacity", tt), { atraso: 300 });
    tracaCaminho(media, 1100);
  });
}

/* ------------------------------------------------------------------ início */
(async function () {
  const d = await carrega("vacinacao.json");

  evento(document.getElementById("fig-evento"),
         document.getElementById("ctrl-evento"), d.estudo_de_evento);
  centragem(document.getElementById("fig-centragem"), d.dentro_do_municipio);
  saturacao(document.getElementById("fig-saturacao"), d.saturacao);

  /* O salto é medido contra o ano ANTERIOR, e não contra a média dos três
     anteriores. É a razão usada em todo o registro do projeto, e é a leitura
     conservadora: o ano −1 já está levemente elevado, de modo que dividir por ele
     subestima o salto em vez de inflá-lo. */
  const ev = d.estudo_de_evento.todos;
  const anterior = ev.find((p) => p.k === -1).residuo_medio;
  const noAno = ev.find((p) => p.k === 0).residuo_medio;
  const def = (id, v) => { document.getElementById(id).textContent = v; };
  def("m-salto", `${fmt.n(noAno / anterior, 1)}×`);

  const lag = new Map(d.por_defasagem.map((p) => [p.defasagem, p]));
  def("m-lag0", fmt.sinal(lag.get(0).rho, 3));
  def("m-lag2", fmt.sinal(lag.get(2).rho, 3));

  const sat = new Map(d.saturacao.map((p) => [p.ano, p]));
  def("cap-amp-2013", fmt.n(sat.get(2013).amplitude, 3));
  def("cap-amp-2025", fmt.n(sat.get(2025).amplitude, 3));
})();
