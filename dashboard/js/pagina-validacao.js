/* Página de validação: transferência regional, calibração e curva de decisão. */

import {
  carrega, cria, svg, linear, banda, eixoX, eixoY, aoEntrar, anima, dica,
  fmt, variavelCss, tracaCaminho, linhaReta,
} from "./nucleo.js";

/* ----------------------------------------------------------------------
   Figura 1 — ROC-AUC por macrorregião, com intervalo.
   Ponto e barra, e não coluna: o que se lê aqui é a posição do intervalo em
   relação a 0,5, e uma coluna a partir de zero desperdiçaria a escala inteira
   na parte que não tem informação.
   ---------------------------------------------------------------------- */
function regioes(hospedeiro, d) {
  const dados = [...d.por_regiao].sort((a, b) => a.roc_auc - b.roc_auc);
  const M = { topo: 30, direita: 130, baixo: 52, esquerda: 130 };
  const largura = 880;
  const altura = M.topo + dados.length * 52 + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label", "ROC-AUC por macrorregião com intervalo de confiança");
  const t = dica();

  const lo = Math.min(0.42, ...dados.map((r) => r.ic[0])) - 0.02;
  const hi = Math.max(...dados.map((r) => r.ic[1])) + 0.03;
  const x = linear([lo, hi], [M.esquerda, largura - M.direita]);
  const y = banda(dados.length, [M.topo, altura - M.baixo], 0.5);

  eixoX(s, x, altura - M.baixo + 8, {
    formato: (v) => fmt.n(v, 2), titulo: "ROC-AUC", tamanho: 6,
  });

  /* Referência do sorteio. */
  cria("line", {
    x1: x(0.5), x2: x(0.5), y1: M.topo - 10, y2: altura - M.baixo,
    stroke: "#999", "stroke-width": 1, "stroke-dasharray": "4 3",
  }, s);
  const refT = cria("text", {
    x: x(0.5), y: M.topo - 16, "text-anchor": "middle", class: "rotulo-eixo",
  }, s);
  refT.textContent = "sorteio";

  const azul = variavelCss("--azul");
  dados.forEach((r, i) => {
    const transfere = r.ic[0] > 0.5;
    const cy = y(i) + y.largura / 2;
    const g = cria("g", {}, s);

    const rot = cria("text", {
      x: M.esquerda - 14, y: cy + 4, "text-anchor": "end", class: "rotulo-serie",
      "font-weight": transfere ? 600 : 400,
      fill: transfere ? "#222" : "#666",
    }, g);
    rot.textContent = r.regiao;

    const barra = cria("line", {
      x1: x(0.5), x2: x(0.5), y1: cy, y2: cy,
      stroke: transfere ? azul : "#b8c0c6", "stroke-width": 3,
      "stroke-linecap": "round",
    }, g);
    const ponto = cria("circle", {
      cx: x(0.5), cy, r: 5.5,
      fill: transfere ? azul : "#8d979e",
      stroke: "#fff", "stroke-width": 1.5,
    }, g);
    const valor = cria("text", {
      x: 0, y: cy + 4, class: "rotulo-valor",
      fill: transfere ? "#222" : "#777",
    }, g);

    const sub = cria("text", {
      x: M.esquerda - 14, y: cy + 19, "text-anchor": "end", class: "rotulo-eixo",
    }, g);
    sub.textContent = `${fmt.inteiro(r.positivos)} eventos · ${
      fmt.n(r.vezes_o_acaso, 1)}× o acaso`;

    const alvo = cria("rect", {
      x: 0, y: cy - 20, width: largura, height: 40,
      fill: "transparent", cursor: "help",
    }, g);
    alvo.addEventListener("mousemove", (e) => t.mostra(
      `<b>${r.regiao}</b><br>` +
      `ROC-AUC <span class="destaque-dica">${fmt.n(r.roc_auc, 3)}</span> ` +
      `[${fmt.n(r.ic[0], 3)}, ${fmt.n(r.ic[1], 3)}]<br>` +
      `PR-AUC ${fmt.n(r.pr_auc, 4)} · ${fmt.n(r.vezes_o_acaso, 1)}× o acaso<br>` +
      `${fmt.inteiro(r.positivos)} eventos em ${fmt.inteiro(r.n)} municípios-ano<br>` +
      `<span style="opacity:.75">${transfere
        ? "intervalo inteiramente acima do sorteio"
        : "intervalo cruza o sorteio"}</span>`, e));
    alvo.addEventListener("mouseleave", () => t.esconde());

    aoEntrar(hospedeiro, () => {
      anima(800, (tt) => {
        const a = 0.5 + (r.ic[0] - 0.5) * tt;
        const b = 0.5 + (r.ic[1] - 0.5) * tt;
        const c = 0.5 + (r.roc_auc - 0.5) * tt;
        barra.setAttribute("x1", x(a));
        barra.setAttribute("x2", x(b));
        ponto.setAttribute("cx", x(c));
        valor.setAttribute("x", x(b) + 12);
        valor.textContent = fmt.n(c, 3);
      }, { atraso: i * 90 });
    });
  });
}

/* ----------------------------------------------------------------------
   Figura 2 — calibração previsto contra observado, em escala log.
   Log porque as faixas cobrem três ordens de grandeza de probabilidade, e em
   escala linear as oito primeiras colapsariam sobre a origem.
   ---------------------------------------------------------------------- */
function calibracao(hospedeiro, d) {
  const pontos = d.calibracao.filter((c) => c.previsto > 0 && c.observado > 0);
  const M = { topo: 30, direita: 40, baixo: 56, esquerda: 74 };
  const largura = 700, altura = 470;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label", "Calibração: probabilidade prevista contra frequência observada");
  const t = dica();

  const todos = pontos.flatMap((c) => [c.previsto, c.observado]);
  const lo = Math.log10(Math.min(...todos) * 0.7);
  const hi = Math.log10(Math.max(...todos) * 1.4);
  const x = linear([lo, hi], [M.esquerda, largura - M.direita]);
  const y = linear([lo, hi], [altura - M.baixo, M.topo]);
  const marcasLog = [];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) marcasLog.push(e);

  const rotLog = (e) => {
    const v = Math.pow(10, e);
    return v >= 0.01 ? fmt.pct(v, v >= 0.1 ? 0 : 1) : `${fmt.n(v * 100, 2)}%`;
  };

  eixoY(s, y, M.esquerda, {
    valores: marcasLog, formato: rotLog, grade: largura - M.direita,
    titulo: "frequência observada",
  });
  eixoX(s, x, altura - M.baixo, {
    valores: marcasLog, formato: rotLog,
    titulo: "probabilidade média prevista",
  });

  /* Diagonal de calibração perfeita. */
  const diag = cria("path", {
    d: linhaReta([[x(lo), y(lo)], [x(hi), y(hi)]]),
    fill: "none", stroke: "#999", "stroke-width": 1.2, "stroke-dasharray": "5 4",
  }, s);
  const dt = cria("text", {
    x: x(hi) - 6, y: y(hi) + 16, "text-anchor": "end", class: "rotulo-eixo",
  }, s);
  dt.textContent = "calibração perfeita";

  const azul = variavelCss("--azul");
  const alto = variavelCss("--alto");
  const caminho = cria("path", {
    d: linhaReta(pontos.map((c) => [x(Math.log10(c.previsto)),
                                    y(Math.log10(c.observado))])),
    fill: "none", stroke: azul, "stroke-width": 2, "stroke-opacity": 0.6,
  }, s);

  const marcas = pontos.map((c, i) => {
    const ultimo = i === pontos.length - 1;
    const p = cria("circle", {
      cx: x(Math.log10(c.previsto)), cy: y(Math.log10(c.observado)),
      r: ultimo ? 7 : 5, fill: ultimo ? alto : azul,
      stroke: "#fff", "stroke-width": 1.5, opacity: 0, cursor: "help",
    }, s);
    const razao = c.observado / c.previsto;
    p.addEventListener("mousemove", (e) => t.mostra(
      `<b>Faixa ${c.faixa + 1}</b> · ${fmt.inteiro(c.n)} municípios-ano<br>` +
      `previsto <span class="destaque-dica">${fmt.n(c.previsto * 100, 3)}%</span><br>` +
      `observado <span class="destaque-dica">${fmt.n(c.observado * 100, 3)}%</span><br>` +
      `${razao > 1 ? "subestimado" : "superestimado"} por ` +
      `<span class="destaque-dica">${fmt.n(razao > 1 ? razao : 1 / razao, 1)}×</span>`, e));
    p.addEventListener("mouseleave", () => t.esconde());
    return p;
  });

  /* Rótulo do decil de topo, que é o que sustenta a legenda da figura. */
  const topo = pontos[pontos.length - 1];
  const rot = cria("text", {
    x: x(Math.log10(topo.previsto)) - 12,
    y: y(Math.log10(topo.observado)) - 12,
    "text-anchor": "end", class: "rotulo-serie", "font-weight": 600,
    fill: alto, opacity: 0,
  }, s);
  rot.textContent = "decil de maior risco previsto";

  aoEntrar(hospedeiro, () => {
    tracaCaminho(caminho, 900);
    marcas.forEach((p, i) => anima(300, (tt) => p.setAttribute("opacity", tt),
                                   { atraso: 250 + i * 70 }));
    anima(400, (tt) => rot.setAttribute("opacity", tt), { atraso: 1100 });
  });

  return topo.previsto / topo.observado;
}

/* ----------------------------------------------------------------------
   Figura 3 — curva de decisão.
   ---------------------------------------------------------------------- */
function decisao(hospedeiro, d) {
  const c = d.curva_de_decisao;
  const M = { topo: 30, direita: 130, baixo: 56, esquerda: 78 };
  const largura = 860, altura = 400;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label", "Curva de decisão clínica sob bloco espaço-temporal");
  const t = dica();

  const x = linear([Math.min(...c.map((p) => p.limiar)),
                    Math.max(...c.map((p) => p.limiar))],
                   [M.esquerda, largura - M.direita]);
  const valores = c.flatMap((p) => [p.modelo, p.todos, 0]);
  const y = linear([Math.min(...valores) * 1.05, Math.max(...valores) * 1.1],
                   [altura - M.baixo, M.topo]);

  eixoY(s, y, M.esquerda, {
    formato: (v) => fmt.n(v, 3), grade: largura - M.direita,
    titulo: "benefício líquido",
  });
  eixoX(s, x, altura - M.baixo, {
    formato: (v) => fmt.pct(v, 1), titulo: "limiar de probabilidade", tamanho: 6,
  });

  const series = [
    { chave: "todos", rotulo: "investigar todos", cor: "#8d979e", tracejado: "5 4" },
    { chave: "nenhum", rotulo: "não investigar nenhum", cor: "#c0c7cc", tracejado: "2 3" },
    { chave: "modelo", rotulo: "seguir o modelo", cor: variavelCss("--azul"), tracejado: null },
  ];

  const caminhos = series.map((serie) => {
    const pts = c.map((p) => [x(p.limiar),
                              y(serie.chave === "nenhum" ? 0 : p[serie.chave])]);
    const path = cria("path", {
      d: linhaReta(pts), fill: "none", stroke: serie.cor,
      "stroke-width": serie.chave === "modelo" ? 2.6 : 1.6,
      "stroke-dasharray": serie.tracejado,
    }, s);
    const fim = pts[pts.length - 1];
    const rot = cria("text", {
      x: largura - M.direita + 10,
      y: (serie.chave === "modelo" ? fim[1] : serie.chave === "todos"
          ? y(c[c.length - 1].todos) : y(0)) + 4,
      class: "rotulo-serie", "font-size": 11.5, fill: serie.cor,
    }, s);
    rot.textContent = serie.rotulo;
    return path;
  });

  /* Leitura por posição do cursor: uma régua vertical é mais informativa que
     acertar um ponto sobre a curva. */
  const regua = cria("line", {
    y1: M.topo, y2: altura - M.baixo, stroke: "#666", "stroke-width": 1,
    opacity: 0,
  }, s);
  const captura = cria("rect", {
    x: M.esquerda, y: M.topo, width: largura - M.direita - M.esquerda,
    height: altura - M.baixo - M.topo, fill: "transparent", cursor: "crosshair",
  }, s);
  captura.addEventListener("mousemove", (e) => {
    const alvo = x.inversa(e.offsetX * (largura / hospedeiro.clientWidth));
    const p = c.reduce((a, b) =>
      Math.abs(b.limiar - alvo) < Math.abs(a.limiar - alvo) ? b : a);
    regua.setAttribute("x1", x(p.limiar));
    regua.setAttribute("x2", x(p.limiar));
    regua.setAttribute("opacity", 0.35);
    t.mostra(
      `limiar <b>${fmt.pct(p.limiar, 2)}</b><br>` +
      `seguir o modelo <span class="destaque-dica">${fmt.n(p.modelo, 4)}</span><br>` +
      `investigar todos <span class="destaque-dica">${fmt.n(p.todos, 4)}</span><br>` +
      `não investigar <span class="destaque-dica">0</span>`, e);
  });
  captura.addEventListener("mouseleave", () => {
    regua.setAttribute("opacity", 0);
    t.esconde();
  });

  aoEntrar(hospedeiro, () => {
    caminhos.forEach((p, i) => tracaCaminho(p, 1000, i * 160));
  });
}

/* ------------------------------------------------------------------ início */
(async function () {
  const d = await carrega("validacao.json");

  const corpo = document.querySelector("#tab-esquemas tbody");
  d.esquemas.forEach((e, i) => {
    const tr = document.createElement("tr");
    if (i === d.esquemas.length - 1) tr.className = "realce";
    const nota = i === 0 ? " <span style=\"color:#888\">(ingênua)</span>"
      : i === d.esquemas.length - 1 ? " <span style=\"color:#888\">(uso pretendido)</span>"
      : "";
    tr.innerHTML =
      `<td>${e.nome}${nota}<br><span style="font-size:.85em;color:#777">${
        e.explicacao}</span></td>` +
      `<td class="n">${fmt.n(e.roc_auc, 3)}</td>` +
      `<td class="n">${fmt.n(e.pr_auc, 4)}</td>` +
      `<td class="n">${fmt.n(e.vezes_o_acaso, 1)}×</td>` +
      `<td class="n">${fmt.n(e.investigados_por_evento, 1)}</td>`;
    corpo.appendChild(tr);
  });

  regioes(document.getElementById("fig-regiao"), d);
  const fator = calibracao(document.getElementById("fig-calibracao"), d);
  decisao(document.getElementById("fig-decisao"), d);

  document.getElementById("cap-fator").textContent = `${fmt.n(fator, 1)}×`;
  document.getElementById("cap-limiares").textContent = String(d.limiares_uteis);
})();
