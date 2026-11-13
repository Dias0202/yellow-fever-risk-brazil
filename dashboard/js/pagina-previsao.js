/* Página de previsão: o que faltava, o que foi acrescentado e a lista. */

import {
  carrega, cria, svg, linear, banda, eixoX, eixoY, aoEntrar, anima, dica,
  fmt, variavelCss, linhaReta, tracaCaminho, apelido,
} from "./nucleo.js";

const ROTULO = {
  forca_k50: "força de infecção · 50 km",
  forca_k200: "força de infecção · 200 km",
  forca_epiz_k100: "força de infecção por epizootia · 100 km",
  dist_foco_mais_proximo: "distância ao foco mais próximo",
  temporadas_desde_evento: "temporadas desde o último evento",
  eventos_locais_10t: "eventos locais nas 10 temporadas",
  cobertura_florestal_t1: "cobertura florestal",
  altitude_media: "altitude média",
  amplitude_altimetrica: "amplitude altimétrica",
  populacao_log_t1: "população (log)",
  tmax_temporada_normal: "temperatura máxima (normal)",
  ppt_sazonalidade_normal: "sazonalidade da chuva (normal)",
};

const MECANISMO = {
  forca_k50: "propagacao", forca_k200: "propagacao",
  forca_epiz_k100: "propagacao", dist_foco_mais_proximo: "propagacao",
  temporadas_desde_evento: "suscetibilidade", eventos_locais_10t: "suscetibilidade",
  cobertura_florestal_t1: "estrutural", altitude_media: "estrutural",
  amplitude_altimetrica: "estrutural", populacao_log_t1: "estrutural",
  tmax_temporada_normal: "estrutural", ppt_sazonalidade_normal: "estrutural",
};

/* ----------------------------------------------------------------------
   Figura 1 — os sinais ausentes, em multiplos do acaso.
   Escala logaritmica porque o que se compara sao ordens de grandeza, e uma
   escala linear faria as tres barras parecerem quase iguais.
   ---------------------------------------------------------------------- */
function contagio(hospedeiro) {
  const dados = [
    { rot: "Persistência local", sub: "evento no próprio município na temporada anterior", v: 19.0 },
    { rot: "Foco a menos de 50 km", sub: "contágio de vizinhança imediata", v: 16.8 },
    { rot: "Foco a menos de 100 km", sub: "escala do modelo final", v: 15.5 },
    { rot: "Foco a menos de 200 km", sub: "alcance longo", v: 14.3 },
  ];
  const M = { topo: 30, direita: 90, baixo: 56, esquerda: 250 };
  const largura = 860, altura = M.topo + dados.length * 56 + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  const t = dica();

  const x = linear([0, Math.max(...dados.map((d) => d.v)) * 1.12],
                   [M.esquerda, largura - M.direita]);
  const y = banda(dados.length, [M.topo, altura - M.baixo], 0.42);

  eixoX(s, x, altura - M.baixo + 8, {
    formato: (v) => `${fmt.n(v, 0)}×`,
    titulo: "vezes mais provável que a taxa de base",
  });
  cria("line", {
    x1: x(1), x2: x(1), y1: M.topo - 8, y2: altura - M.baixo,
    stroke: "#999", "stroke-width": 1, "stroke-dasharray": "4 3",
  }, s);
  const ref = cria("text", {
    x: x(1) + 5, y: M.topo - 12, class: "rotulo-eixo",
  }, s);
  ref.textContent = "sem efeito";

  const azul = variavelCss("--azul");
  dados.forEach((d, i) => {
    const g = cria("g", { transform: `translate(0,${y(i)})` }, s);
    const rot = cria("text", {
      x: M.esquerda - 14, y: y.largura / 2 - 2, "text-anchor": "end",
      class: "rotulo-serie",
    }, g);
    rot.textContent = d.rot;
    const sub = cria("text", {
      x: M.esquerda - 14, y: y.largura / 2 + 14, "text-anchor": "end",
      class: "rotulo-eixo",
    }, g);
    sub.textContent = d.sub;
    const barra = cria("rect", {
      x: x(0), y: 2, width: 0, height: y.largura - 4,
      fill: azul, "fill-opacity": 0.8,
    }, g);
    const val = cria("text", { x: 0, y: y.largura / 2 + 4, class: "rotulo-valor" }, g);

    const alvo = cria("rect", {
      x: 0, y: -4, width: largura, height: y.largura + 8,
      fill: "transparent", cursor: "help",
    }, g);
    alvo.addEventListener("mousemove", (e) => t.mostra(
      `<b>${d.rot}</b><br>${d.sub}<br>` +
      `<span class="destaque-dica">${fmt.n(d.v, 1)}×</span> a taxa de base`, e));
    alvo.addEventListener("mouseleave", () => t.esconde());

    aoEntrar(hospedeiro, () => anima(750, (tt) => {
      const v = d.v * tt;
      barra.setAttribute("width", Math.max(0, x(v) - x(0)));
      val.setAttribute("x", x(v) + 12);
      val.textContent = `${fmt.n(v, 1)}×`;
    }, { atraso: i * 110 }));
  });
}

/* ----------------------------------------------------------------------
   Figura 2 — sazonalidade e o corte da temporada.
   ---------------------------------------------------------------------- */
function sazonalidade(hospedeiro, casos, epiz) {
  const meses = ["jan", "fev", "mar", "abr", "mai", "jun",
                 "jul", "ago", "set", "out", "nov", "dez"];
  const M = { topo: 40, direita: 30, baixo: 56, esquerda: 62 };
  const largura = 860, altura = 340;
  const s = svg(hospedeiro, largura, altura);
  const t = dica();

  const vc = meses.map((_, i) => casos[String(i + 1)] || 0);
  const ve = meses.map((_, i) => epiz[String(i + 1)] || 0);
  const teto = Math.max(...vc, ...ve) * 1.12;

  const x = banda(12, [M.esquerda, largura - M.direita], 0.3);
  const y = linear([0, teto], [altura - M.baixo, M.topo]);

  eixoY(s, y, M.esquerda, {
    formato: fmt.inteiro, grade: largura - M.direita, titulo: "registros",
  });

  /* Marca o inicio da temporada de monitoramento, entre junho e julho. */
  const corte = x(6) - x.passo * 0.16;
  cria("line", {
    x1: corte, x2: corte, y1: M.topo - 14, y2: altura - M.baixo,
    stroke: variavelCss("--alto"), "stroke-width": 1.6,
  }, s);
  const rc = cria("text", {
    x: corte + 6, y: M.topo - 18, class: "rotulo-eixo",
    fill: variavelCss("--alto"),
  }, s);
  rc.textContent = "início da temporada de monitoramento";

  const azul = variavelCss("--azul");
  const verde = variavelCss("--paisagem");
  const series = [
    { v: vc, cor: azul, rot: "casos humanos", desl: 0 },
    { v: ve, cor: verde, rot: "epizootias", desl: x.largura / 2 },
  ];

  series.forEach((serie, k) => {
    serie.v.forEach((valor, i) => {
      const ret = cria("rect", {
        x: x(i) + serie.desl, width: x.largura / 2,
        y: y(0), height: 0, fill: serie.cor, "fill-opacity": 0.85,
      }, s);
      const alvo = cria("rect", {
        x: x(i), y: M.topo, width: x.largura,
        height: altura - M.baixo - M.topo, fill: "transparent", cursor: "help",
      }, s);
      alvo.addEventListener("mousemove", (e) => t.mostra(
        `<b>${meses[i]}</b><br>` +
        `casos humanos <span class="destaque-dica">${fmt.inteiro(vc[i])}</span><br>` +
        `epizootias <span class="destaque-dica">${fmt.inteiro(ve[i])}</span>`, e));
      alvo.addEventListener("mouseleave", () => t.esconde());
      aoEntrar(hospedeiro, () => anima(620, (tt) => {
        ret.setAttribute("y", y(valor * tt));
        ret.setAttribute("height", y(0) - y(valor * tt));
      }, { atraso: i * 30 + k * 90 }));
    });
  });

  eixoX(s, linear([0, 11], [x(0) + x.largura / 2, x(11) + x.largura / 2]),
        altura - M.baixo, {
    valores: meses.map((_, i) => i), formato: (i) => meses[Math.round(i)],
  });

  // Legenda dentro da area do grafico, a direita: o topo esta ocupado pelo
  // rotulo do corte da temporada.
  const leg = cria("g", {}, s);
  series.forEach((serie, k) => {
    const gy = M.topo + 10 + k * 18;
    const gx = largura - M.direita - 150;
    cria("rect", { x: gx, y: gy - 8, width: 9, height: 9, fill: serie.cor }, leg);
    const tx = cria("text", { x: gx + 14, y: gy, class: "rotulo-serie" }, leg);
    tx.textContent = serie.rot;
  });
}

/* ----------------------------------------------------------------------
   Figura 3 — ganho por conjunto de variaveis.
   ---------------------------------------------------------------------- */
function ganho(hospedeiro, conjuntos) {
  const ordem = [
    ["rf_ambiental", "Ambiental", "clima, paisagem, relevo, demografia — adequabilidade estática de habitat"],
    ["rf_suscetibilidade", "+ suscetibilidade", "histórico local e refratariedade"],
    ["rf_propagacao", "+ propagação", "força de infecção, distância ao foco, frente de onda"],
    ["rf_completo", "Completo", "todos os blocos juntos"],
  ].filter(([k]) => conjuntos[k] !== undefined);

  const M = { topo: 30, direita: 90, baixo: 56, esquerda: 220 };
  const largura = 860, altura = M.topo + ordem.length * 58 + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  const t = dica();

  const vals = ordem.map(([k]) => conjuntos[k]);
  const x = linear([0, Math.max(...vals) * 1.14], [M.esquerda, largura - M.direita]);
  const y = banda(ordem.length, [M.topo, altura - M.baixo], 0.4);

  eixoX(s, x, altura - M.baixo + 8, {
    formato: (v) => `${fmt.n(v, 0)}×`,
    titulo: "PR-AUC em múltiplos do acaso, validação prospectiva",
  });

  const azul = variavelCss("--azul");
  const alto = variavelCss("--alto");
  ordem.forEach(([k, rot, sub], i) => {
    const v = conjuntos[k];
    const salto = i === 2;
    const g = cria("g", { transform: `translate(0,${y(i)})` }, s);
    const r = cria("text", {
      x: M.esquerda - 14, y: y.largura / 2 - 2, "text-anchor": "end",
      class: "rotulo-serie", "font-weight": salto ? 600 : 400,
    }, g);
    r.textContent = rot;
    const sb = cria("text", {
      x: M.esquerda - 14, y: y.largura / 2 + 14, "text-anchor": "end",
      class: "rotulo-eixo",
    }, g);
    sb.textContent = sub;
    const barra = cria("rect", {
      x: x(0), y: 2, width: 0, height: y.largura - 4,
      fill: salto ? alto : azul, "fill-opacity": salto ? 0.9 : 0.55,
    }, g);
    const val = cria("text", { x: 0, y: y.largura / 2 + 4, class: "rotulo-valor" }, g);

    const alvo = cria("rect", {
      x: 0, y: -4, width: largura, height: y.largura + 8,
      fill: "transparent", cursor: "help",
    }, g);
    alvo.addEventListener("mousemove", (e) => t.mostra(
      `<b>${rot}</b><br>${sub}<br>` +
      `<span class="destaque-dica">${fmt.n(v, 2)}×</span> o acaso`, e));
    alvo.addEventListener("mouseleave", () => t.esconde());

    aoEntrar(hospedeiro, () => anima(760, (tt) => {
      const a = v * tt;
      barra.setAttribute("width", Math.max(0, x(a) - x(0)));
      val.setAttribute("x", x(a) + 12);
      val.textContent = `${fmt.n(a, 2)}×`;
    }, { atraso: i * 130 }));
  });
}

/* ----------------------------------------------------------------------
   Figura 4 — coeficientes do modelo final.
   ---------------------------------------------------------------------- */
function coeficientes(hospedeiro, coefs) {
  const dados = [...coefs].sort((a, b) => b.coef - a.coef);
  const M = { topo: 40, direita: 80, baixo: 56, esquerda: 290 };
  const largura = 880, altura = M.topo + dados.length * 30 + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  const t = dica();

  const lim = Math.max(...dados.map((d) => Math.abs(d.coef))) * 1.15;
  const x = linear([-lim, lim], [M.esquerda, largura - M.direita]);
  const y = banda(dados.length, [M.topo, altura - M.baixo], 0.3);

  eixoX(s, x, altura - M.baixo + 8, {
    formato: (v) => fmt.sinal(v, 1),
    titulo: "coeficiente em escala padronizada",
  });
  cria("line", {
    x1: x(0), x2: x(0), y1: M.topo - 10, y2: altura - M.baixo,
    stroke: "#999", "stroke-width": 1,
  }, s);

  const cores = {
    propagacao: variavelCss("--azul"),
    suscetibilidade: variavelCss("--paisagem"),
    estrutural: variavelCss("--demografia"),
  };

  dados.forEach((d, i) => {
    const g = cria("g", { transform: `translate(0,${y(i)})` }, s);
    const rot = cria("text", {
      x: M.esquerda - 12, y: y.largura / 2 + 4, "text-anchor": "end",
      class: "rotulo-serie", "font-size": 12,
    }, g);
    rot.textContent = ROTULO[d.variavel] || apelido(d.variavel);
    const cor = cores[MECANISMO[d.variavel]] || "#999";
    const barra = cria("rect", {
      x: x(0), y: 1, width: 0, height: y.largura - 2,
      fill: cor, "fill-opacity": 0.85,
    }, g);
    const val = cria("text", {
      x: 0, y: y.largura / 2 + 4, class: "rotulo-valor", "font-size": 11,
    }, g);

    const alvo = cria("rect", {
      x: 0, y: -2, width: largura, height: y.largura + 4,
      fill: "transparent", cursor: "help",
    }, g);
    alvo.addEventListener("mousemove", (e) => t.mostra(
      `<b>${ROTULO[d.variavel] || d.variavel}</b><br>` +
      `coeficiente <span class="destaque-dica">${fmt.sinal(d.coef, 3)}</span><br>` +
      `<span style="opacity:.75">${MECANISMO[d.variavel] || ""}</span>`, e));
    alvo.addEventListener("mouseleave", () => t.esconde());

    aoEntrar(hospedeiro, () => anima(700, (tt) => {
      const v = d.coef * tt;
      barra.setAttribute("x", Math.min(x(0), x(v)));
      barra.setAttribute("width", Math.abs(x(v) - x(0)));
      const fim = v >= 0 ? x(v) + 8 : x(v) - 8;
      val.setAttribute("x", fim);
      val.setAttribute("text-anchor", v >= 0 ? "start" : "end");
      val.textContent = fmt.sinal(v, 2);
    }, { atraso: i * 45 }));
  });

  const leg = cria("g", {}, s);
  Object.entries({ propagacao: "propagação", suscetibilidade: "suscetibilidade",
                   estrutural: "adequabilidade estrutural" })
    .forEach(([k, rot], i) => {
      const gx = M.esquerda + i * 170;
      cria("rect", { x: gx, y: 8, width: 9, height: 9, fill: cores[k] }, leg);
      const tx = cria("text", { x: gx + 14, y: 16.5, class: "rotulo-serie",
                                "font-size": 11.5 }, leg);
      tx.textContent = rot;
    });
}

/* ----------------------------------------------------------------------
   Figura 5 — calibracao previsto x observado.
   ---------------------------------------------------------------------- */
function calibracao(hospedeiro, calib) {
  const M = { topo: 30, direita: 40, baixo: 58, esquerda: 74 };
  const largura = 700, altura = 460;
  const s = svg(hospedeiro, largura, altura);
  const t = dica();

  const todos = calib.flatMap((c) => [c.previsto, c.observado]).filter((v) => v > 0);
  const lo = Math.log10(Math.min(...todos) * 0.7);
  const hi = Math.log10(Math.max(...todos) * 1.5);
  const x = linear([lo, hi], [M.esquerda, largura - M.direita]);
  const y = linear([lo, hi], [altura - M.baixo, M.topo]);
  const marcas = [];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) marcas.push(e);
  const rot = (e) => {
    const v = Math.pow(10, e);
    return v >= 0.01 ? fmt.pct(v, 0) : `${fmt.n(v * 100, 2)}%`;
  };

  eixoY(s, y, M.esquerda, {
    valores: marcas, formato: rot, grade: largura - M.direita,
    titulo: "frequência observada",
  });
  eixoX(s, x, altura - M.baixo, {
    valores: marcas, formato: rot, titulo: "probabilidade prevista",
  });

  const diag = cria("path", {
    d: linhaReta([[x(lo), y(lo)], [x(hi), y(hi)]]),
    fill: "none", stroke: "#999", "stroke-width": 1.2, "stroke-dasharray": "5 4",
  }, s);
  const dt = cria("text", {
    x: x(hi) - 6, y: y(hi) + 16, "text-anchor": "end", class: "rotulo-eixo",
  }, s);
  dt.textContent = "calibração perfeita";

  const azul = variavelCss("--azul");
  const pts = calib.filter((c) => c.previsto > 0 && c.observado > 0);
  const caminho = cria("path", {
    d: linhaReta(pts.map((c) => [x(Math.log10(c.previsto)),
                                 y(Math.log10(c.observado))])),
    fill: "none", stroke: azul, "stroke-width": 1.8, "stroke-opacity": 0.55,
  }, s);

  const marcasPt = pts.map((c) => {
    const p = cria("circle", {
      cx: x(Math.log10(c.previsto)), cy: y(Math.log10(c.observado)),
      r: 6, fill: azul, stroke: "#fff", "stroke-width": 1.5,
      opacity: 0, cursor: "help",
    }, s);
    p.addEventListener("mousemove", (e) => t.mostra(
      `<b>Decil ${c.faixa}</b> · ${fmt.inteiro(c.n)} municípios-temporada<br>` +
      `previsto <span class="destaque-dica">${fmt.n(c.previsto * 100, 3)}%</span><br>` +
      `observado <span class="destaque-dica">${fmt.n(c.observado * 100, 3)}%</span>`, e));
    p.addEventListener("mouseleave", () => t.esconde());
    return p;
  });

  aoEntrar(hospedeiro, () => {
    tracaCaminho(caminho, 900);
    marcasPt.forEach((p, i) => anima(300, (tt) => p.setAttribute("opacity", tt),
                                     { atraso: 250 + i * 70 }));
  });
}

/* ------------------------------------------------------------------ lista */
function lista(corpo, controles, topo) {
  let ordem = "posto_logistica";
  let regiao = "";

  function desenha() {
    const filtrado = topo
      .filter((r) => !regiao || r.regiao === regiao)
      .slice()
      .sort((a, b) => a[ordem] - b[ordem])
      .slice(0, 100);
    corpo.innerHTML = "";
    for (const r of filtrado) {
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td class="n">${r[ordem]}</td>` +
        `<td>${r.nome_mun}</td><td>${r.uf}</td>` +
        `<td class="n">${fmt.n(r.probabilidade * 100, 2)}%</td>` +
        `<td class="n">${fmt.n(r.dist_foco_mais_proximo, 0)} km</td>` +
        `<td class="n">${fmt.inteiro(r.focos_em_100km)}</td>` +
        `<td class="n">${fmt.inteiro(r.temporadas_desde_evento)}</td>`;
      corpo.appendChild(tr);
    }
  }

  /* Botões de região montados a partir do que a lista de fato contém. */
  const regioes = [...new Set(topo.map((r) => r.regiao))].filter(Boolean).sort();
  for (const r of regioes) {
    const b = document.createElement("button");
    b.className = "botao";
    b.dataset.regiao = r;
    b.setAttribute("aria-pressed", "false");
    b.textContent = r;
    controles.appendChild(b);
  }

  controles.addEventListener("click", (e) => {
    const bo = e.target.closest("[data-ordem]");
    const br = e.target.closest("[data-regiao]");
    if (bo) {
      for (const o of controles.querySelectorAll("[data-ordem]")) {
        o.setAttribute("aria-pressed", String(o === bo));
      }
      ordem = bo.dataset.ordem;
    }
    if (br) {
      for (const o of controles.querySelectorAll("[data-regiao]")) {
        o.setAttribute("aria-pressed", String(o === br));
      }
      regiao = br.dataset.regiao;
    }
    if (bo || br) desenha();
  });

  desenha();
}

/* ------------------------------------------------------------------ início */
(async function () {
  const d = await carrega("previsao.json");
  const t = d.temporada_prevista;
  const rot = `${t}/${String(t + 1).slice(2)}`;
  const def = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.textContent = v;
  };
  def("rot-temporada", rot);
  def("rot-temporada-2", rot);

  const log = d.desempenho.logistica;
  const forca = d.desempenho.forca;
  def("m-temporadas", String(log.temporadas));
  def("m-vezes", `${fmt.n(forca.pr_auc_sobre_prevalencia, 1)}×`);
  def("m-rec500", fmt.pct(Math.max(log.recall_top500, forca.recall_top500), 0));
  def("m-rec1000", fmt.pct(forca.recall_top1000, 0));

  const probs = d.topo.map((r) => r.probabilidade);
  def("m-faixa", `até ${fmt.n(Math.max(...probs) * 100, 1)}%`);

  contagio(document.getElementById("fig-contagio"));
  sazonalidade(document.getElementById("fig-sazonalidade"),
               d.sazonalidade_casos, d.sazonalidade_epizootias);
  ganho(document.getElementById("fig-ganho"), d.ganho_por_conjunto);
  coeficientes(document.getElementById("fig-coeficientes"), d.coeficientes);
  calibracao(document.getElementById("fig-calibracao"), d.calibracao);
  lista(document.querySelector("#tab-lista tbody"),
        document.getElementById("ctrl-lista"), d.topo);
})();
