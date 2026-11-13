/* Página de determinantes: enxame SHAP, confronto de ordenações, permutação e ALE. */

import {
  carrega, cria, svg, linear, banda, eixoX, eixoY, aoEntrar, anima, dica,
  apelido, CORES_GRUPO, NOME_GRUPO, fmt, variavelCss, tracaCaminho, linhaSuave,
  semMovimento,
} from "./nucleo.js";
import { enxame, legendaCor } from "./enxame.js";

/* ----------------------------------------------------------------------
   Figura 2 — as duas ordenações, ligadas variável a variável.
   Um gráfico de inclinação. Linhas paralelas significariam concordância;
   o cruzamento é o resultado.
   ---------------------------------------------------------------------- */
function postos(hospedeiro, corte) {
  const vars = corte.variaveis;
  const M = { topo: 40, baixo: 24, esquerda: 200, direita: 200 };
  const passo = 21;
  const largura = 900;
  const altura = M.topo + vars.length * passo + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label",
    "Posição de cada variável na ordenação por Gini e na ordenação por permutação");
  const t = dica();

  const xg = M.esquerda, xp = largura - M.direita;
  const yDe = (posto) => M.topo + (posto - 1) * passo;

  for (const [x, rot] of [[xg, "importância de Gini"],
                          [xp, "permutação fora da amostra"]]) {
    const a = cria("text", {
      x, y: M.topo - 20, "text-anchor": "middle", class: "rotulo-eixo",
    }, s);
    a.textContent = rot;
  }

  const azul = variavelCss("--azul");
  const linhas = vars.map((v) => {
    const robusta = v.ic[0] > 0;
    const g = cria("g", {}, s);
    const y1 = yDe(v.posto_mdi), y2 = yDe(v.posto_permutacao);

    const caminho = cria("path", {
      d: linhaSuave([[xg, y1], [xp, y2]]),
      fill: "none",
      stroke: robusta ? azul : "#c6ccd2",
      "stroke-width": robusta ? 1.8 : 1.1,
      "stroke-opacity": robusta ? 0.85 : 0.6,
    }, g);

    for (const [x, y, anchor, dx] of [[xg, y1, "end", -8], [xp, y2, "start", 8]]) {
      cria("circle", {
        cx: x, cy: y, r: 3,
        fill: robusta ? azul : "#c6ccd2",
      }, g);
      const tx = cria("text", {
        x: x + dx, y: y + 3.5, "text-anchor": anchor, class: "rotulo-serie",
        "font-size": 11.5, fill: robusta ? "#333" : "#9aa2a8",
      }, g);
      tx.textContent = apelido(v.nome);
    }

    g.style.cursor = "help";
    g.addEventListener("mouseenter", () => {
      for (const o of linhas) o.g.style.opacity = 0.15;
      g.style.opacity = 1;
    });
    g.addEventListener("mousemove", (e) => t.mostra(
      `<b>${apelido(v.nome)}</b><br>` +
      `Gini <span class="destaque-dica">#${v.posto_mdi}</span> · ` +
      `permutação <span class="destaque-dica">#${v.posto_permutacao}</span><br>` +
      `importância ${fmt.sinal(v.permutacao, 5)} ` +
      `[${fmt.sinal(v.ic[0], 5)}, ${fmt.sinal(v.ic[1], 5)}]`, e));
    g.addEventListener("mouseleave", () => {
      for (const o of linhas) o.g.style.opacity = 1;
      t.esconde();
    });
    return { g, caminho };
  });

  aoEntrar(hospedeiro, () => {
    linhas.forEach((l, i) => tracaCaminho(l.caminho, 700, i * 26));
  });
}

/* ----------------------------------------------------------------------
   Figura 3 — permutação com intervalo, alternando entre os dois cortes.
   ---------------------------------------------------------------------- */
function permutacao(hospedeiro, controles, dados) {
  const cores = CORES_GRUPO();
  const M = { topo: 44, direita: 40, baixo: 52, esquerda: 210 };
  const largura = 900;
  const t = dica();

  const nomes = dados.cortes["2019"].variaveis
    .slice().sort((a, b) => b.permutacao - a.permutacao).map((v) => v.nome);
  const altura = M.topo + nomes.length * 19 + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label", "Importância por permutação com intervalo de confiança");

  const y = banda(nomes.length, [M.topo, altura - M.baixo], 0.32);
  const camadaEixo = cria("g", {}, s);
  const camadaZero = cria("g", {}, s);
  const camada = cria("g", {}, s);

  const itens = nomes.map((nome, i) => {
    const g = cria("g", { transform: `translate(0,${y(i)})` }, camada);
    const rot = cria("text", {
      x: M.esquerda - 12, y: y.largura / 2 + 4, "text-anchor": "end",
      class: "rotulo-serie", "font-size": 11.5,
    }, g);
    rot.textContent = apelido(nome);
    const barra = cria("rect", {
      x: 0, y: 1, width: 0, height: y.largura - 2, fill: "#ccc",
    }, g);
    const traco = cria("line", {
      y1: y.largura / 2, y2: y.largura / 2, stroke: "#5c6670", "stroke-width": 1.2,
    }, g);
    const alvo = cria("rect", {
      x: 0, y: -2, width: largura, height: y.largura + 4,
      fill: "transparent", cursor: "help",
    }, g);
    return { nome, g, barra, traco, rot, alvo };
  });

  let x = null;

  function desenha(chave, duracao) {
    const corte = dados.cortes[chave];
    const mapa = new Map(corte.variaveis.map((v) => [v.nome, v]));
    const lo = Math.min(0, ...corte.variaveis.map((v) => v.ic[0]));
    const hi = Math.max(...corte.variaveis.map((v) => v.ic[1]));
    const novoX = linear([lo * 1.05, hi * 1.05],
                         [M.esquerda, largura - M.direita]);

    camadaEixo.innerHTML = "";
    eixoX(camadaEixo, novoX, altura - M.baixo + 10, {
      formato: (v) => fmt.sinal(v, 3),
      titulo: "queda na PR-AUC ao embaralhar a variável", tamanho: 5,
    });

    camadaZero.innerHTML = "";
    cria("line", {
      x1: novoX(0), x2: novoX(0), y1: M.topo - 12, y2: altura - M.baixo,
      stroke: "#999", "stroke-width": 1,
    }, camadaZero);

    for (const it of itens) {
      const v = mapa.get(it.nome);
      if (!v) continue;
      const robusta = v.ic[0] > 0;
      const cor = robusta ? (cores[v.grupo] || "#888") : "#cdd3d8";
      const deV = it.__v ?? { p: 0, lo: 0, hi: 0 };
      const paraV = { p: v.permutacao, lo: v.ic[0], hi: v.ic[1] };

      it.barra.setAttribute("fill", cor);
      it.barra.setAttribute("fill-opacity", robusta ? 0.85 : 0.5);
      it.traco.setAttribute("stroke", robusta ? "#3c464e" : "#b6bcc2");
      it.rot.setAttribute("fill", robusta ? "#333" : "#9aa2a8");

      anima(duracao, (tt) => {
        const p = deV.p + (paraV.p - deV.p) * tt;
        const l = deV.lo + (paraV.lo - deV.lo) * tt;
        const h = deV.hi + (paraV.hi - deV.hi) * tt;
        const x0 = Math.min(novoX(0), novoX(p));
        it.barra.setAttribute("x", x0);
        it.barra.setAttribute("width", Math.abs(novoX(p) - novoX(0)));
        it.traco.setAttribute("x1", novoX(l));
        it.traco.setAttribute("x2", novoX(h));
      });
      it.__v = paraV;

      it.alvo.onmousemove = (e) => t.mostra(
        `<b>${apelido(v.nome)}</b> · ${NOME_GRUPO[v.grupo] || v.grupo}<br>` +
        `importância <span class="destaque-dica">${fmt.sinal(v.permutacao, 5)}</span><br>` +
        `IC 95% [${fmt.sinal(v.ic[0], 5)}, ${fmt.sinal(v.ic[1], 5)}]<br>` +
        `<span style="opacity:.75">${robusta
          ? "intervalo acima de zero" : "intervalo inclui zero"}</span>`, e);
      it.alvo.onmouseleave = () => t.esconde();
    }
    x = novoX;
  }

  /* Legenda de famílias, no topo. */
  const leg = cria("g", {}, s);
  Object.entries(NOME_GRUPO).forEach(([chave, nome], k) => {
    const gx = M.esquerda + k * 128;
    cria("rect", { x: gx, y: 4, width: 9, height: 9, fill: cores[chave] }, leg);
    const tx = cria("text", {
      x: gx + 13, y: 12.5, class: "rotulo-serie", "font-size": 11,
    }, leg);
    tx.textContent = nome;
  });

  aoEntrar(hospedeiro, () => desenha("2019", 850));

  controles.addEventListener("click", (e) => {
    const b = e.target.closest("[data-corte]");
    if (!b) return;
    for (const o of controles.querySelectorAll("[data-corte]")) {
      o.setAttribute("aria-pressed", String(o === b));
    }
    desenha(b.dataset.corte, 700);
  });
}

/* ----------------------------------------------------------------------
   Figura 4 — curvas ALE, os dois cortes sobrepostos.
   ---------------------------------------------------------------------- */
function ale(hospedeiro, controles, dados) {
  const M = { topo: 26, direita: 26, baixo: 52, esquerda: 66 };
  const largura = 860, altura = 340;
  const s = svg(hospedeiro, largura, altura);
  const t = dica();

  const de2019 = new Map(dados.cortes["2019"].map((c) => [c.variavel, c.pontos]));
  const de2014 = new Map((dados.cortes["2014"] || []).map((c) => [c.variavel, c.pontos]));
  const disponiveis = [...de2019.keys()];

  disponiveis.forEach((nome, i) => {
    const b = document.createElement("button");
    b.className = "botao";
    b.dataset.variavel = nome;
    b.setAttribute("aria-pressed", String(i === 0));
    b.textContent = apelido(nome);
    controles.appendChild(b);
  });

  function desenha(nome) {
    s.innerHTML = "";
    const a = de2019.get(nome) || [];
    const b = de2014.get(nome) || [];
    const todos = [...a, ...b];
    if (!todos.length) return;

    const x = linear([Math.min(...todos.map((p) => p.x)),
                      Math.max(...todos.map((p) => p.x))],
                     [M.esquerda, largura - M.direita]);
    const folga = (Math.max(...todos.map((p) => p.y))
                   - Math.min(...todos.map((p) => p.y))) * 0.16 || 0.001;
    const y = linear([Math.min(...todos.map((p) => p.y)) - folga,
                      Math.max(...todos.map((p) => p.y)) + folga],
                     [altura - M.baixo, M.topo]);

    eixoY(s, y, M.esquerda, {
      formato: (v) => fmt.sinal(v, 3), grade: largura - M.direita,
      titulo: "efeito na predição", tamanho: 5,
    });
    eixoX(s, x, altura - M.baixo, {
      formato: (v) => fmt.n(v, Math.abs(v) >= 100 ? 0 : 2),
      titulo: apelido(nome), tamanho: 6,
    });

    cria("line", {
      x1: M.esquerda, x2: largura - M.direita, y1: y(0), y2: y(0),
      stroke: "#bbb", "stroke-width": 1, "stroke-dasharray": "3 3",
    }, s);

    /* Densidade: onde há poucos municípios, a curva sustenta menos leitura. */
    const passos = a.map((p) => p.x);
    const faixa = cria("g", {}, s);
    for (let i = 0; i < passos.length - 1; i++) {
      cria("rect", {
        x: x(passos[i]), y: altura - M.baixo - 5,
        width: Math.max(1, x(passos[i + 1]) - x(passos[i]) - 1), height: 5,
        fill: "#9aa2a8", opacity: 0.35,
      }, faixa);
    }

    const traçados = [];
    if (b.length) {
      const p = cria("path", {
        d: linhaSuave(b.map((q) => [x(q.x), y(q.y)])),
        fill: "none", stroke: "#9aa2a8", "stroke-width": 1.6,
        "stroke-dasharray": "5 4",
      }, s);
      traçados.push(p);
    }
    const principal = cria("path", {
      d: linhaSuave(a.map((q) => [x(q.x), y(q.y)])),
      fill: "none", stroke: variavelCss("--azul"), "stroke-width": 2.4,
      "stroke-linecap": "round",
    }, s);
    traçados.push(principal);

    for (const q of a) {
      const c = cria("circle", {
        cx: x(q.x), cy: y(q.y), r: 3.2,
        fill: variavelCss("--azul"), "fill-opacity": 0.9, cursor: "help",
      }, s);
      c.addEventListener("mousemove", (e) => t.mostra(
        `${apelido(nome)} <span class="destaque-dica">${fmt.n(q.x, 3)}</span><br>` +
        `efeito <span class="destaque-dica">${fmt.sinal(q.y, 4)}</span>`, e));
      c.addEventListener("mouseleave", () => t.esconde());
    }

    /* Legenda dos dois cortes. */
    const leg = cria("g", {}, s);
    const itens = b.length
      ? [["treino até 2019", variavelCss("--azul"), null],
         ["treino até 2014", "#9aa2a8", "5 4"]]
      : [["treino até 2019", variavelCss("--azul"), null]];
    itens.forEach(([rotulo, cor, tracejado], k) => {
      const gy = M.topo + k * 16;
      cria("line", {
        x1: largura - M.direita - 150, x2: largura - M.direita - 128,
        y1: gy, y2: gy, stroke: cor, "stroke-width": 2,
        "stroke-dasharray": tracejado,
      }, leg);
      const tx = cria("text", {
        x: largura - M.direita - 122, y: gy + 4,
        class: "rotulo-serie", "font-size": 11.5,
      }, leg);
      tx.textContent = rotulo;
    });

    if (!semMovimento()) traçados.forEach((p, i) => tracaCaminho(p, 900, i * 120));
  }

  controles.addEventListener("click", (e) => {
    const b = e.target.closest("[data-variavel]");
    if (!b) return;
    for (const o of controles.querySelectorAll("[data-variavel]")) {
      o.setAttribute("aria-pressed", String(o === b));
    }
    desenha(b.dataset.variavel);
  });

  aoEntrar(hospedeiro, () => desenha(disponiveis[0]));
}

/* ------------------------------------------------------------------ início */
(async function () {
  const [sh, imp, curvas] = await Promise.all([
    carrega("shap.json"), carrega("importancia.json"), carrega("ale.json"),
  ]);

  const c19 = imp.cortes["2019"];
  const def = (id, v) => { document.getElementById(id).textContent = v; };
  def("m-roc", fmt.n(c19.roc_auc, 3));
  def("m-pr", fmt.n(c19.pr_auc, 4));
  def("m-vezes", fmt.n(c19.vezes_o_acaso, 1));
  def("m-rho-treino", fmt.sinal(c19.rho_treino, 2));
  def("m-rho-fora", fmt.sinal(c19.rho_fora, 2));
  def("m-rho-p", `p = ${fmt.n(c19.rho_fora_p, 2)}`);
  def("cap-rho", `${fmt.sinal(c19.rho_fora, 2)} (p = ${fmt.n(c19.rho_fora_p, 2)})`);
  def("m-zero", String(c19.variaveis.filter((v) => v.ic[0] <= 0).length));
  def("cap-enxame",
    `São ${fmt.inteiro(sh.n_pontos)} municípios-ano e as ${sh.n_variaveis_mostradas} `
    + `variáveis de maior contribuição média, de ${sh.n_variaveis_no_modelo} no modelo.`);

  legendaCor(document.getElementById("legenda-cor"));

  const controle = await enxame(
    document.getElementById("fig-enxame"), sh, imp);
  document.getElementById("ctrl-enxame").addEventListener("click", (e) => {
    const b = e.target.closest("[data-ordem]");
    if (!b) return;
    for (const o of document.querySelectorAll("#ctrl-enxame [data-ordem]")) {
      o.setAttribute("aria-pressed", String(o === b));
    }
    controle.ordenaPor(b.dataset.ordem);
  });

  postos(document.getElementById("fig-postos"), c19);
  permutacao(document.getElementById("fig-permutacao"),
             document.getElementById("ctrl-corte"), imp);
  ale(document.getElementById("fig-ale"),
      document.getElementById("ctrl-ale"), curvas);

  /* Tabela das variáveis que sobrevivem aos dois cortes. */
  const c14 = imp.cortes["2014"];
  const p14 = new Map(c14.variaveis.map((v) => [v.nome, v]));
  const corpo = document.querySelector("#tab-sobreviventes tbody");
  for (const nome of imp.sobrevivem_aos_dois) {
    const a = c19.variaveis.find((v) => v.nome === nome);
    const b = p14.get(nome);
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${apelido(nome)}</td><td>${NOME_GRUPO[a.grupo] || a.grupo}</td>` +
      `<td class="n">${b.posto_permutacao}</td>` +
      `<td class="n">${a.posto_permutacao}</td>` +
      `<td class="n">${a.posto_mdi}</td>`;
    corpo.appendChild(tr);
  }
})();
