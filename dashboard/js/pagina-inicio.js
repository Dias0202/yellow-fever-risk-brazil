/* Página de abertura: a série observada e o colapso do desempenho. */

import {
  carrega, cria, svg, linear, banda, eixoX, eixoY, aoEntrar, anima, dica,
  fmt, variavelCss, marcas, semMovimento,
} from "./nucleo.js";

const CORES_REGIAO = () => ({
  Norte: variavelCss("--clima"),
  Nordeste: variavelCss("--demografia"),
  Sudeste: variavelCss("--imunidade"),
  Sul: variavelCss("--paisagem"),
  "Centro-Oeste": variavelCss("--clima-def"),
});

/* ----------------------------------------------------------------------
   Figura 1 — série empilhada por macrorregião.
   Colunas empilhadas e não linhas: o que interessa é o total e a composição,
   e uma linha por região com quatro delas rentes a zero seria ilegível.
   ---------------------------------------------------------------------- */
function serie(hospedeiro, d) {
  const M = { topo: 50, direita: 20, baixo: 44, esquerda: 52 };
  const largura = 1000, altura = 400;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label",
    "Municípios-ano com evento de febre amarela por ano e macrorregião, 1994 a 2025");
  const cores = CORES_REGIAO();
  const t = dica();

  const totais = d.anos.map((_, i) =>
    d.regioes.reduce((a, r) => a + d.por_regiao[r][i], 0));
  const teto = Math.max(...totais);

  const x = banda(d.anos.length, [M.esquerda, largura - M.direita], 0.26);
  const y = linear([0, teto * 1.06], [altura - M.baixo, M.topo]);

  eixoY(s, y, M.esquerda, {
    formato: fmt.inteiro, grade: largura - M.direita,
    titulo: "municípios-ano com evento", tamanho: 5,
  });

  /* Anos sem vigilância de epizootia: hachura, não cor. É ausência de dado, e
     tem de se distinguir visualmente de um valor baixo. */
  const defs = cria("defs", {}, s);
  const hachura = cria("pattern", {
    id: "sem-vigilancia", width: 5, height: 5,
    patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)",
  }, defs);
  cria("line", {
    x1: 0, y1: 0, x2: 0, y2: 5,
    stroke: "#c9c9c9", "stroke-width": 1.6,
  }, hachura);

  const semVig = new Set(d.sem_vigilancia_epizootias);
  d.anos.forEach((ano, i) => {
    if (!semVig.has(ano)) return;
    cria("rect", {
      x: x(i) - x.largura * 0.16, y: M.topo - 6,
      width: x.largura * 1.32, height: altura - M.baixo - M.topo + 6,
      fill: "url(#sem-vigilancia)", opacity: 0.5,
    }, s);
  });

  const camada = cria("g", {}, s);
  const colunas = d.anos.map((ano, i) => {
    const g = cria("g", {}, camada);
    let acumulado = 0;
    const partes = d.regioes.map((r) => {
      const v = d.por_regiao[r][i];
      const base = acumulado;
      acumulado += v;
      const ret = cria("rect", {
        x: x(i), width: x.largura,
        y: y(0), height: 0,
        fill: cores[r] || "#999",
        "fill-opacity": 0.9,
      }, g);
      ret.__alvo = { y: y(base + v), h: Math.max(0, y(base) - y(base + v)) };
      return { regiao: r, valor: v, ret };
    });

    const alvo = cria("rect", {
      x: x(i) - x.largura * 0.2, y: M.topo - 6,
      width: x.largura * 1.4, height: altura - M.baixo - M.topo + 6,
      fill: "transparent", cursor: "crosshair",
    }, g);

    alvo.addEventListener("mousemove", (e) => {
      camada.style.opacity = 0.42;
      g.style.opacity = 1;
      const linhas = partes.filter((p) => p.valor > 0)
        .sort((a, b) => b.valor - a.valor)
        .map((p) => `<span style="color:${cores[p.regiao]}">■</span> ${
          p.regiao} <span class="destaque-dica">${p.valor}</span>`)
        .join("<br>");
      t.mostra(
        `<b>${ano}</b> · <span class="destaque-dica">${totais[i]}</span> municípios-ano` +
        (semVig.has(ano)
          ? '<br><span style="opacity:.75">sem registro nacional de epizootia</span>'
          : "") +
        (linhas ? `<br>${linhas}` : ""),
        e);
    });
    alvo.addEventListener("mouseleave", () => {
      camada.style.opacity = 1;
      g.style.opacity = 1;
      t.esconde();
    });
    return partes;
  });

  eixoX(s, linear([0, d.anos.length - 1],
                  [x(0) + x.largura / 2, x(d.anos.length - 1) + x.largura / 2]),
        altura - M.baixo, {
    valores: d.anos.map((a, i) => i).filter((i) => d.anos[i] % 5 === 0
                                                  || i === d.anos.length - 1),
    formato: (i) => d.anos[Math.round(i)],
  });

  /* Marca a onda com uma faixa de fundo, e não com chave acima do gráfico:
     acima disputaria espaço com a legenda, e a faixa ainda tem a vantagem de
     mostrar quais anos exatos ela cobre. */
  const i0 = d.anos.indexOf(2016), i1 = d.anos.indexOf(2019);
  if (i0 >= 0 && i1 >= 0) {
    const xa = x(i0) - x.largura * 0.22;
    const xb = x(i1) + x.largura * 1.22;
    const faixa = cria("g", {}, s);
    cria("rect", {
      x: xa, y: M.topo, width: xb - xa, height: altura - M.baixo - M.topo,
      fill: variavelCss("--imunidade"), opacity: 0.055,
    }, faixa);
    const rot = cria("text", {
      x: xa + 5, y: M.topo + 13, class: "rotulo-eixo",
    }, faixa);
    rot.textContent = "onda da mata atlântica";
    s.insertBefore(faixa, s.firstChild);
  }

  aoEntrar(hospedeiro, () => {
    colunas.forEach((partes, i) => {
      anima(620, (t) => {
        for (const p of partes) {
          p.ret.setAttribute("y", y(0) - (y(0) - p.ret.__alvo.y) * t);
          p.ret.setAttribute("height", p.ret.__alvo.h * t);
        }
      }, { atraso: i * 22 });
    });
  });

  /* Legenda encostada à direita, para não disputar espaço com o título do eixo,
     que fica à esquerda na mesma linha. */
  const leg = cria("g", {}, s);
  const larguraItem = 104;
  const inicio = largura - M.direita - d.regioes.length * larguraItem;
  d.regioes.forEach((r, k) => {
    const gx = inicio + k * larguraItem;
    cria("rect", { x: gx, y: 7, width: 9, height: 9, fill: cores[r] }, leg);
    const tx = cria("text", { x: gx + 14, y: 15.5, class: "rotulo-serie" }, leg);
    tx.textContent = r;
  });
}

/* ----------------------------------------------------------------------
   Figura 2 — o colapso por partição.
   Barras horizontais que se re-escalam ao trocar de métrica. A transição é o
   ponto: mostra que a mesma ordenação convive com magnitudes incomparáveis.
   ---------------------------------------------------------------------- */
function validacao(hospedeiro, controles, d) {
  const M = { topo: 30, direita: 120, baixo: 46, esquerda: 168 };
  const largura = 1000;
  const altura = M.topo + d.esquemas.length * 74 + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label", "Desempenho do modelo sob quatro partições");
  const t = dica();

  const METRICAS = {
    vezes: {
      valor: (e) => e.vezes_o_acaso,
      rotulo: (v) => `${fmt.n(v, 1)}×`,
      titulo: "PR-AUC em múltiplos do acaso",
      referencia: 1, nomeRef: "acaso",
    },
    roc: {
      valor: (e) => e.roc_auc,
      rotulo: (v) => fmt.n(v, 3),
      titulo: "ROC-AUC",
      referencia: 0.5, nomeRef: "sorteio",
    },
    custo: {
      valor: (e) => e.investigados_por_evento,
      rotulo: (v) => fmt.n(v, 1),
      titulo: "municípios investigados por surto encontrado",
      referencia: null,
    },
  };

  const y = banda(d.esquemas.length, [M.topo, altura - M.baixo], 0.42);
  let atual = "vezes";
  let x = null;

  const camadaEixo = cria("g", {}, s);
  const camadaRef = cria("g", {}, s);
  const camada = cria("g", {}, s);

  const barras = d.esquemas.map((e, i) => {
    const g = cria("g", { transform: `translate(0,${y(i)})` }, camada);
    const ret = cria("rect", {
      x: M.esquerda, y: 0, height: y.largura, width: 0,
      fill: i === d.esquemas.length - 1
        ? variavelCss("--alto") : variavelCss("--azul"),
      "fill-opacity": i === d.esquemas.length - 1 ? 1 : 0.55,
    }, g);
    const nome = cria("text", {
      x: M.esquerda - 14, y: y.largura / 2 - 3, "text-anchor": "end",
      class: "rotulo-serie",
    }, g);
    nome.textContent = e.nome;
    const sub = cria("text", {
      x: M.esquerda - 14, y: y.largura / 2 + 13, "text-anchor": "end",
      class: "rotulo-eixo",
    }, g);
    sub.textContent = i === 0 ? "atalho comum"
      : i === d.esquemas.length - 1 ? "uso pretendido" : "";
    const valor = cria("text", {
      x: M.esquerda + 8, y: y.largura / 2 + 4, class: "rotulo-valor",
    }, g);

    const alvo = cria("rect", {
      x: 0, y: -8, width: largura, height: y.largura + 16,
      fill: "transparent", cursor: "help",
    }, g);
    alvo.addEventListener("mousemove", (ev) => t.mostra(
      `<b>${e.nome}</b><br>${e.explicacao}<br>` +
      `ROC-AUC <span class="destaque-dica">${fmt.n(e.roc_auc, 3)}</span> · ` +
      `PR-AUC <span class="destaque-dica">${fmt.n(e.pr_auc, 4)}</span>`, ev));
    alvo.addEventListener("mouseleave", () => t.esconde());

    return { e, ret, valor, g };
  });

  function desenha(chave, duracao) {
    const m = METRICAS[chave];
    const valores = d.esquemas.map(m.valor);
    const teto = Math.max(...valores) * 1.12;
    const novoX = linear([0, teto], [M.esquerda, largura - M.direita]);

    camadaEixo.innerHTML = "";
    eixoX(camadaEixo, novoX, altura - M.baixo + 8, {
      formato: (v) => m.rotulo(v), titulo: m.titulo, tamanho: 5,
    });

    camadaRef.innerHTML = "";
    if (m.referencia !== null && m.referencia <= teto) {
      const xr = novoX(m.referencia);
      cria("line", {
        x1: xr, x2: xr, y1: M.topo - 12, y2: altura - M.baixo,
        stroke: "#bbb", "stroke-width": 1,
        "stroke-dasharray": "3 3",
      }, camadaRef);
      const r = cria("text", {
        x: xr + 5, y: M.topo - 16, class: "rotulo-eixo",
      }, camadaRef);
      r.textContent = m.nomeRef;
    }

    for (const b of barras) {
      const de = b.__v ?? 0;
      const para = m.valor(b.e);
      const deX = x ? x : novoX;
      anima(duracao, (tt) => {
        const v = de + (para - de) * tt;
        const larg = novoX(v) - M.esquerda;
        b.ret.setAttribute("width", Math.max(0, larg));
        b.valor.setAttribute("x", M.esquerda + Math.max(0, larg) + 9);
        b.valor.textContent = m.rotulo(v);
      });
      b.__v = para;
    }
    x = novoX;
  }

  aoEntrar(hospedeiro, () => desenha("vezes", 900));

  controles.addEventListener("click", (e) => {
    const b = e.target.closest("[data-metrica]");
    if (!b) return;
    for (const outro of controles.querySelectorAll("[data-metrica]")) {
      outro.setAttribute("aria-pressed", String(outro === b));
    }
    atual = b.dataset.metrica;
    desenha(atual, 760);
  });
}

/* ------------------------------------------------------------------ início */
(async function () {
  const [s, v] = await Promise.all([carrega("serie.json"), carrega("validacao.json")]);

  document.getElementById("total-eventos").textContent =
    fmt.inteiro(s.total_eventos);
  document.getElementById("total-linhas").textContent = fmt.inteiro(s.linhas);

  serie(document.getElementById("fig-serie"), s);
  validacao(document.getElementById("fig-validacao"),
            document.getElementById("ctrl-validacao"), v);
})();
