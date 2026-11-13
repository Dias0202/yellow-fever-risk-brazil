/* Página de cenários: matriz GCM × SSP, contrastes pareados e recorte regional. */

import {
  carrega, cria, svg, linear, banda, eixoX, eixoY, aoEntrar, anima, dica,
  rampa, fmt, variavelCss,
} from "./nucleo.js";

/* ----------------------------------------------------------------------
   Figura 1 — matriz de calor GCM × SSP.
   Matriz e não barras agrupadas: a leitura que interessa é justamente comparar
   a variação ao longo de uma dimensão contra a variação ao longo da outra, e a
   matriz coloca as duas comparações lado a lado sem privilegiar nenhuma.
   ---------------------------------------------------------------------- */
function matriz(hospedeiro, controles, d) {
  const M = { topo: 58, direita: 150, baixo: 40, esquerda: 168 };
  const celula = { largura: 132, altura: 56 };
  const largura = M.esquerda + d.ssps.length * celula.largura + M.direita;
  const altura = M.topo + d.gcms.length * celula.altura + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  s.setAttribute("aria-label", "Projeção por modelo climático e trajetória de emissão");
  const t = dica();

  const risco = new Map(d.projecao.map((p) => [`${p.gcm}|${p.ssp}`, p]));
  const aquec = new Map(d.aquecimento.map((p) => [`${p.gcm}|${p.ssp}`, p]));

  /* Ordena os GCMs pelo aquecimento médio: a matriz fica legível de cima para
     baixo e a dominância da linha do UKESM aparece sozinha. */
  const gcms = [...d.gcms].sort((a, b) => {
    const m = (g) => d.aquecimento.filter((p) => p.gcm === g)
      .reduce((acc, p) => acc + p.delta_tmax_medio, 0);
    return m(a) - m(b);
  });

  const CAMPOS = {
    risco: {
      valor: (k) => risco.get(k)?.aumento_relativo ?? 0,
      rotulo: (v) => `+${fmt.n(v * 100, 1)}%`,
      legenda: "aumento do risco médio previsto sobre a linha de base",
    },
    aquecimento: {
      valor: (k) => aquec.get(k)?.delta_tmax_medio ?? 0,
      rotulo: (v) => `+${fmt.n(v, 2)} °C`,
      legenda: "aumento da temperatura máxima média projetada",
    },
  };

  /* Sequencial e monotônica em luminosidade. Uma rampa de azul a vermelho
     passando por roxo daria a células intermediárias uma cor mais escura que a de
     células maiores — a ordem visual não acompanharia a ordem dos valores. */
  const cor = rampa(["#f6f8f9", "#f2c9a4", variavelCss("--alto")]);

  for (const [j, ssp] of d.ssps.entries()) {
    const tx = cria("text", {
      x: M.esquerda + j * celula.largura + celula.largura / 2, y: M.topo - 14,
      "text-anchor": "middle", class: "rotulo-serie", "font-weight": 600,
    }, s);
    tx.textContent = ssp.replace("ssp", "SSP").replace(/(\d)(\d)(\d)/, "$1-$2.$3");
  }
  const tituloSsp = cria("text", {
    x: M.esquerda, y: M.topo - 34, class: "rotulo-eixo",
  }, s);
  tituloSsp.textContent = "trajetória de emissão";

  const celulas = [];
  gcms.forEach((gcm, i) => {
    const ty = cria("text", {
      x: M.esquerda - 14, y: M.topo + i * celula.altura + celula.altura / 2 + 4,
      "text-anchor": "end", class: "rotulo-serie",
    }, s);
    ty.textContent = gcm;

    d.ssps.forEach((ssp, j) => {
      const k = `${gcm}|${ssp}`;
      const g = cria("g", {}, s);
      const ret = cria("rect", {
        x: M.esquerda + j * celula.largura + 2,
        y: M.topo + i * celula.altura + 2,
        width: celula.largura - 4, height: celula.altura - 4,
        fill: "#f2f4f6", cursor: "help",
      }, g);
      const val = cria("text", {
        x: M.esquerda + j * celula.largura + celula.largura / 2,
        y: M.topo + i * celula.altura + celula.altura / 2 + 5,
        "text-anchor": "middle", class: "rotulo-valor", "font-size": 13.5,
      }, g);
      celulas.push({ k, gcm, ssp, ret, val });
    });
  });

  /* Escala de cor, à direita. */
  const legenda = cria("g", {}, s);

  function desenha(campo, duracao) {
    const c = CAMPOS[campo];
    const valores = celulas.map((x) => c.valor(x.k));
    const lo = Math.min(...valores), hi = Math.max(...valores);
    const norm = (v) => (hi === lo ? 0.5 : (v - lo) / (hi - lo));

    for (const cel of celulas) {
      const para = c.valor(cel.k);
      const de = cel.__v ?? para;
      cel.ret.setAttribute("fill", cor(norm(para)));
      cel.val.setAttribute("fill", norm(para) > 0.66 ? "#fff" : "#2c3e50");
      anima(duracao, (tt) => {
        cel.val.textContent = c.rotulo(de + (para - de) * tt);
      });
      cel.__v = para;

      cel.ret.onmousemove = (e) => {
        const r = risco.get(cel.k), a = aquec.get(cel.k);
        t.mostra(
          `<b>${cel.gcm}</b> · ${cel.ssp}<br>` +
          `risco previsto <span class="destaque-dica">${
            fmt.n(r.risco_medio * 100, 3)}%</span><br>` +
          `aumento sobre a base <span class="destaque-dica">+${
            fmt.n(r.aumento_relativo * 100, 1)}%</span><br>` +
          `aquecimento <span class="destaque-dica">+${
            fmt.n(a.delta_tmax_medio, 2)} °C</span><br>` +
          `chuva <span class="destaque-dica">×${
            fmt.n(a.razao_ppt_mediana, 3)}</span>`, e);
      };
      cel.ret.onmouseleave = () => t.esconde();
    }

    legenda.innerHTML = "";
    const lx = M.esquerda + d.ssps.length * celula.largura + 34;
    const lh = Math.min(180, d.gcms.length * celula.altura - 20);
    const grad = cria("linearGradient", {
      id: `rampa-${campo}`, x1: 0, y1: 1, x2: 0, y2: 0,
    }, cria("defs", {}, legenda));
    for (let i = 0; i <= 10; i++) {
      cria("stop", { offset: `${i * 10}%`, "stop-color": cor(i / 10) }, grad);
    }
    cria("rect", {
      x: lx, y: M.topo + 6, width: 13, height: lh,
      fill: `url(#rampa-${campo})`, stroke: "#ddd",
    }, legenda);
    for (const [v, dy] of [[hi, 10], [lo, lh + 4]]) {
      const tx = cria("text", {
        x: lx + 19, y: M.topo + dy, class: "rotulo-eixo",
      }, legenda);
      tx.textContent = c.rotulo(v);
    }
    const cap = cria("text", {
      x: M.esquerda, y: altura - 12, class: "rotulo-eixo",
    }, legenda);
    cap.textContent = c.legenda;
  }

  aoEntrar(hospedeiro, () => desenha("risco", 700));

  controles.addEventListener("click", (e) => {
    const b = e.target.closest("[data-campo]");
    if (!b) return;
    for (const o of controles.querySelectorAll("[data-campo]")) {
      o.setAttribute("aria-pressed", String(o === b));
    }
    desenha(b.dataset.campo, 600);
  });
}

/* ----------------------------------------------------------------------
   Figura 2 — contrastes pareados com intervalo.
   ---------------------------------------------------------------------- */
function contrastes(hospedeiro, d) {
  const itens = [
    { chave: "futuro_menos_base", rotulo: "Clima projetado − linha de base",
      sub: "efeito do aquecimento sobre o risco previsto" },
    { chave: "ssp585_menos_ssp126", rotulo: "SSP5-8.5 − SSP1-2.6",
      sub: "diferença entre a trajetória mais severa e a mais branda" },
  ].map((i) => ({ ...i, ...d.contrastes[i.chave] }));

  const M = { topo: 44, direita: 150, baixo: 56, esquerda: 250 };
  const largura = 880, altura = M.topo + itens.length * 78 + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  const t = dica();

  const lo = Math.min(0, ...itens.map((i) => i.ic95[0])) * 1.25;
  const hi = Math.max(...itens.map((i) => i.ic95[1])) * 1.25;
  const x = linear([lo, hi], [M.esquerda, largura - M.direita]);
  const y = banda(itens.length, [M.topo, altura - M.baixo], 0.55);

  eixoX(s, x, altura - M.baixo + 8, {
    formato: (v) => fmt.sinal(v, 4),
    titulo: "variação da probabilidade média prevista",
  });
  cria("line", {
    x1: x(0), x2: x(0), y1: M.topo - 12, y2: altura - M.baixo,
    stroke: "#999", "stroke-width": 1,
  }, s);
  const zeroT = cria("text", {
    x: x(0), y: M.topo - 18, "text-anchor": "middle", class: "rotulo-eixo",
  }, s);
  zeroT.textContent = "sem efeito";

  const azul = variavelCss("--azul");
  itens.forEach((it, i) => {
    const separa = it.ic95[0] > 0 || it.ic95[1] < 0;
    const cy = y(i) + y.largura / 2;
    const g = cria("g", {}, s);

    const rot = cria("text", {
      x: M.esquerda - 16, y: cy - 2, "text-anchor": "end",
      class: "rotulo-serie", "font-weight": separa ? 600 : 400,
      fill: separa ? "#222" : "#777",
    }, g);
    rot.textContent = it.rotulo;
    const sub = cria("text", {
      x: M.esquerda - 16, y: cy + 15, "text-anchor": "end", class: "rotulo-eixo",
    }, g);
    sub.textContent = it.sub;

    const barra = cria("line", {
      x1: x(0), x2: x(0), y1: cy, y2: cy,
      stroke: separa ? azul : "#b8c0c6", "stroke-width": 3,
      "stroke-linecap": "round",
    }, g);
    const ponto = cria("circle", {
      cx: x(0), cy, r: 6, fill: separa ? azul : "#8d979e",
      stroke: "#fff", "stroke-width": 1.6,
    }, g);
    const val = cria("text", { x: 0, y: cy + 4, class: "rotulo-valor" }, g);
    const nota = cria("text", { x: 0, y: cy + 19, class: "rotulo-eixo" }, g);

    const alvo = cria("rect", {
      x: 0, y: cy - 26, width: largura, height: 52,
      fill: "transparent", cursor: "help",
    }, g);
    alvo.addEventListener("mousemove", (e) => t.mostra(
      `<b>${it.rotulo}</b><br>` +
      `média <span class="destaque-dica">${fmt.sinal(it.media, 5)}</span><br>` +
      `IC 95% [${fmt.sinal(it.ic95[0], 5)}, ${fmt.sinal(it.ic95[1], 5)}]<br>` +
      `<span class="destaque-dica">${it.reajustes_com_sinal_positivo}</span>` +
      ` de ${it.n_reajustes} reajustes com sinal positivo<br>` +
      `<span style="opacity:.75">${separa
        ? "intervalo não inclui zero" : "intervalo inclui zero"}</span>`, e));
    alvo.addEventListener("mouseleave", () => t.esconde());

    aoEntrar(hospedeiro, () => {
      anima(850, (tt) => {
        barra.setAttribute("x1", x(it.ic95[0] * tt));
        barra.setAttribute("x2", x(it.ic95[1] * tt));
        ponto.setAttribute("cx", x(it.media * tt));
        val.setAttribute("x", x(it.ic95[1] * tt) + 14);
        val.textContent = fmt.sinal(it.media * tt, 5);
      }, { atraso: i * 180 });
      anima(400, () => {
        nota.setAttribute("x", x(it.ic95[1]) + 14);
        nota.textContent = `${it.reajustes_com_sinal_positivo}/${
          it.n_reajustes} reajustes positivos`;
      }, { atraso: 900 });
    });
  });
}

/* ----------------------------------------------------------------------
   Figura 3 — variação por macrorregião.
   ---------------------------------------------------------------------- */
function porRegiao(hospedeiro, d) {
  const dados = [...d.por_regiao].sort((a, b) => a.delta_medio - b.delta_medio);
  const M = { topo: 34, direita: 190, baixo: 56, esquerda: 130 };
  const largura = 860, altura = M.topo + dados.length * 50 + M.baixo;
  const s = svg(hospedeiro, largura, altura);
  const t = dica();

  const lo = Math.min(0, ...dados.map((r) => r.delta_medio)) * 1.3;
  const hi = Math.max(...dados.map((r) => r.delta_medio)) * 1.25;
  const x = linear([lo, hi], [M.esquerda, largura - M.direita]);
  const y = banda(dados.length, [M.topo, altura - M.baixo], 0.42);

  eixoX(s, x, altura - M.baixo + 8, {
    formato: (v) => fmt.sinal(v, 4),
    titulo: "variação do risco médio previsto",
  });
  cria("line", {
    x1: x(0), x2: x(0), y1: M.topo - 8, y2: altura - M.baixo,
    stroke: "#999", "stroke-width": 1,
  }, s);

  const azul = variavelCss("--azul");
  const paisagem = variavelCss("--paisagem");

  dados.forEach((r, i) => {
    const desce = r.delta_medio < 0;
    const g = cria("g", { transform: `translate(0,${y(i)})` }, s);
    const rot = cria("text", {
      x: M.esquerda - 14, y: y.largura / 2 + 4, "text-anchor": "end",
      class: "rotulo-serie",
    }, g);
    rot.textContent = r.regiao;
    const barra = cria("rect", {
      x: x(0), y: 2, width: 0, height: y.largura - 4,
      fill: desce ? paisagem : azul, "fill-opacity": 0.8,
    }, g);
    const val = cria("text", { x: 0, y: y.largura / 2 + 4, class: "rotulo-valor" }, g);
    const sr = cria("text", { x: 0, y: y.largura / 2 + 4, class: "rotulo-eixo" }, g);

    const alvo = cria("rect", {
      x: 0, y: -3, width: largura, height: y.largura + 6,
      fill: "transparent", cursor: "help",
    }, g);
    alvo.addEventListener("mousemove", (e) => t.mostra(
      `<b>${r.regiao}</b><br>` +
      `variação <span class="destaque-dica">${fmt.sinal(r.delta_medio, 5)}</span><br>` +
      `dispersão entre cenários ${fmt.n(r.dp_entre_cenarios_media, 5)}<br>` +
      `dispersão entre ajustes ${fmt.n(r.dp_entre_ajustes_media, 5)}<br>` +
      `razão sinal/ruído <span class="destaque-dica">${
        fmt.n(r.razao_sinal_ruido_mediana, 2)}</span>`, e));
    alvo.addEventListener("mouseleave", () => t.esconde());

    aoEntrar(hospedeiro, () => {
      anima(800, (tt) => {
        const v = r.delta_medio * tt;
        barra.setAttribute("x", Math.min(x(0), x(v)));
        barra.setAttribute("width", Math.abs(x(v) - x(0)));
        const fim = Math.max(x(0), x(v));
        val.setAttribute("x", fim + 12);
        val.textContent = fmt.sinal(v, 5);
        sr.setAttribute("x", fim + 96);
      }, { atraso: i * 110 });
      anima(300, () => {
        sr.textContent = `sinal/ruído ${fmt.n(r.razao_sinal_ruido_mediana, 2)}`;
      }, { atraso: 900 });
    });
  });
}

/* ------------------------------------------------------------------ início */
(async function () {
  const d = await carrega("clima.json");
  const def = (id, v) => { document.getElementById(id).textContent = v; };

  def("m-gcms", String(d.gcms.length));
  def("m-disp-gcm", `${fmt.n(d.dispersao_entre_gcms, 2)} °C`);
  def("m-disp-ssp", `${fmt.n(d.dispersao_entre_ssps, 2)} °C`);
  def("cap-matriz",
    `A dispersão do aquecimento entre modelos é ${
      fmt.n(d.dispersao_entre_gcms / d.dispersao_entre_ssps, 1)} vezes a dispersão `
    + `entre trajetórias de emissão.`);

  matriz(document.getElementById("fig-matriz"),
         document.getElementById("ctrl-matriz"), d);
  contrastes(document.getElementById("fig-contrastes"), d);
  porRegiao(document.getElementById("fig-regiao-clima"), d);
})();
