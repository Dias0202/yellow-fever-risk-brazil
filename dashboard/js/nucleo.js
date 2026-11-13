/* Núcleo de desenho do painel.

   Sem biblioteca de gráficos. A razão não é purismo: cada figura aqui responde a
   uma pergunta específica — a ordenação que se desfaz, o enxame que se separa, o
   salto no ano do surto — e nenhuma delas é um tipo de gráfico pronto. Escrever o
   SVG à mão custa mais linhas e dá a forma exata, além de manter o painel sem
   dependência externa e legível offline.

   As animações são de transição, não de entrada: elas existem quando há dois
   estados a comparar. `prefers-reduced-motion` desliga todas. */

export const SVG = "http://www.w3.org/2000/svg";

/* --------------------------------------------------------------- básicos */
export function cria(tag, atrs = {}, pai = null) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(atrs)) {
    if (v === null || v === undefined) continue;
    el.setAttribute(k, v);
  }
  if (pai) pai.appendChild(el);
  return el;
}

export function svg(hospedeiro, largura, altura) {
  hospedeiro.innerHTML = "";
  const s = cria("svg", {
    viewBox: `0 0 ${largura} ${altura}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
  }, hospedeiro);
  return s;
}

export const semMovimento = () =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------------------------------------------------------------- escalas */
export function linear(dominio, alcance) {
  const [d0, d1] = dominio, [r0, r1] = alcance;
  const span = d1 - d0 || 1;
  const f = (v) => r0 + ((v - d0) / span) * (r1 - r0);
  f.inversa = (p) => d0 + ((p - r0) / (r1 - r0)) * span;
  f.dominio = dominio;
  f.alcance = alcance;
  return f;
}

export function banda(n, alcance, folga = 0.25) {
  const [r0, r1] = alcance;
  const passo = (r1 - r0) / n;
  const largura = passo * (1 - folga);
  const f = (i) => r0 + i * passo + (passo - largura) / 2;
  f.largura = largura;
  f.passo = passo;
  return f;
}

/* Marcas de eixo em passos legíveis — 1, 2, 2.5 ou 5 vezes potência de dez. */
export function marcas(d0, d1, alvo = 5) {
  const bruto = (d1 - d0) / alvo;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(bruto) || 1)));
  const norm = bruto / mag;
  const passo = (norm >= 5 ? 5 : norm >= 2.5 ? 2.5 : norm >= 2 ? 2 : 1) * mag;
  const saida = [];
  for (let v = Math.ceil(d0 / passo) * passo; v <= d1 + 1e-9; v += passo) {
    saida.push(Math.abs(v) < 1e-12 ? 0 : v);
  }
  return saida;
}

/* ------------------------------------------------------------------ eixos */
export function eixoX(pai, escala, y, { formato = String, valores = null,
                                        titulo = null, tamanho = 5 } = {}) {
  const g = cria("g", { class: "eixo" }, pai);
  const vs = valores || marcas(escala.dominio[0], escala.dominio[1], tamanho);
  cria("line", {
    x1: escala.alcance[0], x2: escala.alcance[1], y1: y, y2: y,
  }, g);
  for (const v of vs) {
    const x = escala(v);
    cria("line", { x1: x, x2: x, y1: y, y2: y + 4 }, g);
    const t = cria("text", { x, y: y + 16, "text-anchor": "middle" }, g);
    t.textContent = formato(v);
  }
  if (titulo) {
    const t = cria("text", {
      x: escala.alcance[1], y: y + 32, "text-anchor": "end", class: "rotulo-eixo",
    }, g);
    t.textContent = titulo;
  }
  return g;
}

export function eixoY(pai, escala, x, { formato = String, valores = null,
                                        grade = null, titulo = null,
                                        tamanho = 5 } = {}) {
  const g = cria("g", { class: "eixo" }, pai);
  const vs = valores || marcas(escala.dominio[0], escala.dominio[1], tamanho);
  for (const v of vs) {
    const y = escala(v);
    if (grade) {
      const gg = cria("g", { class: "grade" }, pai);
      cria("line", { x1: x, x2: grade, y1: y, y2: y }, gg);
      pai.insertBefore(gg, pai.firstChild);
    }
    const t = cria("text", {
      x: x - 8, y: y + 3.5, "text-anchor": "end",
    }, g);
    t.textContent = formato(v);
  }
  if (titulo) {
    const t = cria("text", {
      x: x - 8, y: escala.alcance[1] - 14, "text-anchor": "end",
      class: "rotulo-eixo",
    }, g);
    t.textContent = titulo;
  }
  return g;
}

/* ------------------------------------------------------------------ cores */
/* Interpola no espaço sRGB. Suficiente para rampas curtas e sem dependência. */
function hexRgb(h) {
  const n = parseInt(h.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function rampa(paradas) {
  const cores = paradas.map(hexRgb);
  return (t) => {
    const u = Math.max(0, Math.min(1, t)) * (cores.length - 1);
    const i = Math.min(Math.floor(u), cores.length - 2);
    const f = u - i;
    const c = cores[i].map((v, k) => Math.round(v + (cores[i + 1][k] - v) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
}

export function variavelCss(nome) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(nome).trim();
}

export const CORES_GRUPO = () => ({
  CLIMA: variavelCss("--clima"),
  CLIMA_DEFASADO: variavelCss("--clima-def"),
  PAISAGEM: variavelCss("--paisagem"),
  IMUNIDADE: variavelCss("--imunidade"),
  DEMOGRAFIA: variavelCss("--demografia"),
});

export const NOME_GRUPO = {
  CLIMA: "Clima", CLIMA_DEFASADO: "Clima defasado", PAISAGEM: "Paisagem",
  IMUNIDADE: "Imunidade", DEMOGRAFIA: "Demografia",
};

/* --------------------------------------------------------------- rótulos */
const APELIDOS = {
  ppt_total: "chuva anual", ppt_sazonalidade: "sazonalidade da chuva",
  ppt_trimestre_seco: "chuva no trimestre seco",
  tmax_medio: "temperatura máxima", tmin_medio: "temperatura mínima",
  tmax_mes_mais_quente: "máxima do mês mais quente",
  amplitude_termica: "amplitude térmica", vpd_medio: "déficit de pressão de vapor",
  ws_medio: "velocidade do vento",
  floresta_frac: "floresta", silvicultura_frac: "silvicultura",
  pastagem_frac: "pastagem", urbano_frac: "área urbana",
  areas_umidas_frac: "áreas úmidas", savana_frac: "savana",
  mosaico_agropecuario_frac: "mosaico agropecuário",
  interface_floresta_antropico: "interface floresta–antrópico",
  cobertura_florestal: "cobertura florestal",
  perda_ano_km2: "perda florestal no ano",
  perda_acumulada_km2: "perda florestal acumulada",
  imunidade_defasada2: "imunidade (defasada 2 anos)",
  populacao_log: "população (log)", densidade_hab_km2: "densidade",
  area_km2: "área do município",
};

export function apelido(nome) {
  const base = nome.replace(/_lag(\d)$/, "");
  const sufixo = nome.match(/_lag(\d)$/);
  const raiz = APELIDOS[base] || base.replace(/_/g, " ");
  return sufixo ? `${raiz} · t−${sufixo[1]}` : raiz;
}

/* ---------------------------------------------------------------- formato */
export const fmt = {
  n: (v, c = 2) => v.toLocaleString("pt-BR", {
    minimumFractionDigits: c, maximumFractionDigits: c }),
  pct: (v, c = 0) => (v * 100).toLocaleString("pt-BR", {
    minimumFractionDigits: c, maximumFractionDigits: c }) + "%",
  sinal: (v, c = 4) => (v >= 0 ? "+" : "−") + Math.abs(v).toLocaleString("pt-BR", {
    minimumFractionDigits: c, maximumFractionDigits: c }),
  inteiro: (v) => Math.round(v).toLocaleString("pt-BR"),
};

/* ------------------------------------------------------------------ dica */
let dicaEl = null;

export function dica() {
  if (!dicaEl) {
    dicaEl = document.createElement("div");
    dicaEl.className = "dica";
    dicaEl.setAttribute("role", "status");
    document.body.appendChild(dicaEl);
  }
  return {
    mostra(html, evento) {
      dicaEl.innerHTML = html;
      dicaEl.style.left = `${evento.clientX}px`;
      dicaEl.style.top = `${evento.clientY}px`;
      dicaEl.classList.add("ativa");
    },
    esconde() { dicaEl.classList.remove("ativa"); },
  };
}

/* ---------------------------------------------- disparo por rolagem */
const observador = new IntersectionObserver((entradas) => {
  for (const e of entradas) {
    if (!e.isIntersecting) continue;
    e.target.classList.add("visivel");
    if (e.target.__reserva) {
      clearTimeout(e.target.__reserva);
      e.target.__reserva = null;
    }
    const fila = e.target.__aoEntrar;
    if (fila) {
      e.target.__aoEntrar = null;
      for (const fn of fila) fn();
    }
    observador.unobserve(e.target);
  }
}, { threshold: 0.02, rootMargin: "0px 0px -40px 0px" });

/** Executa `fn` na primeira vez que o elemento aparece na tela.
 *
 * Acumula em FILA, e nao substitui. Varias figuras registram uma chamada por
 * barra ou por serie, dentro de um laco sobre o mesmo elemento hospedeiro; com
 * substituicao, so a ultima chamada rodava e a figura aparecia com um unico
 * elemento desenhado. */
export function aoEntrar(el, fn) {
  el.setAttribute("data-animar", "");
  if (semMovimento()) { el.classList.add("visivel"); fn(); return; }
  (el.__aoEntrar ||= []).push(fn);
  observador.observe(el);

  // Rede de seguranca. `IntersectionObserver` nao dispara enquanto o documento
  // esta oculto, e uma figura abaixo da dobra ficaria em branco ao imprimir a
  // pagina, ao abri-la em aba de segundo plano ou ao indexa-la. O conteudo nao
  // pode depender de o leitor ter rolado ate ele.
  if (!el.__reserva) {
    el.__reserva = setTimeout(() => desperta(el), 4000);
  }
}

function desperta(el) {
  const fila = el.__aoEntrar;
  if (!fila) return;
  el.__aoEntrar = null;
  el.classList.add("visivel");
  observador.unobserve(el);
  for (const fn of fila) fn();
}

/* ------------------------------------------------------------- animação */
const suave = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const saida = (t) => 1 - Math.pow(1 - t, 3);

/** Interpola de 0 a 1 chamando `passo(t)`. Devolve promessa.
 *
 * Tem rede de seguranca. `requestAnimationFrame` e estrangulado ou suspenso
 * quando a aba nao esta em primeiro plano, e sem a rede a figura fica congelada
 * num estado parcial - barras pela metade, curvas incompletas. Um relogio
 * independente garante o estado final mesmo que nenhum quadro chegue a rodar.
 *
 * A animacao e enriquecimento; o estado final e o conteudo. */
export function anima(duracao, passo, { atraso = 0, curva = saida } = {}) {
  if (semMovimento() || document.hidden) { passo(1); return Promise.resolve(); }
  return new Promise((resolve) => {
    let terminou = false;
    const finaliza = () => {
      if (terminou) return;
      terminou = true;
      passo(1);
      resolve();
    };
    const rede = setTimeout(finaliza, atraso + duracao + 250);
    const inicio = performance.now() + atraso;
    function quadro(agora) {
      if (terminou) return;
      const t = (agora - inicio) / duracao;
      if (t < 0) { requestAnimationFrame(quadro); return; }
      if (t >= 1) { clearTimeout(rede); finaliza(); return; }
      passo(curva(t));
      requestAnimationFrame(quadro);
    }
    requestAnimationFrame(quadro);
  });
}

export { suave, saida };

/* Desenha uma linha progressivamente, do início ao fim. */
export function tracaCaminho(caminho, duracao = 1100, atraso = 0) {
  let comprimento = 0;
  try { comprimento = caminho.getTotalLength(); } catch { comprimento = 0; }
  if (!comprimento) return Promise.resolve();
  caminho.style.strokeDasharray = comprimento;
  caminho.style.strokeDashoffset = comprimento;
  return anima(duracao, (t) => {
    caminho.style.strokeDashoffset = comprimento * (1 - t);
  }, { atraso }).then(() => { caminho.style.strokeDasharray = "none"; });
}

/* Conta um número até o valor final, para os destaques de texto. */
export function contaAte(el, valor, { casas = 0, prefixo = "", sufixo = "",
                                      duracao = 900 } = {}) {
  return anima(duracao, (t) => {
    el.textContent = prefixo + fmt.n(valor * t, casas) + sufixo;
  });
}

/* ------------------------------------------------------------- caminhos */
export function linhaSuave(pontos) {
  if (pontos.length < 2) return "";
  let d = `M${pontos[0][0]},${pontos[0][1]}`;
  for (let i = 1; i < pontos.length; i++) {
    const [x0, y0] = pontos[i - 1];
    const [x1, y1] = pontos[i];
    const mx = (x0 + x1) / 2;
    d += `C${mx},${y0} ${mx},${y1} ${x1},${y1}`;
  }
  return d;
}

export function linhaReta(pontos) {
  return pontos.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join("");
}

/* ------------------------------------------------------------ dados */
export async function carrega(nome) {
  const r = await fetch(`dados/${nome}`);
  if (!r.ok) throw new Error(`não foi possível ler dados/${nome}`);
  return r.json();
}
