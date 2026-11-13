"""Variaveis de propagacao: onde a onda esteve, e para onde ela aponta.

Este modulo traduz em variaveis a estrutura epidemiologica da febre amarela silvestre no Brasil:

1. **A doenca chega de fora.** O virus circula em ciclo enzootico entre mosquitos
   Haemagogus e Sabethes e primatas nao humanos, e avanca como frente de onda por
   floresta contigua. Um municipio e invadido porque a onda chegou perto dele, nao
   porque seu clima mudou. Dai a familia de FORCA DE INFECCAO.

2. **A area queimada fica refrataria.** Depois que a onda passa, a populacao local
   de bugios e dizimada e a sobrevivente fica imune. O sistema so volta a ser
   invadivel quando a demografia dos hospedeiros se recupera, o que leva anos.
   Dai a familia de SUSCETIBILIDADE.

3. **A epizootia antecede o caso humano.** O ser humano e hospedeiro acidental do
   ciclo silvestre; quando ha caso humano, ja houve circulacao em primatas. A
   vigilancia brasileira e construida sobre esse fato. Dai as variaveis de
   epizootia entrarem separadas das de caso humano.

Regra de causalidade temporal
-----------------------------
Toda variavel deste modulo para a temporada t usa exclusivamente desfechos das
temporadas <= t-1. Nenhuma linha enxerga a propria temporada. E o que torna
legitimo usar contagio como preditor: na operacao real, ao fim de junho sabe-se o
que aconteceu na temporada que acabou, e a pergunta e sobre a que vem.

Isso e diferente do grupo HISTORICO do painel anual, que fica fora dos modelos de
projecao climatica por nao existir num horizonte de vinte anos. A objecao vale
para projecao climatica; nao vale para previsao da proxima temporada, que e o
objetivo aqui.

Sobre o nucleo de decaimento
----------------------------
A contribuicao de um municipio j com evento para o risco de i decai com a
distancia segundo exp(-d/kappa). Nao se sabe de antemao qual kappa descreve a
propagacao, e ele nao e o mesmo para introducao por dispersao de mosquito, por
deslocamento de primata ou por transporte humano. Por isso quatro escalas entram
juntas, de 25 a 200 km, e o modelo escolhe.

Distancia efetiva por floresta entra em paralelo a euclidiana, pela mesma razao:
sao duas hipoteses sobre o meio de propagacao, e a comparacao entre as duas
familias e informativa por si.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from yfvrisk import config as cfg
from yfvrisk.features.espacial import carregar as carregar_espacial

# Escalas de decaimento, em quilometros. Ver a nota do cabecalho.
KAPPAS = [25, 50, 100, 200]

# Raios para contagem simples de vizinhos com evento. Contagem e nucleo medem
# coisas diferentes: o nucleo pondera, a contagem responde "quantos focos ha
# perto", que e o que um gestor pergunta.
RAIOS = [50, 100, 200]

# Teto para "temporadas desde o ultimo evento local". Acima disso a distincao
# deixa de ter sentido epidemiologico - trinta anos sem evento nao e mais
# informativo que vinte -, e o teto evita que a variavel vire um relogio.
TETO_REFRATARIO = 15

# Janela de pressao acumulada, em temporadas, e o fator de desconto por temporada
# de distancia. 0,5 significa que a temporada retrasada pesa metade da anterior.
JANELA_PRESSAO = 3
DESCONTO_PRESSAO = 0.5

SAIDA = cfg.PROCESSADO / "epidemia_temporada.parquet"
DIAGNOSTICO = cfg.METRICAS / "f6_epidemia_diagnostico.json"


def _matriz_nucleo(origem, destino, distancia, kappa, n) -> sparse.csr_matrix:
    """Matriz simetrica esparsa de pesos exp(-d/kappa).

    Pesos abaixo de 1e-4 sao descartados: para kappa de 25 km isso corta em
    230 km, e manter arestas de peso 0,00005 apenas multiplicaria o custo.
    """
    w = np.exp(-distancia / kappa).astype(np.float32)
    manter = w > 1e-4
    i, j, w = origem[manter], destino[manter], w[manter]
    return sparse.csr_matrix(
        (np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n))


def derivar(codigos, xy, arestas, evento, epizootia, temporadas,
            chave="cod_mun") -> tuple[pd.DataFrame, dict]:
    """Deriva as variaveis de propagacao para QUALQUER unidade espacial.

    A funcao nao sabe se `codigos` sao municipios, microrregioes ou regioes
    imediatas: ela recebe centroides, arestas com distancia, e as matrizes de
    desfecho (unidade x temporada). E o que permite rodar a mesma epidemiologia
    em varias escalas sem duplicar a logica - e sem o risco de as duas versoes
    divergirem em silencio.

    `arestas` e um dicionario com `origem`, `destino`, `distancia` e
    `distancia_efetiva`.
    """
    n = len(codigos)
    saida = {}

    # --------------------------------------------------------------- nucleos
    for rotulo, dist in [("", arestas["distancia"]),
                         ("_flor", arestas["distancia_efetiva"])]:
        for kappa in KAPPAS:
            K = _matriz_nucleo(arestas["origem"], arestas["destino"], dist,
                               kappa, n)
            # Defasagem de uma temporada: a coluna t recebe o produto da coluna
            # t-1. A primeira temporada fica sem, e sai como ausente.
            for nome, M in [("forca", evento), ("forca_epiz", epizootia)]:
                if rotulo and nome == "forca_epiz":
                    continue  # o par floresta x epizootia nao acrescenta leitura
                bruto = K @ M                                  # (n, T)
                desl = np.full_like(bruto, np.nan)
                desl[:, 1:] = bruto[:, :-1]
                saida[f"{nome}{rotulo}_k{kappa}"] = desl
            del K
            print(f"[epidemia] nucleo{rotulo} kappa={kappa} km")

    # ------------------------------------------------ distancia e contagem
    dist_prox = np.full((n, len(temporadas)), np.nan, dtype=np.float32)
    dist_prox3 = np.full((n, len(temporadas)), np.nan, dtype=np.float32)
    contagens = {r: np.full((n, len(temporadas)), np.nan, dtype=np.float32)
                 for r in RAIOS}
    dist_epiz = np.full((n, len(temporadas)), np.nan, dtype=np.float32)

    # Sentinela para "nenhum foco no alcance". Nao usar ausente: a informacao
    # "nao ha foco em 600 km" e forte e nao deve virar mediana do imputador.
    SEM_FOCO = 1500.0

    for t in range(1, len(temporadas)):
        for alvo, M, saida_dist in [("evento", evento, dist_prox),
                                    ("epizootia", epizootia, dist_epiz)]:
            focos = np.where(M[:, t - 1] > 0)[0]
            if len(focos) == 0:
                saida_dist[:, t] = SEM_FOCO
                continue
            d, _ = cKDTree(xy[focos]).query(xy, k=1)
            saida_dist[:, t] = np.minimum(d, SEM_FOCO)

        # Tres temporadas anteriores juntas.
        ini = max(0, t - JANELA_PRESSAO)
        focos3 = np.where(evento[:, ini:t].sum(axis=1) > 0)[0]
        if len(focos3):
            d3, _ = cKDTree(xy[focos3]).query(xy, k=1)
            dist_prox3[:, t] = np.minimum(d3, SEM_FOCO)
        else:
            dist_prox3[:, t] = SEM_FOCO

        for r in RAIOS:
            focos = np.where(evento[:, t - 1] > 0)[0]
            if len(focos) == 0:
                contagens[r][:, t] = 0
                continue
            viz = cKDTree(xy[focos]).query_ball_point(xy, r=r)
            contagens[r][:, t] = [len(v) for v in viz]

    saida["dist_foco_mais_proximo"] = dist_prox
    saida["dist_foco_3temporadas"] = dist_prox3
    saida["dist_epizootia_mais_proxima"] = dist_epiz
    for r in RAIOS:
        saida[f"focos_em_{r}km"] = contagens[r]
    print("[epidemia] distancias e contagens")

    # ------------------------------------------------------- suscetibilidade
    desde = np.full((n, len(temporadas)), float(TETO_REFRATARIO), dtype=np.float32)
    ultimo = np.full(n, -np.inf)
    for t in range(len(temporadas)):
        desde[:, t] = np.minimum(t - ultimo, TETO_REFRATARIO)
        ultimo = np.where(evento[:, t] > 0, t, ultimo)
    saida["temporadas_desde_evento"] = desde

    # Historico local acumulado, sempre defasado.
    for janela in (3, 5, 10):
        acc = np.full((n, len(temporadas)), np.nan, dtype=np.float32)
        for t in range(1, len(temporadas)):
            acc[:, t] = evento[:, max(0, t - janela):t].sum(axis=1)
        saida[f"eventos_locais_{janela}t"] = acc

    ev_ant = np.full((n, len(temporadas)), np.nan, dtype=np.float32)
    ev_ant[:, 1:] = evento[:, :-1]
    saida["evento_temporada_anterior"] = ev_ant
    ep_ant = np.full((n, len(temporadas)), np.nan, dtype=np.float32)
    ep_ant[:, 1:] = epizootia[:, :-1]
    saida["epizootia_temporada_anterior"] = ep_ant

    # Pressao acumulada: soma descontada da forca de infeccao das ultimas
    # temporadas. Captura area sob pressao sustentada, que e diferente de area
    # que recebeu um foco isolado.
    base = saida["forca_k100"]
    pressao = np.full_like(base, np.nan)
    for t in range(1, len(temporadas)):
        acc = np.zeros(n)
        peso = 0.0
        for k in range(JANELA_PRESSAO):
            if t - k < 1:
                break
            f = DESCONTO_PRESSAO ** k
            acc += f * np.nan_to_num(base[:, t - k])
            peso += f
        pressao[:, t] = acc / max(peso, 1e-9)
    saida["pressao_acumulada"] = pressao
    print("[epidemia] suscetibilidade e pressao")

    # ------------------------------------------------------------ frente
    # Deslocamento do centroide dos focos entre as duas temporadas anteriores,
    # e projecao de cada municipio sobre essa direcao. Positivo significa "a
    # frente da onda"; negativo, "atras dela, em area ja percorrida".
    proj = np.full((n, len(temporadas)), np.nan, dtype=np.float32)
    avanco = np.full(len(temporadas), np.nan)
    for t in range(2, len(temporadas)):
        a = np.where(evento[:, t - 2] > 0)[0]
        b = np.where(evento[:, t - 1] > 0)[0]
        if len(a) < 3 or len(b) < 3:
            continue
        ca, cb = xy[a].mean(axis=0), xy[b].mean(axis=0)
        v = cb - ca
        norma = np.linalg.norm(v)
        avanco[t] = norma
        if norma < 1e-6:
            continue
        u = v / norma
        proj[:, t] = (xy - cb) @ u
    saida["projecao_na_frente"] = proj
    saida["avanco_da_frente"] = np.tile(avanco, (n, 1)).astype(np.float32)
    print("[epidemia] frente de onda")

    # ------------------------------------------------------------- montagem
    quadro = pd.DataFrame({
        chave: np.repeat(codigos, len(temporadas)),
        "temporada": np.tile(temporadas, n),
    })
    for nome, M in saida.items():
        quadro[nome] = M.reshape(-1)

    diag = {
        "linhas": int(len(quadro)),
        "variaveis": [c for c in quadro.columns
                      if c not in ("cod_mun", "temporada")],
        "kappas_km": KAPPAS, "raios_km": RAIOS,
        "teto_refratario": TETO_REFRATARIO,
        "janela_pressao": JANELA_PRESSAO,
        "desconto_pressao": DESCONTO_PRESSAO,
        "sem_foco_km": SEM_FOCO,
        "temporadas": [int(temporadas[0]), int(temporadas[-1])],
        "avanco_da_frente_km_por_temporada": {
            int(temporadas[t]): (None if not np.isfinite(avanco[t])
                                 else round(float(avanco[t]), 1))
            for t in range(len(temporadas))},
    }
    return quadro, diag


def construir() -> tuple[pd.DataFrame, dict]:
    """A versao municipal, que e a usada pelo painel principal."""
    cfg.garantir_diretorios()
    z = carregar_espacial()
    codigos = [str(c) for c in z["codigos"]]

    epi = pd.read_parquet(cfg.PROCESSADO / "epi_temporada.parquet")
    temporadas = sorted(epi["temporada"].unique())

    def _desfecho(coluna):
        return (epi.pivot(index="cod_mun", columns="temporada", values=coluna)
                .reindex(codigos)[temporadas].fillna(0).to_numpy(dtype=np.float32))

    quadro, diag = derivar(
        codigos, z["xy"],
        {k: z[k] for k in ("origem", "destino", "distancia", "distancia_efetiva")},
        _desfecho("yf_event"), _desfecho("epizootia"), temporadas)

    quadro.to_parquet(SAIDA, index=False)
    DIAGNOSTICO.write_text(json.dumps(diag, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return quadro, diag


if __name__ == "__main__":
    q, d = construir()
    print(f"\nepidemia: {len(q):,} linhas x {len(d['variaveis'])} variaveis")
    print("variaveis:", ", ".join(d["variaveis"]))
    print("\navanco da frente entre temporadas (km):")
    for t, v in d["avanco_da_frente_km_por_temporada"].items():
        if v and v > 100:
            print(f"  {t}/{str(t + 1)[2:]}  {v:7.1f}")
