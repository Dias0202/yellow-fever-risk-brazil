"""Modelo de previsao, passo 4 - Modelo final, calibracao e utilidade operacional.

O que os passos anteriores estabeleceram:

  - a validacao que corresponde ao uso e prospectiva por temporada;
  - peso de classe destroi a ordenacao e foi removido;
  - a escala municipal e a melhor: agregar em microrregiao ou regiao imediata
    eleva a prevalencia mas reduz a razao sobre ela;
  - dois modelos superam a regra sem aprendizado, e por caminhos diferentes.

O ultimo ponto e o que este passo explora. As curvas nao se dominam:

    modelo                      pooled   rec@5%   rec@20%
    base: forca de infeccao       5,27    0,198     0,582
    logistica log-propagacao      7,75    0,240     0,427
    floresta sem peso             6,61    0,252     0,520

A logistica e a melhor no topo da lista, a base e a melhor na cauda. Ordenacoes
que se cruzam assim sao o caso classico em que combinar ajuda - e combinar por
POSTO, e nao por probabilidade, porque as tres saidas estao em escalas
incomparaveis e o que se quer delas e a ordem.

O que mais este passo faz
-------------------------
1. **Desfechos separados.** Epizootia em primata nao humano e caso humano tem
   sistemas de vigilancia, vieses e utilidades distintas. A epizootia e o sinal
   sentinela do sistema brasileiro e merece modelo proprio.

2. **Calibracao.** Ordenar bem nao e o mesmo que acertar a magnitude. A
   calibracao e de Platt, ajustada sobre as temporadas ANTERIORES a cada previsao,
   nunca sobre a propria - seria vazamento. Platt e nao isotonica porque a
   isotonica, sobre 117 mil pontos com 0,7% de positivos, colapsa milhares de
   municipios em poucos degraus e destroi a ordenacao dentro da temporada.

3. **Chegada contra continuacao.** O passo 1 ja indicava que o modelo preve bem a
   continuacao de uma onda e mal a sua chegada. Aqui isso e medido: cada unidade
   de teste e classificada conforme houvesse ou nao foco a menos de 200 km na
   temporada anterior, e o desempenho e reportado nos dois conjuntos.

4. **Tabela operacional.** Quantos municipios investigar, quantos surtos se
   captura, quantos alarmes falsos por surto. E a forma em que a decisao e de
   fato tomada.

Rodar:
    PYTHONPATH=src python analysis/f6_04_modelo_final.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yfvrisk import config as cfg                               # noqa: E402
from yfvrisk.features import painel_temporada as pt             # noqa: E402
from yfvrisk.modeling import prospectiva as pv                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6_02_ajuste import (                                      # noqa: E402
    COMPACTO, fabrica_logistica, fabrica_rf)

warnings.filterwarnings("ignore")

SAIDA = cfg.METRICAS / "f6_04_modelo_final.json"
LOG = cfg.METRICAS / "f6_04_modelo_final.log"
OOF = cfg.PROCESSADO / "f6_04_predicoes_oof.parquet"

# Distancia abaixo da qual se considera que havia onda ativa por perto na
# temporada anterior. 200 km e a escala em que o enriquecimento por contagio
# ainda e de 14x (ver `yfvrisk/features/espacial.py`).
RAIO_ONDA = 200.0

TOPOS = [50, 100, 250, 500, 1000]

_log: list[str] = []


def diga(t=""):
    print(t)
    _log.append(t)


def posto_normalizado(v: np.ndarray) -> np.ndarray:
    """Posto em [0, 1]. Combinar por posto e nao por probabilidade porque as
    saidas dos componentes estao em escalas incomparaveis."""
    ordem = np.argsort(np.argsort(v))
    return ordem / max(len(v) - 1, 1)


def rodar_componentes(painel, alvo):
    """Uma passada prospectiva produzindo as saidas de todos os componentes.

    Ajustar os tres na mesma varredura garante que compartilhem exatamente as
    mesmas particoes, o que e condicao para que a combinacao por posto seja
    legitima.
    """
    logistica = fabrica_logistica()
    floresta = fabrica_rf(peso_classe=False, max_features="sqrt")

    linhas = []
    for t in pv.temporadas_de_teste(painel):
        treino = painel[painel["temporada"] < t]
        teste = painel[painel["temporada"] == t]
        if treino[alvo].sum() < 10 or teste[alvo].sum() == 0:
            continue

        p_log = logistica(treino[COMPACTO], treino[alvo], treino).predict_proba(
            teste[COMPACTO])[:, 1]
        p_rf = floresta(treino[COMPACTO], treino[alvo], treino).predict_proba(
            teste[COMPACTO])[:, 1]
        p_base = teste["forca_k100"].fillna(0).to_numpy(dtype=float)

        linhas.append(pd.DataFrame({
            "cod_mun": teste["cod_mun"].to_numpy(),
            "temporada": t,
            "regiao": teste["regiao"].to_numpy(),
            "y": teste[alvo].to_numpy(),
            "dist_foco": teste["dist_foco_mais_proximo"].to_numpy(),
            "logistica": p_log, "floresta": p_rf, "base": p_base,
        }))
        diga(f"  {t}/{str(t + 1)[2:]}  n={len(teste):,}  pos={int(teste[alvo].sum()):4d}")
    return pd.concat(linhas, ignore_index=True)


def combinar(quadro: pd.DataFrame, pesos: dict) -> np.ndarray:
    """Media ponderada dos postos, calculada DENTRO de cada temporada.

    Dentro da temporada porque a lista de prioridade e produzida temporada a
    temporada: o posto de um municipio so tem sentido contra os demais daquela
    mesma temporada.
    """
    saida = np.zeros(len(quadro))
    for _, idx in quadro.groupby("temporada").groups.items():
        pos = quadro.index.get_indexer(idx)
        acc = np.zeros(len(pos))
        for nome, w in pesos.items():
            acc += w * posto_normalizado(quadro[nome].to_numpy()[pos])
        saida[pos] = acc / sum(pesos.values())
    return saida


def avaliar(quadro, coluna, rotulo) -> dict:
    """Metricas agregadas a partir das POR TEMPORADA, nao sobre o conjunto unido.

    A distincao nao e cosmetica. Uma lista de prioridade e produzida temporada a
    temporada: o gestor escolhe os 100 municipios daquele ano, nao os 100 de
    vinte anos juntos. Calcular recall@100 sobre as 117 mil linhas unidas mede
    outra coisa - e para a forca de infeccao crua chega a dar zero, porque os cem
    maiores valores de todo o periodo caem todos dentro das duas temporadas da
    onda.

    Por isso todo recall@K aqui e a soma dos capturados por temporada sobre o
    total de surtos, e a razao PR-AUC/prevalencia agregada e a media das
    temporadas ponderada pelo numero de eventos.
    """
    por_t = []
    for t, g in quadro.groupby("temporada"):
        mt = pv.metricas(g["y"], g[coluna], topos=TOPOS)
        if "erro" in mt:
            continue
        mt["temporada"] = int(t)
        por_t.append(mt)

    w = np.array([x["positivos"] for x in por_t], float)
    m = {
        "n": int(len(quadro)), "positivos": int(quadro["y"].sum()),
        "prevalencia": float(quadro["y"].mean()),
        "roc_auc": float(np.average([x["roc_auc"] for x in por_t], weights=w)),
        "pr_auc": float(np.average([x["pr_auc"] for x in por_t], weights=w)),
        "pr_auc_sobre_prevalencia": float(np.average(
            [x["pr_auc_sobre_prevalencia"] for x in por_t], weights=w)),
        "rotulo": rotulo,
        "n_temporadas": len(por_t),
    }
    # Recall@K: capturados somados sobre surtos somados, o que corresponde a
    # operar a lista todas as temporadas.
    for k in TOPOS:
        capt = tot = 0
        for _, g in quadro.groupby("temporada"):
            y = g["y"].to_numpy(int)
            ordem = np.argsort(-g[coluna].to_numpy(float))[:k]
            capt += int(y[ordem].sum())
            tot += int(y.sum())
        m[f"recall_top{k}"] = capt / tot if tot else float("nan")
        m[f"investigados_por_surto_top{k}"] = (
            (k * len(por_t)) / capt if capt else float("nan"))

    curva = pv.curva_de_decisao(quadro["y"], quadro[coluna])
    m["limiares_uteis"] = pv.limiares_uteis(curva)
    return {"agregado": m, "por_temporada": por_t, "curva_de_decisao": curva}


def calibrar(quadro, coluna) -> np.ndarray:
    """Calibracao de Platt sobre as temporadas anteriores a cada previsao.

    Platt, e nao isotonica. A isotonica sobre 117 mil pontos com 0,7% de
    positivos produz uma escada de poucos degraus, e cada degrau colapsa milhares
    de municipios no mesmo valor. Isso destroi a ordenacao dentro da temporada -
    medido: a ROC cai de 0,72 para 0,51 sem que nada tenha mudado no modelo. Um
    calibrador nao pode piorar a ordenacao; se piora, ele esta errado para o caso.

    Platt e uma logistica de um parametro sobre o escore e por construcao
    estritamente monotona, de modo que a ordem e preservada exatamente e so a
    escala muda - que e tudo o que se pede de uma calibracao aqui, ja que e a
    escala que a curva de decisao precisa.

    Ajustada sempre sobre o passado. Ajustar sobre a propria temporada prevista
    seria vazamento.
    """
    from sklearn.linear_model import LogisticRegression

    escore = quadro[coluna].to_numpy(float)
    y = quadro["y"].to_numpy(int)
    temporadas = quadro["temporada"].to_numpy()
    saida = np.full(len(quadro), np.nan)

    for i, t in enumerate(sorted(np.unique(temporadas))):
        sel = temporadas == t
        antes = temporadas < t
        if antes.sum() < 2000 or y[antes].sum() < 20:
            # Sem passado suficiente, devolve a prevalencia observada ate ali -
            # honesto e monotono, ainda que sem informacao.
            saida[sel] = y[antes].mean() if antes.sum() else quadro["y"].mean()
            continue
        cal = LogisticRegression(C=1e6, max_iter=1000)
        cal.fit(escore[antes].reshape(-1, 1), y[antes])
        saida[sel] = cal.predict_proba(escore[sel].reshape(-1, 1))[:, 1]
    return saida


def erro_de_calibracao(quadro, coluna, faixas=10) -> list[dict]:
    """Previsto contra observado, por decil de risco previsto."""
    q = quadro.copy()
    q["faixa"] = pd.qcut(q[coluna].rank(method="first"), faixas,
                         labels=False, duplicates="drop")
    saida = []
    for f, g in q.groupby("faixa"):
        saida.append({
            "faixa": int(f), "n": int(len(g)),
            "previsto": float(g[coluna].mean()),
            "observado": float(g["y"].mean()),
        })
    return saida


def tabela_operacional(quadro, coluna) -> list[dict]:
    """Quantos investigar, quantos capturar, a que custo - por temporada."""
    linhas = []
    for k in TOPOS:
        capt = tot = 0
        for _, g in quadro.groupby("temporada"):
            p = g[coluna].to_numpy(float)
            y = g["y"].to_numpy(int)
            ordem = np.argsort(-p)[:k]
            capt += int(y[ordem].sum())
            tot += int(y.sum())
        n_temp = quadro["temporada"].nunique()
        linhas.append({
            "municipios_investigados": k,
            "surtos_capturados": capt, "surtos_totais": tot,
            "recall": capt / tot if tot else float("nan"),
            "investigados_por_surto": (k * n_temp) / capt if capt else float("nan"),
        })
    return linhas


def main():
    cfg.garantir_diretorios()
    inicio = time.time()
    painel = pt.carregar()
    resultados = {}

    for alvo in ("yf_event", "caso_humano", "epizootia"):
        diga("")
        diga("=" * 96)
        diga(f"DESFECHO: {alvo}  |  prevalencia {painel[alvo].mean():.4%}")
        diga("=" * 96)
        quadro = rodar_componentes(painel, alvo)

        quadro["conjunto"] = combinar(
            quadro, {"logistica": 1.0, "floresta": 1.0, "base": 1.0})
        quadro["conjunto_2"] = combinar(
            quadro, {"logistica": 2.0, "floresta": 1.0, "base": 1.0})

        diga("")
        diga(f"{'componente':22s} ROC   PR-AUC  vezes  rec@50  rec@100  "
             f"rec@500  rec@1000  limiares")
        aval = {}
        for col in ("base", "logistica", "floresta", "conjunto", "conjunto_2"):
            a = avaliar(quadro, col, col)
            aval[col] = a
            g = a["agregado"]
            diga(f"{col:22s} {g['roc_auc']:.3f}  {g['pr_auc']:.4f}  "
                 f"{g['pr_auc_sobre_prevalencia']:5.2f}  "
                 f"{g['recall_top50']:6.3f}  {g['recall_top100']:7.3f}  "
                 f"{g['recall_top500']:7.3f}  {g['recall_top1000']:8.3f}  "
                 f"{g['limiares_uteis']:6d}")

        # O criterio de escolha e o recall a 500 municipios, e nao a PR-AUC.
        # A PR-AUC resume a curva inteira, inclusive faixas de limiar que ninguem
        # operaria; 500 municipios por temporada e uma escala de mobilizacao
        # plausivel e e nela que a escolha tem consequencia.
        melhor = max(aval.items(),
                     key=lambda kv: kv[1]["agregado"]["recall_top500"])[0]
        diga(f"melhor: {melhor}")

        quadro["calibrado"] = calibrar(quadro, melhor)
        cal = avaliar(quadro, "calibrado", "calibrado")
        calib = erro_de_calibracao(quadro, "calibrado")
        g = cal["agregado"]
        diga(f"{'calibrado (Platt)':22s} {g['roc_auc']:.3f}  {g['pr_auc']:.4f}  "
             f"{g['pr_auc_sobre_prevalencia']:5.2f}  "
             f"{g['recall_top50']:6.3f}  {g['recall_top100']:7.3f}  "
             f"{g['recall_top500']:7.3f}  {g['recall_top1000']:8.3f}  "
             f"{g['limiares_uteis']:6d}")

        # ------------------------------------------------ chegada x continuacao
        perto = quadro["dist_foco"].to_numpy() <= RAIO_ONDA
        diga("")
        diga(f"chegada contra continuacao (foco a menos de {RAIO_ONDA:.0f} km "
             f"na temporada anterior)")
        recorte = {}
        for nome, sel in [("continuacao (perto de foco)", perto),
                          ("chegada (longe de foco)", ~perto)]:
            sub = quadro[sel]
            if sub["y"].sum() < 10:
                continue
            m = pv.metricas(sub["y"], sub[melhor], topos=[100, 500])
            recorte[nome] = m
            diga(f"  {nome:30s} n={m['n']:6,}  pos={m['positivos']:4d}  "
                 f"prev={m['prevalencia']:.3%}  ROC {m['roc_auc']:.3f}  "
                 f"{m['pr_auc_sobre_prevalencia']:6.2f}x")

        tabela = tabela_operacional(quadro, melhor)
        if alvo == "yf_event":
            diga("")
            diga("tabela operacional (por temporada)")
            diga("  investigados   surtos capturados   recall   investigados/surto")
            for linha in tabela:
                diga(f"  {linha['municipios_investigados']:11d}   "
                     f"{linha['surtos_capturados']:6d} de {linha['surtos_totais']:<6d}  "
                     f"{linha['recall']:7.1%}   {linha['investigados_por_surto']:14.1f}")
            quadro.to_parquet(OOF, index=False)

        resultados[alvo] = {
            "prevalencia": float(painel[alvo].mean()),
            "melhor_componente": melhor,
            "componentes": {k: v["agregado"] for k, v in aval.items()},
            "por_temporada_do_melhor": aval[melhor]["por_temporada"],
            "calibrado": cal["agregado"],
            "curva_de_decisao": aval[melhor]["curva_de_decisao"],
            "chegada_x_continuacao": recorte,
            "calibracao": calib,
            "curva_de_decisao_calibrada": cal["curva_de_decisao"],
            "tabela_operacional": tabela,
        }

    SAIDA.write_text(json.dumps(
        {"raio_onda_km": RAIO_ONDA, "topos": TOPOS, "resultados": resultados,
         "segundos": round(time.time() - inicio, 1)},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    LOG.write_text("\n".join(_log), encoding="utf-8")


if __name__ == "__main__":
    main()
