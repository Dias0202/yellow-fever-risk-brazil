"""Modelo de previsao, passo 6 - Multicolinearidade, regularizacao e monotonicidade.

A calibracao do passo 5 revelou um defeito que a metrica media escondia. O decil
de MENOR risco previsto tem taxa observada de 1,90% - maior que a dos oito decis
seguintes e quatro vezes a media. Dos 222 eventos que caem nele, 214 sao do
Sudeste. O modelo esta pondo no fundo da lista municipios que depois tem surto.

A causa esta nos coeficientes:

    pressao_acumulada            -1,85
    forca_k100                   +0,88
    focos_em_200km               -0,62
    forca_k25                    -0,47
    evento_temporada_anterior    -0,25

`pressao_acumulada` e uma media descontada de `forca_k100`, e as duas recebem
coeficientes grandes de sinais opostos. `evento_temporada_anterior` sai NEGATIVO,
contra os 19x de enriquecimento por persistencia medidos na exploracao inicial,
e as tres contagens de focos saem negativas, contra os 14x a 17x de contagio.

Isso e cancelamento entre preditores quase colineares. Nada nele e ilegal: o
ajuste minimiza a perda no conjunto todo e encontra uma combinacao que funciona na
media. Mas a combinacao e um fio de navalha, e onde a estrutura de correlacao
difere - o Sudeste durante a onda - ela produz predicao com o sinal errado.

Arvores nao sofrem disso; modelos lineares sim. E a logistica era o melhor modelo
do passo 2, de modo que o defeito precisa ser corrigido e nao contornado.

Duas correcoes, testadas prospectivamente
------------------------------------------
  A. REGULARIZACAO. L2 mais forte encolhe coeficientes de preditores colineares
     em vez de deixa-los se cancelarem. Testa-se uma faixa de C.

  B. PARCIMONIA. Dez variaveis de propagacao sao dez formas de medir "quao perto
     esteve a onda". Um conjunto com um representante por mecanismo remove a
     colinearidade na origem, em vez de pedir ao ajuste que a administre.

O criterio de aceitacao inclui MONOTONICIDADE da calibracao: um modelo cuja faixa
de menor risco previsto tem taxa observada acima da media nao pode ser publicado,
por melhor que seja a sua PR-AUC.

Rodar:
    PYTHONPATH=src python \\
        analysis/f6_06_regularizacao.py
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
    ESTRUTURAL, PROPAGACAO, SUSCETIBILIDADE, fabrica_logistica, fabrica_rf)

warnings.filterwarnings("ignore")

ALVO = "yf_event"
OPERACIONAL = PROPAGACAO + SUSCETIBILIDADE + ESTRUTURAL

# Um representante por mecanismo. A escolha nao e por busca automatica: cada
# entrada corresponde a uma quantidade epidemiologica distinta, e o que se remove
# sao formas alternativas de medir a mesma coisa.
PARCIMONIOSO = [
    # pressao de infeccao, em duas escalas - uma curta, de dispersao local do
    # vetor, e uma longa, compativel com deslocamento de hospedeiro e humano
    "forca_k50", "forca_k200",
    # o mesmo pela via da epizootia, que e o sinal sentinela
    "forca_epiz_k100",
    # proximidade bruta do foco, que nao e redundante com o nucleo: distingue
    # "um foco a 30 km" de "muitos focos a 150 km"
    "dist_foco_mais_proximo",
    # suscetibilidade
    "temporadas_desde_evento", "eventos_locais_10t",
    # adequabilidade estrutural
    "cobertura_florestal_t1", "altitude_media", "amplitude_altimetrica",
    "populacao_log_t1", "tmax_temporada_normal", "ppt_sazonalidade_normal",
]

CS = [0.003, 0.01, 0.03, 0.1, 0.5]
TOPOS = [50, 100, 250, 500, 1000]

SAIDA = cfg.METRICAS / "f6_06_regularizacao.json"
LOG = cfg.METRICAS / "f6_06_regularizacao.log"

_log: list[str] = []


def diga(t=""):
    print(t)
    _log.append(t)


def varredura(painel, colunas, ajustar):
    linhas = []
    for t in pv.temporadas_de_teste(painel):
        tr = painel[painel["temporada"] < t]
        te = painel[painel["temporada"] == t]
        if tr[ALVO].sum() < 10 or te[ALVO].sum() == 0:
            continue
        m = ajustar(tr[colunas], tr[ALVO], tr)
        linhas.append(pd.DataFrame({
            "temporada": t, "y": te[ALVO].to_numpy(),
            "p": m.predict_proba(te[colunas])[:, 1]}))
    return pd.concat(linhas, ignore_index=True)


def avaliar(oof):
    por_t, capt, tot = [], {k: 0 for k in TOPOS}, 0
    for t, g in oof.groupby("temporada"):
        m = pv.metricas(g["y"], g["p"], topos=[])
        if "erro" in m:
            continue
        por_t.append(m)
        y = g["y"].to_numpy(int)
        p = g["p"].to_numpy(float)
        tot += int(y.sum())
        for k in TOPOS:
            capt[k] += int(y[np.argsort(-p)[:k]].sum())
    w = np.array([m["positivos"] for m in por_t], float)

    # Monotonicidade da calibracao por decil.
    dec = pd.qcut(oof["p"].rank(method="first"), 10, labels=False,
                  duplicates="drop")
    taxa = oof.groupby(dec)["y"].mean().to_numpy()
    media = oof["y"].mean()
    return {
        "roc_auc": float(np.average([m["roc_auc"] for m in por_t], weights=w)),
        "vezes": float(np.average(
            [m["pr_auc_sobre_prevalencia"] for m in por_t], weights=w)),
        **{f"recall_top{k}": capt[k] / tot for k in TOPOS},
        "taxa_decil_1": float(taxa[0]),
        "taxa_decil_10": float(taxa[-1]),
        "razao_decil_10_sobre_1": float(taxa[-1] / max(taxa[0], 1e-12)),
        # Criterio de aceitacao: o decil mais baixo tem de ficar abaixo da media.
        "decil_1_abaixo_da_media": bool(taxa[0] < media),
        "correlacao_posto_decil": float(
            pd.Series(taxa).corr(pd.Series(range(len(taxa))), method="spearman")),
        "taxas_por_decil": [float(v) for v in taxa],
    }


def linha(nome, d):
    ok = "sim" if d["decil_1_abaixo_da_media"] else "NAO"
    return (f"{nome:34s} {d['roc_auc']:.3f} {d['vezes']:6.2f} "
            f"{d['recall_top100']:7.3f} {d['recall_top500']:7.3f} "
            f"{d['taxa_decil_1']:9.3%} {d['taxa_decil_10']:9.3%} "
            f"{d['correlacao_posto_decil']:+.2f}  {ok:>4s}")


CAB = (f"{'configuracao':34s} ROC   vezes  rec@100  rec@500  "
       f"dec1      dec10   rho  monot")


def main():
    cfg.garantir_diretorios()
    inicio = time.time()
    painel = pt.carregar()
    obs = painel[painel["observado"] == 1]
    resultados = {}

    diga(f"painel observado: {len(obs):,} linhas, {int(obs[ALVO].sum()):,} eventos")
    diga(f"prevalencia {obs[ALVO].mean():.4%}")
    diga("")
    diga("=" * 104)
    diga("A. REGULARIZACAO, conjunto operacional (31 variaveis)")
    diga("=" * 104)
    diga(CAB)
    for C in CS:
        d = avaliar(varredura(obs, OPERACIONAL, fabrica_logistica(C=C)))
        resultados[f"operacional_C{C}"] = d
        diga(linha(f"logistica 31 var, C={C}", d))

    diga("")
    diga("=" * 104)
    diga(f"B. PARCIMONIA ({len(PARCIMONIOSO)} variaveis)")
    diga("=" * 104)
    diga(CAB)
    for C in CS:
        d = avaliar(varredura(obs, PARCIMONIOSO, fabrica_logistica(C=C)))
        resultados[f"parcimonioso_C{C}"] = d
        diga(linha(f"logistica {len(PARCIMONIOSO)} var, C={C}", d))

    diga("")
    diga("=" * 104)
    diga("REFERENCIAS")
    diga("=" * 104)
    diga(CAB)
    d = avaliar(varredura(obs, PARCIMONIOSO,
                          fabrica_rf(peso_classe=False, max_features="sqrt")))
    resultados["floresta_parcimonioso"] = d
    diga(linha("floresta, parcimonioso", d))
    d = avaliar(varredura(obs, OPERACIONAL,
                          fabrica_rf(peso_classe=False, max_features="sqrt")))
    resultados["floresta_operacional"] = d
    diga(linha("floresta, operacional", d))

    base = obs[["temporada", ALVO, "forca_k100"]].rename(
        columns={ALVO: "y", "forca_k100": "p"}).copy()
    base = base[base["temporada"] >= pv.PRIMEIRA_PREVISTA]
    base["p"] = base["p"].fillna(0)
    d = avaliar(base)
    resultados["base_forca"] = d
    diga(linha("base: forca de infeccao", d))

    # ----------------------------------------------- coeficientes do melhor
    aceitos = {k: v for k, v in resultados.items()
               if v["decil_1_abaixo_da_media"] and k.startswith(("operacional",
                                                                 "parcimonioso"))}
    if aceitos:
        melhor = max(aceitos.items(), key=lambda kv: kv[1]["recall_top500"])
        diga("")
        diga(f"melhor entre os monotonos: {melhor[0]}")
        nome = melhor[0]
        cols = PARCIMONIOSO if nome.startswith("parcimonioso") else OPERACIONAL
        C = float(nome.split("_C")[1])
        m = fabrica_logistica(C=C)(obs[cols], obs[ALVO], obs)
        coef = pd.DataFrame({
            "variavel": cols,
            "coef": m.named_steps["modelo"].coef_[0]}).sort_values(
                "coef", ascending=False)
        diga("")
        diga("coeficientes (escala padronizada):")
        for _, r in coef.iterrows():
            diga(f"  {r['variavel']:30s} {r['coef']:+.3f}")
        resultados["coeficientes_do_melhor"] = coef.to_dict("records")
        resultados["melhor"] = nome
    else:
        diga("")
        diga("NENHUMA configuracao passou no criterio de monotonicidade.")

    SAIDA.write_text(json.dumps(
        {"cs": CS, "parcimonioso": PARCIMONIOSO, "operacional": OPERACIONAL,
         "resultados": resultados, "segundos": round(time.time() - inicio, 1)},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    LOG.write_text("\n".join(_log), encoding="utf-8")


if __name__ == "__main__":
    main()
