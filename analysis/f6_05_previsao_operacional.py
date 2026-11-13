"""Modelo de previsao, passo 5 - A previsao para a temporada que ainda nao aconteceu.

Os passos anteriores mediram o que o modelo consegue. Este produz o artefato: a
lista ordenada de municipios para a proxima temporada, com probabilidade
calibrada, treinada em tudo que existe.

O modelo emitido e o parcimonioso com regularizacao forte
---------------------------------------------------------
Duas restricoes se somaram para determina-lo.

A primeira e de disponibilidade. Um modelo so pode usar, na hora de prever, o que
existe na hora de prever, e quatro variaveis do conjunto compacto - as anomalias
climaticas da temporada anterior - nao existem para a temporada a prever, porque a
serie do TerraClimate termina no ano civil de 2024 e deixa a temporada 2024/25 sem
os meses de janeiro a junho.

A segunda e de validade, e e a que mais pesou. O passo 6 mostrou que a logistica
sobre as 31 variaveis restantes sofria de multicolinearidade severa: dez formas
de medir "quao perto esteve a onda" recebiam coeficientes grandes de sinais
opostos, e `evento_temporada_anterior` saia NEGATIVO, contra os 19x de
persistencia medidos. O efeito visivel era o decil de MENOR risco previsto ter
taxa observada de 1,90%, quatro vezes a media.

O conjunto parcimonioso, com um representante por mecanismo e C = 0,003, corrige:
o decil mais baixo cai para 0,026%, a correlacao de posto entre decis vai a +1,00
e todo coeficiente fica com o sinal que a epidemiologia preve.

O desempenho e REMEDIDO prospectivamente com esse conjunto antes de a lista ser
emitida: seria desonesto anunciar o numero de um modelo e publicar a lista de
outro.

Escolha do modelo, e por que nao ha um so
------------------------------------------
Medido prospectivamente em 22 temporadas, nenhum componente domina:

    componente                 rec@50   rec@100   rec@500   rec@1000
    logistica log-propagacao    0,088     0,149     0,320      0,407
    forca de infeccao crua      0,039     0,083     0,362      0,541

A logistica e melhor no topo, a forca crua e melhor na cauda, e a diferenca no
topo e de quase o dobro. A leitura e epidemiologica: a logistica pondera pressao
de infeccao contra adequabilidade e refratariedade, o que separa os poucos
municipios de risco muito alto; a forca crua e monotona na proximidade do foco, o
que a torna melhor para varrer uma faixa larga em torno da onda.

A saida traz as duas ordenacoes, com a recomendacao declarada:

    lista curta, ate ~100 municipios   ->  ordenar por `posto_logistica`
    lista larga, 500 ou mais           ->  ordenar por `posto_forca`

Rodar:
    PYTHONPATH=src python \\
        analysis/f6_05_previsao_operacional.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression             # noqa: E402

from yfvrisk import config as cfg                               # noqa: E402
from yfvrisk.features import painel_temporada as pt             # noqa: E402
from yfvrisk.modeling import prospectiva as pv                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f6_02_ajuste import fabrica_logistica, fabrica_rf         # noqa: E402
from f6_06_regularizacao import PARCIMONIOSO                    # noqa: E402

warnings.filterwarnings("ignore")

ALVO = "yf_event"

# O conjunto parcimonioso do passo 6, com um representante por mecanismo. As
# variaveis dele estao todas disponiveis no instante da previsao.
OPERACIONAL = PARCIMONIOSO

# Regularizacao forte, escolhida prospectivamente no passo 6 pelo criterio de
# monotonicidade da calibracao. Ver o cabecalho.
C_REGULARIZACAO = 0.003

SAIDA = cfg.METRICAS / "f6_05_previsao_operacional.json"
TABELA = cfg.TABELAS / "f6_05_previsao_proxima_temporada.csv"
LOG = cfg.METRICAS / "f6_05_previsao_operacional.log"

TOPOS = [50, 100, 250, 500, 1000]

_log: list[str] = []


def diga(t=""):
    print(t)
    _log.append(t)


def varredura(painel, colunas):
    """Passada prospectiva com o conjunto operacional.

    Devolve as predicoes fora da amostra, que servem a duas coisas: medir o
    desempenho que sera anunciado junto com a lista, e ajustar a calibracao.
    """
    linhas = []
    for t in pv.temporadas_de_teste(painel):
        tr = painel[(painel["temporada"] < t) & (painel["observado"] == 1)]
        te = painel[painel["temporada"] == t]
        if tr[ALVO].sum() < 10 or te[ALVO].sum() == 0:
            continue
        log = fabrica_logistica(C=C_REGULARIZACAO)(tr[colunas], tr[ALVO], tr)
        linhas.append(pd.DataFrame({
            "temporada": t, "y": te[ALVO].to_numpy(),
            "logistica": log.predict_proba(te[colunas])[:, 1],
            "forca": te["forca_k100"].fillna(0).to_numpy(),
        }))
        diga(f"  {t}/{str(t + 1)[2:]}  n={len(te):,}  pos={int(te[ALVO].sum()):4d}")
    return pd.concat(linhas, ignore_index=True)


def desempenho(quadro, coluna):
    por_t, capt, tot = [], {k: 0 for k in TOPOS}, 0
    for t, g in quadro.groupby("temporada"):
        m = pv.metricas(g["y"], g[coluna], topos=[])
        if "erro" in m:
            continue
        por_t.append(m)
        y = g["y"].to_numpy(int)
        p = g[coluna].to_numpy(float)
        tot += int(y.sum())
        for k in TOPOS:
            capt[k] += int(y[np.argsort(-p)[:k]].sum())
    w = np.array([m["positivos"] for m in por_t], float)
    saida = {
        "roc_auc": float(np.average([m["roc_auc"] for m in por_t], weights=w)),
        "pr_auc_sobre_prevalencia": float(np.average(
            [m["pr_auc_sobre_prevalencia"] for m in por_t], weights=w)),
        "temporadas": len(por_t),
    }
    for k in TOPOS:
        saida[f"recall_top{k}"] = capt[k] / tot
        saida[f"investigados_por_surto_top{k}"] = (
            k * len(por_t) / capt[k] if capt[k] else float("nan"))
    return saida


def main():
    cfg.garantir_diretorios()
    painel = pt.carregar()
    observado = painel[painel["observado"] == 1]
    ultima_obs = int(observado["temporada"].max())
    prever = int(painel["temporada"].max())

    diga(f"ultima temporada observada: {ultima_obs}/{str(ultima_obs + 1)[2:]}")
    diga(f"temporada a prever:         {prever}/{str(prever + 1)[2:]}")
    diga(f"conjunto operacional: {len(OPERACIONAL)} variaveis")
    faltam = [c for c in OPERACIONAL
              if painel.loc[painel["temporada"] == prever, c].isna().mean() > 0.5]
    if faltam:
        raise SystemExit(f"variaveis sem cobertura na temporada a prever: {faltam}")
    diga("")

    # ------------------------------------------ desempenho a ser anunciado
    diga("varredura prospectiva com o conjunto operacional")
    oof = varredura(observado, OPERACIONAL)
    diga("")
    diga(f"{'ordenacao':16s} ROC   vezes  " + "  ".join(f"rec@{k}" for k in TOPOS))
    perf = {}
    for col, rot in [("logistica", "logistica"), ("forca", "forca de infeccao")]:
        d = desempenho(oof, col)
        perf[col] = d
        diga(f"{rot:16s} {d['roc_auc']:.3f} {d['pr_auc_sobre_prevalencia']:6.2f}  "
             + "  ".join(f"{d[f'recall_top{k}']:6.3f}" for k in TOPOS))

    # ------------------------------------------------------- calibracao
    # Platt sobre o log da saida da logistica, ajustado nas predicoes FORA da
    # amostra. Calibrar sobre o treino devolveria a confianca do sobreajuste.
    x = np.log(np.clip(oof["logistica"].to_numpy(), 1e-9, 1)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(x, oof["y"].to_numpy(int))

    faixas = pd.qcut(oof["logistica"].rank(method="first"), 10,
                     labels=False, duplicates="drop")
    calib = []
    diga("")
    diga("calibracao fora da amostra, por decil de risco previsto:")
    diga("  decil        n   previsto   observado")
    for f in sorted(pd.unique(faixas)):
        sel = faixas == f
        pv_ = float(cal.predict_proba(x[sel.to_numpy()])[:, 1].mean())
        ob = float(oof.loc[sel, "y"].mean())
        calib.append({"faixa": int(f) + 1, "n": int(sel.sum()),
                      "previsto": pv_, "observado": ob})
        diga(f"  {f + 1:5d} {int(sel.sum()):8,}   {pv_:8.4%}   {ob:9.4%}")

    # ------------------------------------------------- modelo e previsao
    treino = observado
    diga("")
    diga(f"treino final: {len(treino):,} linhas, {int(treino[ALVO].sum()):,} eventos")
    modelo = fabrica_logistica(C=C_REGULARIZACAO)(
        treino[OPERACIONAL], treino[ALVO], treino)
    floresta = fabrica_rf(peso_classe=False, max_features="sqrt")(
        treino[OPERACIONAL], treino[ALVO], treino)

    alvo_linhas = painel[painel["temporada"] == prever].copy()
    alvo_linhas["risco_logistica"] = modelo.predict_proba(
        alvo_linhas[OPERACIONAL])[:, 1]
    alvo_linhas["risco_floresta"] = floresta.predict_proba(
        alvo_linhas[OPERACIONAL])[:, 1]
    alvo_linhas["risco_forca"] = alvo_linhas["forca_k100"].fillna(0)
    alvo_linhas["probabilidade"] = cal.predict_proba(
        np.log(np.clip(alvo_linhas["risco_logistica"], 1e-9, 1)
               ).to_numpy().reshape(-1, 1))[:, 1]
    alvo_linhas["posto_logistica"] = alvo_linhas["risco_logistica"].rank(
        ascending=False, method="first").astype(int)
    alvo_linhas["posto_forca"] = alvo_linhas["risco_forca"].rank(
        ascending=False, method="first").astype(int)

    nomes = pd.read_parquet(cfg.PROCESSADO / "municipios_referencia.parquet",
                            columns=["cod_mun", "nome_mun"])
    colunas = ["cod_mun", "nome_mun", "uf", "regiao", "temporada",
               "probabilidade", "posto_logistica", "posto_forca",
               "risco_logistica", "risco_floresta", "risco_forca",
               "dist_foco_mais_proximo", "focos_em_100km",
               "temporadas_desde_evento", "evento_temporada_anterior",
               "cobertura_florestal_t1", "altitude_media", "populacao_log_t1"]
    tabela = (alvo_linhas.merge(nomes, on="cod_mun", how="left")
              [[c for c in colunas if c in alvo_linhas.columns
                or c == "nome_mun"]]
              .sort_values("posto_logistica"))
    tabela.to_csv(TABELA, index=False, encoding="utf-8")

    diga("")
    diga(f"30 municipios de maior risco previsto - temporada "
         f"{prever}/{str(prever + 1)[2:]}")
    diga("  #  municipio                  UF   prob.   foco+prox  focos<100km  desde")
    for _, r in tabela.head(30).iterrows():
        diga(f"{r['posto_logistica']:3d}  {str(r['nome_mun'])[:24]:24s} "
             f"{r['uf']:3s} {r['probabilidade']:6.2%}  "
             f"{r['dist_foco_mais_proximo']:8.0f} km  "
             f"{r['focos_em_100km']:10.0f}  {r['temporadas_desde_evento']:5.0f}")

    diga("")
    diga("distribuicao regional dos 250 primeiros")
    for reg, n in tabela.head(250)["regiao"].value_counts().items():
        diga(f"  {reg:14s} {n:4d}")

    SAIDA.write_text(json.dumps({
        "temporada_prevista": prever,
        "ultima_observada": ultima_obs,
        "variaveis_operacionais": OPERACIONAL,
        "C_regularizacao": C_REGULARIZACAO,
        "variaveis_excluidas_por_indisponibilidade": [
            "ppt_pre_estacao_anomalia", "ppt_transmissao_anomalia",
            "tmax_transmissao_anomalia", "vpd_temporada_anomalia"],
        "motivo_exclusao": (
            "A serie do TerraClimate termina no ano civil de 2024, o que deixa a "
            "temporada 2024/25 incompleta e sem as anomalias climaticas. As "
            "demais foram removidas por multicolinearidade - ver o passo 6."),
        "desempenho_prospectivo": perf,
        "calibracao_fora_da_amostra": calib,
        "n_treino": int(len(treino)), "eventos_treino": int(treino[ALVO].sum()),
        "tabela": str(TABELA),
        "recomendacao": (
            "Lista curta (ate ~100): ordenar por `posto_logistica`. Lista larga "
            "(500+): ordenar por `posto_forca`. Nenhuma domina a outra."),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    LOG.write_text("\n".join(_log), encoding="utf-8")
    print(f"\ntabela em {TABELA}")


if __name__ == "__main__":
    main()
