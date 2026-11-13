"""Modelo de previsao, passo 3 - Em que escala espacial a previsao se sustenta.

O passo 2 mostrou que o modelo, corrigido o peso de classe, supera a regra sem
aprendizado ao nivel municipal: 7,5x a 9,9x contra 8,1x da base, e 27 limiares com
beneficio liquido positivo. Ainda assim, capturar 35% dos
surtos numa lista de 500 municipios e desempenho modesto para operacao.

Este passo testa a hipotese de escala. A divisa municipal e administrativa e nao
epidemiologica: o ciclo enzootico ocorre em manchas de mata que a atravessam, e um
foco detectado num municipio e frequentemente o mesmo que apareceria no vizinho se
a equipe tivesse ido la. Agregar troca resolucao espacial por densidade de sinal.

    escala             unidades   prevalencia por temporada
    municipio             5.570        0,73%
    microrregiao            558        3,26%
    regiao imediata         510        3,38%

A comparacao entre escalas exige cuidado com a metrica. PR-AUC sobre prevalencia
ja normaliza pela raridade e e comparavel. Recall numa lista de tamanho fixo NAO
e: cem municipios sao 1,8% do pais e cem regioes imediatas sao 20%. Por isso as
listas de prioridade aqui sao proporcionais - 5%, 10% e 20% das unidades - e nao
de tamanho absoluto.

Rodar:
    PYTHONPATH=src python analysis/f6_03_escala.py
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
    COMPACTO, fabrica_logistica, fabrica_rf, fabrica_xgb)

warnings.filterwarnings("ignore")

ALVO = "yf_event"
SAIDA = cfg.METRICAS / "f6_03_escala.json"
LOG = cfg.METRICAS / "f6_03_escala.log"

# Listas de prioridade proporcionais ao numero de unidades. Ver o cabecalho.
FRACOES = [0.05, 0.10, 0.20]

ESCALAS = {
    "municipio": (None, "cod_mun"),
    "microrregiao": ("painel_microrregiao.parquet", "cod_microrregiao"),
    "regiao_imediata": ("painel_regiao_imediata.parquet", "cod_regiao_imediata"),
}

_log: list[str] = []


def diga(t=""):
    print(t)
    _log.append(t)


def metricas_proporcionais(y, proba) -> dict:
    """Como `prospectiva.metricas`, com listas de prioridade proporcionais."""
    y = np.asarray(y, int)
    proba = np.asarray(proba, float)
    n, pos = len(y), int(y.sum())
    if pos == 0:
        return {}
    base = pv.metricas(y, proba, topos=[])
    ordem = np.argsort(-proba)
    for f in FRACOES:
        k = max(1, int(round(f * n)))
        capt = int(y[ordem[:k]].sum())
        base[f"recall_top{int(f * 100)}pct"] = capt / pos
        base[f"unidades_por_surto_top{int(f * 100)}pct"] = (
            k / capt if capt else float("nan"))
    return base


def rodar(painel, colunas, ajustar, chave):
    testes = pv.temporadas_de_teste(painel)
    por_temporada, oof = [], []
    for t in testes:
        treino = painel[painel["temporada"] < t]
        teste = painel[painel["temporada"] == t]
        if treino[ALVO].sum() < 10 or teste[ALVO].sum() == 0:
            continue
        if getattr(ajustar, "__eh_base__", False):
            proba = ajustar(treino, teste)
        else:
            modelo = ajustar(treino[colunas], treino[ALVO], treino)
            proba = modelo.predict_proba(teste[colunas])[:, 1]
        m = metricas_proporcionais(teste[ALVO], proba)
        if not m:
            continue
        m["temporada"] = int(t)
        por_temporada.append(m)
        oof.append(pd.DataFrame({
            "unidade": teste[chave].to_numpy(), "temporada": t,
            "regiao": teste["regiao"].to_numpy(),
            "y": teste[ALVO].to_numpy(), "proba": proba}))
    oof = pd.concat(oof, ignore_index=True)
    curva = pv.curva_de_decisao(oof["y"], oof["proba"])
    return {"por_temporada": por_temporada,
            "agregado": metricas_proporcionais(oof["y"], oof["proba"]),
            "limiares_uteis": pv.limiares_uteis(curva),
            "curva_de_decisao": curva, "oof": oof}


def resumo(r):
    pts = [m for m in r["por_temporada"] if "pr_auc_sobre_prevalencia" in m]
    w = np.array([m["positivos"] for m in pts], float)
    return {
        "pooled_vezes": r["agregado"]["pr_auc_sobre_prevalencia"],
        "media_pond_vezes": float(np.average(
            [m["pr_auc_sobre_prevalencia"] for m in pts], weights=w)),
        "roc": r["agregado"]["roc_auc"],
        "recall_5pct": float(np.average(
            [m["recall_top5pct"] for m in pts], weights=w)),
        "recall_10pct": float(np.average(
            [m["recall_top10pct"] for m in pts], weights=w)),
        "recall_20pct": float(np.average(
            [m["recall_top20pct"] for m in pts], weights=w)),
        "unidades_por_surto_10pct": r["agregado"].get(
            "unidades_por_surto_top10pct"),
        "limiares_uteis": r["limiares_uteis"],
        "prevalencia": r["agregado"]["prevalencia"],
    }


def linha(nome, s):
    return (f"{nome:34s} {s['roc']:.3f} {s['pooled_vezes']:7.2f} "
            f"{s['media_pond_vezes']:8.2f} {s['recall_5pct']:8.3f} "
            f"{s['recall_10pct']:8.3f} {s['recall_20pct']:8.3f} "
            f"{s['limiares_uteis']:6d}")


CAB = (f"{'configuracao':34s} ROC   pooled  med_pond  rec@5%  rec@10%  "
       f"rec@20%  limiares")


def main():
    cfg.garantir_diretorios()
    inicio = time.time()
    resultados = {}
    melhores = {}

    for escala, (arquivo, chave) in ESCALAS.items():
        if arquivo is None:
            painel = pt.carregar()
        else:
            caminho = cfg.PROCESSADO / arquivo
            if not caminho.exists():
                diga(f"[pular] {escala}: {arquivo} nao existe")
                continue
            painel = pd.read_parquet(caminho)

        faltando = [c for c in COMPACTO if c not in painel.columns]
        colunas = [c for c in COMPACTO if c in painel.columns]

        diga("")
        diga("=" * 100)
        diga(f"ESCALA: {escala}  |  {painel[chave].nunique():,} unidades  |  "
             f"prevalencia {painel[ALVO].mean():.3%}  |  "
             f"{len(colunas)} variaveis")
        if faltando:
            diga(f"  variaveis ausentes nesta escala: {faltando}")
        diga("=" * 100)
        diga(CAB)

        conf = {
            "base: forca 100 km": (None, pv.BASES["vizinhanca"]),
            "logistica log-propagacao": (colunas, fabrica_logistica()),
            "floresta sem peso": (colunas,
                                  fabrica_rf(peso_classe=False,
                                             max_features="sqrt")),
            "xgb sem peso": (colunas, fabrica_xgb(peso_classe=False)),
        }
        for nome, (cols, aj) in conf.items():
            r = rodar(painel, cols, aj, chave)
            s = resumo(r)
            resultados[f"{escala}|{nome}"] = {
                "resumo": s, "por_temporada": r["por_temporada"]}
            diga(linha(nome, s))
            chaveM = f"{escala}|{nome}"
            if (escala not in melhores
                    or s["media_pond_vezes"] > melhores[escala][1]["media_pond_vezes"]):
                melhores[escala] = (chaveM, s, r)

    diga("")
    diga("=" * 100)
    diga("MELHOR POR ESCALA")
    diga("=" * 100)
    diga(CAB)
    for escala, (nome, s, _) in melhores.items():
        diga(linha(nome, s))

    # A melhor combinacao guarda as predicoes, para calibracao no passo seguinte.
    campea = max(melhores.items(), key=lambda kv: kv[1][1]["media_pond_vezes"])
    escala_campea, (nome_campeao, _, r_campeao) = campea
    r_campeao["oof"].to_parquet(
        cfg.PROCESSADO / "f6_03_predicoes_oof.parquet", index=False)
    diga("")
    diga(f"campea: {nome_campeao}")

    SAIDA.write_text(json.dumps({
        "fracoes": FRACOES, "resultados": resultados,
        "campea": nome_campeao,
        "curva_de_decisao_campea": r_campeao["curva_de_decisao"],
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    LOG.write_text("\n".join(_log), encoding="utf-8")


if __name__ == "__main__":
    main()
