"""Validacao em bloco - Quanto a validacao ingenua superestima.

Compara, no mesmo painel e com o mesmo modelo, particao aleatoria, espacial e
temporal, contra modelos de referencia triviais, e separa o desfecho humano do
epizootico.

A pergunta central nao e "qual o melhor modelo", e sim "quanto do desempenho
medido sobrevive quando o modelo precisa de fato prever o futuro".

Rodar:
    python analysis/f3_01_comparar_validacao.py
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

from sklearn.ensemble import RandomForestClassifier          # noqa: E402
from sklearn.impute import SimpleImputer                     # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.pipeline import Pipeline                        # noqa: E402
from sklearn.preprocessing import StandardScaler             # noqa: E402

from yfvrisk import config as cfg                            # noqa: E402
from yfvrisk.features.painel import GRUPOS                   # noqa: E402
from yfvrisk.modeling.metricas import avaliar                # noqa: E402
from yfvrisk.modeling.validacao import (                     # noqa: E402
    BlocoEspacial, JanelaTemporal, particao_aleatoria_ingenu,
)

warnings.filterwarnings("ignore")

SAIDA = cfg.METRICAS / "f3_01_comparacao_validacao.json"
ALVO = "yf_event"
ANO_INICIAL_TESTE = 2005


def montar_variaveis(painel, com_historico: bool):
    grupos = ["CLIMA", "CLIMA_DEFASADO", "PAISAGEM", "IMUNIDADE", "DEMOGRAFIA"]
    if com_historico:
        grupos.append("HISTORICO")
    colunas = [c for g in grupos for c in GRUPOS.get(g, []) if c in painel.columns]
    return colunas


def floresta_aleatoria():
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=42)),
    ])


def regressao_logistica():
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=2000, class_weight="balanced",
                                      C=1.0, random_state=42)),
    ])


def referencia_historico(painel, treino, teste):
    """Prever pelo historico do proprio municipio, sem nenhuma variavel ambiental.

    E a referencia obrigatoria. Um modelo ambiental
    que nao supere isto nao esta agregando informacao ecologica: esta apenas
    memorizando onde ja houve surto.
    """
    hist = painel.iloc[treino].groupby("cod_mun")[ALVO].mean()
    return painel.iloc[teste]["cod_mun"].map(hist).fillna(hist.mean()).to_numpy()


def referencia_prevalencia(painel, treino, teste):
    return np.full(len(teste), painel.iloc[treino][ALVO].mean())


def executar():
    painel = pd.read_parquet(cfg.PROCESSADO / "painel_analitico.parquet")
    painel = painel.sort_values(["cod_mun", "ano"]).reset_index(drop=True)

    y = painel[ALVO].to_numpy()
    anos = painel["ano"].to_numpy()
    ufs = painel["uf"].to_numpy()

    print("=" * 92)
    print("VALIDACAO EM BLOCO - Quanto a validacao ingenua superestima")
    print("=" * 92)
    print(f"\npainel: {len(painel):,} linhas, prevalencia {y.mean():.4%}")

    resultados = []

    def registrar(rotulo, esquema, res, extra=None):
        linha = {"rotulo": rotulo, "esquema": esquema, **res}
        if extra:
            linha.update(extra)
        resultados.append(linha)
        if "erro" in res:
            print(f"  {rotulo:52s} {res['erro']}")
        else:
            print(f"  {rotulo:52s} AUC {res['roc_auc']:.4f}  "
                  f"PR-AUC {res['pr_auc']:.4f}  prec {res['precisao']:.3f}  "
                  f"rec {res['recall']:.3f}")

    # ---------------------------------------------------------- referencias
    print("\n[REFERENCIAS] sem nenhuma variavel ambiental")
    tr, te = particao_aleatoria_ingenu(len(painel))
    registrar("prevalencia constante", "aleatorio",
              avaliar(y[te], referencia_prevalencia(painel, tr, te)))
    registrar("historico do proprio municipio", "aleatorio",
              avaliar(y[te], referencia_historico(painel, tr, te)))

    # ---------------------------------------------------------- ingenuo
    for com_hist in (False, True):
        cols = montar_variaveis(painel, com_hist)
        X = painel[cols]
        sufixo = "com historico" if com_hist else "so ambiental"
        print(f"\n[A] Particao aleatoria por linha - {sufixo}")
        t0 = time.time()
        m = floresta_aleatoria().fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        registrar(f"floresta aleatoria | {sufixo}", "A_aleatorio",
                  avaliar(y[te], p), {"segundos": round(time.time() - t0, 1),
                                      "n_variaveis": len(cols)})

    # ---------------------------------------------------------- temporal
    for com_hist in (False, True):
        cols = montar_variaveis(painel, com_hist)
        X = painel[cols]
        sufixo = "com historico" if com_hist else "so ambiental"
        print(f"\n[B] Janela temporal expansiva, teste {ANO_INICIAL_TESTE}-2025 - {sufixo}")

        oof = np.full(len(painel), np.nan)
        por_ano = []
        cv = JanelaTemporal(ano_inicial_teste=ANO_INICIAL_TESTE, minimo_treino=10)
        t0 = time.time()
        for treino, teste, info in cv.split(anos):
            m = floresta_aleatoria().fit(X.iloc[treino], y[treino])
            pr = m.predict_proba(X.iloc[teste])[:, 1]
            oof[teste] = pr
            if len(np.unique(y[teste])) == 2:
                r = avaliar(y[teste], pr)
                por_ano.append({"ano": info["ano_teste"], "roc_auc": r["roc_auc"],
                                "pr_auc": r["pr_auc"], "positivos": r["positivos"]})
        valido = ~np.isnan(oof)
        registrar(f"floresta aleatoria | {sufixo}", "B_temporal",
                  avaliar(y[valido], oof[valido]),
                  {"segundos": round(time.time() - t0, 1),
                   "por_ano": por_ano, "n_variaveis": len(cols)})

        if not com_hist:
            print("      desempenho por ano de teste:")
            for r in por_ano:
                print(f"        {r['ano']}  AUC {r['roc_auc']:.3f}  "
                      f"PR-AUC {r['pr_auc']:.3f}  ({r['positivos']} positivos)")

    # ---------------------------------------------------------- espacial
    cols = montar_variaveis(painel, com_historico=False)
    X = painel[cols]
    print("\n[C] Bloco espacial, leave-UF-out - so ambiental")
    oof = np.full(len(painel), np.nan)
    t0 = time.time()
    for treino, teste, info in BlocoEspacial().split(ufs):
        m = floresta_aleatoria().fit(X.iloc[treino], y[treino])
        oof[teste] = m.predict_proba(X.iloc[teste])[:, 1]
    registrar("floresta aleatoria | so ambiental", "C_espacial",
              avaliar(y, oof), {"segundos": round(time.time() - t0, 1)})

    # ---------------------------------------------------------- regressao
    print("\n[D] Regressao logistica penalizada, janela temporal - so ambiental")
    oof = np.full(len(painel), np.nan)
    t0 = time.time()
    for treino, teste, info in JanelaTemporal(ANO_INICIAL_TESTE, minimo_treino=10).split(anos):
        m = regressao_logistica().fit(X.iloc[treino], y[treino])
        oof[teste] = m.predict_proba(X.iloc[teste])[:, 1]
    valido = ~np.isnan(oof)
    registrar("regressao logistica | so ambiental", "D_temporal_logistica",
              avaliar(y[valido], oof[valido]), {"segundos": round(time.time() - t0, 1)})

    # ---------------------------------------------------------- desfechos
    print("\n[E] Desfechos separados, janela temporal - so ambiental")
    for alvo, rotulo in [("caso_humano", "caso humano"), ("epizootia", "epizootia")]:
        ya = painel[alvo].to_numpy()
        oof = np.full(len(painel), np.nan)
        for treino, teste, info in JanelaTemporal(ANO_INICIAL_TESTE, minimo_treino=10).split(anos):
            m = floresta_aleatoria().fit(X.iloc[treino], ya[treino])
            oof[teste] = m.predict_proba(X.iloc[teste])[:, 1]
        valido = ~np.isnan(oof)
        registrar(f"desfecho: {rotulo}", "E_desfecho_separado",
                  avaliar(ya[valido], oof[valido]))

    # ---------------------------------------------------------- sintese
    print("\n" + "=" * 92)
    print("SINTESE")
    print("=" * 92)
    print(f"{'esquema':26s} {'modelo':34s} {'AUC':>7s} {'PR-AUC':>8s} "
          f"{'prec':>6s} {'rec':>6s} {'invest/evento':>14s}")
    for r in resultados:
        if "erro" in r:
            continue
        inv = r.get("municipios_investigados_por_evento")
        print(f"{r['esquema']:26s} {r['rotulo'][:34]:34s} {r['roc_auc']:7.4f} "
              f"{r['pr_auc']:8.4f} {r['precisao']:6.3f} {r['recall']:6.3f} "
              f"{inv if inv is None else round(inv, 1):>14}")

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps({"prevalencia": float(y.mean()),
                                 "ano_inicial_teste": ANO_INICIAL_TESTE,
                                 "resultados": resultados},
                                indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saida] {SAIDA.relative_to(cfg.RAIZ)}")


if __name__ == "__main__":
    executar()
