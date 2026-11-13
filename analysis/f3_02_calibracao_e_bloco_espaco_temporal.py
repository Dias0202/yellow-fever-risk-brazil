"""Validacao em bloco - Calibracao, utilidade de decisao e o teste mais severo.

Tres medicoes que complementam `f3_01_comparar_validacao.py`.

1. Calibracao. Um mapa de risco pinta uma rampa de cor a partir da probabilidade
   prevista. Sem curva de calibracao, essa rampa nao significa nada: a cor
   rotulada como risco 0,3 pode corresponder a 3% ou a 60% de eventos reais. A
   floresta aleatoria com class_weight balanceado e conhecidamente descalibrada
   para cima em eventos raros, porque o reponderamento afasta as probabilidades
   da prevalencia. Aqui isso e medido, e nao suposto.

2. Curva de decisao. Traduz a predicao na pergunta que o gestor faz de fato:
   agir sobre os municipios acima deste limiar rende mais do que nao agir sobre
   nenhum, ou do que agir sobre todos? Um modelo pode discriminar bem e nao ter
   nenhuma faixa de limiar em que agir compense.

3. Bloco espaco-temporal. Nem a regiao nem o ano de teste aparecem no treino. E
   o esquema que corresponde a projetar o risco para uma regiao inteira, em anos
   futuros - e, impondo as duas restricoes ao mesmo tempo,
   deve ficar abaixo dos 0,736 do temporal puro e dos 0,855 do espacial. Se nao
   ficar, e sinal de que uma das duas particoes nao isola o que deveria.

   A regiao aqui e a macrorregiao do IBGE, nao a UF. Sao 5 blocos em vez de 27,
   o que reduz o custo de 567 ajustes para 105 - mas a razao principal nao e
   custo: deixar uma UF de fora ainda mantem UFs vizinhas do mesmo bioma no
   treino, o que nao testa transferencia regional. Deixar o Sul inteiro de fora
   e o teste que corresponde a uma afirmacao de transferibilidade ecologica.

Tudo sobre predicoes fora da amostra, com intervalo por reamostragem: com 1.316
positivos em 178.240 linhas, a incerteza nao e desprezivel.

Rodar:
    PYTHONPATH=src python \\
        analysis/f3_02_calibracao_e_bloco_espaco_temporal.py
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
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline                        # noqa: E402

from yfvrisk import config as cfg                            # noqa: E402
from yfvrisk.features.painel import GRUPOS                   # noqa: E402
from yfvrisk.modeling.metricas import (                      # noqa: E402
    avaliar, calibracao, curva_de_decisao,
)
from yfvrisk.modeling.validacao import (                     # noqa: E402
    BlocoEspacoTemporal, JanelaTemporal,
)

warnings.filterwarnings("ignore")

SAIDA = cfg.METRICAS / "f3_02_calibracao_decisao_bloco.json"
ALVO = "yf_event"
ANO_INICIAL_TESTE = 2005
GRUPOS_USADOS = ["CLIMA", "CLIMA_DEFASADO", "PAISAGEM", "IMUNIDADE", "DEMOGRAFIA"]
N_REAMOSTRAGENS = 500

# Primeiro digito do codigo de UF do IBGE.
MACRORREGIOES = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul",
                 5: "Centro-Oeste"}


def floresta_aleatoria():
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=42)),
    ])


def intervalo_por_reamostragem(y, proba, metrica, n=N_REAMOSTRAGENS, semente=42):
    """IC 95% percentil por bootstrap estratificado pelo desfecho.

    Estratificado porque, com prevalencia de 0,74%, a reamostragem simples
    produz replicas sem nenhum positivo, em que a metrica e indefinida.
    """
    rng = np.random.default_rng(semente)
    y = np.asarray(y)
    proba = np.asarray(proba)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    valores = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        try:
            valores.append(float(metrica(y[idx], proba[idx])))
        except ValueError:
            continue
    if not valores:
        return None
    return {"mediana": float(np.median(valores)),
            "ic_inferior": float(np.percentile(valores, 2.5)),
            "ic_superior": float(np.percentile(valores, 97.5)),
            "n_replicas": len(valores)}


def predicoes_temporais(X, y, anos):
    oof = np.full(len(y), np.nan)
    dobras = []
    for treino, teste, info in JanelaTemporal(ANO_INICIAL_TESTE,
                                              minimo_treino=10).split(anos):
        m = floresta_aleatoria().fit(X.iloc[treino], y[treino])
        oof[teste] = m.predict_proba(X.iloc[teste])[:, 1]
        dobras.append({**info, "n_treino": int(len(treino)),
                       "n_teste": int(len(teste)),
                       "positivos_teste": int(y[teste].sum())})
    return oof, dobras


def predicoes_espaco_temporais(X, y, anos, regioes):
    """Um ajuste por (regiao excluida, ano de teste). Sao 5 x 21 = 105.

    Nao ha ajuste reaproveitavel entre dobras: o treino depende das duas
    restricoes ao mesmo tempo.
    """
    oof = np.full(len(y), np.nan)
    dobras = []
    cv = BlocoEspacoTemporal(ANO_INICIAL_TESTE, minimo_treino=10)
    total = 0
    for treino, teste, info in cv.split(anos, regioes):
        m = floresta_aleatoria().fit(X.iloc[treino], y[treino])
        oof[teste] = m.predict_proba(X.iloc[teste])[:, 1]
        dobras.append({**info, "n_treino": int(len(treino)),
                       "n_teste": int(len(teste)),
                       "positivos_teste": int(y[teste].sum())})
        total += 1
        if total % 20 == 0:
            print(f"    {total} dobras...", flush=True)
    return oof, dobras


def desempenho_por_regiao(y, oof, regioes):
    """Quanto o modelo transfere para cada macrorregiao nunca vista.

    Com IC por reamostragem. O resultado do Sul no bloco espaco-temporal e a
    afirmacao mais forte desta analise - a unica regiao que transfere sem
    estar no treino - e uma afirmacao dessas nao vai sem intervalo, ainda mais
    apoiada em 251 positivos.
    """
    linhas = []
    for reg in sorted(pd.unique(regioes)):
        sel = (regioes == reg) & ~np.isnan(oof)
        if sel.sum() == 0 or len(np.unique(y[sel])) < 2:
            continue
        yr, pr_ = y[sel], oof[sel]
        r = avaliar(yr, pr_)
        linhas.append({"regiao": str(reg), "n": int(sel.sum()),
                       "positivos": r["positivos"], "roc_auc": r["roc_auc"],
                       "pr_auc": r["pr_auc"], "prevalencia": r["prevalencia"],
                       "pr_auc_sobre_prevalencia": r["pr_auc_sobre_prevalencia"],
                       "ic_roc_auc": intervalo_por_reamostragem(
                           yr, pr_, roc_auc_score),
                       "ic_pr_auc": intervalo_por_reamostragem(
                           yr, pr_, average_precision_score)})
    return linhas


def imprimir_calibracao(faixas):
    print("\n  calibracao (decis de probabilidade prevista):")
    print(f"    {'faixa':>5s} {'n':>9s} {'prevista':>10s} {'observada':>10s} "
          f"{'razao':>8s}")
    for f in faixas:
        obs = f["frequencia_observada"]
        razao = (f"{f['probabilidade_media_prevista'] / obs:8.1f}"
                 if obs > 0 else f"{'-':>8s}")
        print(f"    {f['faixa']:5d} {f['n']:9,d} "
              f"{f['probabilidade_media_prevista']:10.4f} {obs:10.4f} {razao}")


def analisar(nome, y, oof, regioes):
    valido = ~np.isnan(oof)
    yv, pv = y[valido], oof[valido]
    print(f"  {valido.sum():,} predicoes fora da amostra, "
          f"{int(yv.sum())} positivos")

    res = avaliar(yv, pv)
    inv = res["municipios_investigados_por_evento"]
    linha = (f"  AUC {res['roc_auc']:.4f}  PR-AUC {res['pr_auc']:.4f}  "
             f"prec {res['precisao']:.3f}  rec {res['recall']:.3f}")
    if inv:
        linha += f"  investigar {inv:.1f}"
    print(linha)

    ic_auc = intervalo_por_reamostragem(yv, pv, roc_auc_score)
    ic_pr = intervalo_por_reamostragem(yv, pv, average_precision_score)
    print(f"  IC 95% por reamostragem ({N_REAMOSTRAGENS} replicas):")
    print(f"    ROC-AUC {ic_auc['mediana']:.4f} "
          f"[{ic_auc['ic_inferior']:.4f}, {ic_auc['ic_superior']:.4f}]")
    print(f"    PR-AUC  {ic_pr['mediana']:.4f} "
          f"[{ic_pr['ic_inferior']:.4f}, {ic_pr['ic_superior']:.4f}]")

    faixas = calibracao(yv, pv, n_faixas=10)
    imprimir_calibracao(faixas)
    if faixas:
        alta = faixas[-1]
        obs = alta["frequencia_observada"]
        print(f"\n  faixa superior: previsto "
              f"{alta['probabilidade_media_prevista']:.3f}, observado {obs:.3f}"
              f"  ({alta['eventos']} eventos em {alta['n']:,} municipios-ano)")
        if obs > 0:
            print(f"  a probabilidade prevista no decil de topo excede a "
                  f"frequencia real em {alta['probabilidade_media_prevista'] / obs:.0f}x")

    decisao = curva_de_decisao(yv, pv)
    util = [d for d in decisao
            if d["beneficio_liquido"] > max(d["beneficio_tratar_todos"], 0)]
    print(f"\n  curva de decisao: o modelo supera 'agir sobre todos' e 'agir "
          f"sobre nenhum' em {len(util)} de {len(decisao)} limiares")
    if util:
        melhor = max(util, key=lambda d: d["beneficio_liquido"])
        print(f"    melhor limiar {melhor['limiar']:.3f}: beneficio liquido "
              f"{melhor['beneficio_liquido']:.5f}, "
              f"{melhor['municipios_sinalizados']:,} municipios-ano sinalizados")
        print(f"    faixa util: {util[0]['limiar']:.3f} a {util[-1]['limiar']:.3f}")
    else:
        print("    NENHUM limiar em que agir sobre o alerta compense.")

    por_regiao = desempenho_por_regiao(y, oof, regioes)
    if por_regiao:
        print("\n  por macrorregiao:")
        for r in por_regiao:
            a_, p_ = r["ic_roc_auc"], r["ic_pr_auc"]
            print(f"    {r['regiao']:14s} "
                  f"AUC {r['roc_auc']:.3f} "
                  f"[{a_['ic_inferior']:.3f}, {a_['ic_superior']:.3f}]  "
                  f"PR-AUC {r['pr_auc']:.4f} "
                  f"[{p_['ic_inferior']:.4f}, {p_['ic_superior']:.4f}]  "
                  f"({r['positivos']} pos, "
                  f"{r['pr_auc_sobre_prevalencia']:.1f}x o acaso)")

    return {"metricas": res, "ic_roc_auc": ic_auc, "ic_pr_auc": ic_pr,
            "calibracao": faixas, "curva_de_decisao": decisao,
            "limiares_uteis": len(util), "por_regiao": por_regiao}


def executar():
    t_inicio = time.time()
    painel = pd.read_parquet(cfg.PROCESSADO / "painel_analitico.parquet")
    painel = painel.sort_values(["cod_mun", "ano"]).reset_index(drop=True)

    painel["regiao"] = (painel["cod_uf"].astype(int) // 10).map(MACRORREGIOES)
    assert painel["regiao"].notna().all(), "codigo de UF fora do esperado"

    colunas = [c for g in GRUPOS_USADOS for c in GRUPOS[g] if c in painel.columns]
    faltando = [c for g in GRUPOS_USADOS for c in GRUPOS[g]
                if c not in painel.columns]
    if faltando:
        raise SystemExit(f"variaveis ausentes do painel: {faltando}")

    X = painel[colunas]
    y = painel[ALVO].to_numpy()
    anos = painel["ano"].to_numpy()
    regioes = painel["regiao"].to_numpy()

    print("=" * 92)
    print("VALIDACAO EM BLOCO - Calibracao, decisao e bloco espaco-temporal")
    print("=" * 92)
    print(f"\npainel {len(painel):,} linhas, {len(colunas)} variaveis, "
          f"prevalencia {y.mean():.4%}")
    print(f"macrorregioes: {', '.join(sorted(pd.unique(regioes)))}")

    resultados = {}

    print(f"\n{'-' * 92}\n[B] Janela temporal expansiva (referencia de f3_01)")
    t0 = time.time()
    oof_t, dobras_t = predicoes_temporais(X, y, anos)
    print(f"  {time.time() - t0:.0f}s")
    resultados["temporal"] = analisar("temporal", y, oof_t, regioes)
    resultados["temporal"]["dobras"] = dobras_t
    resultados["temporal"]["segundos"] = round(time.time() - t0, 1)

    print(f"\n{'-' * 92}\n[F] Bloco espaco-temporal (nem a regiao nem o ano)")
    t0 = time.time()
    oof_et, dobras_et = predicoes_espaco_temporais(X, y, anos, regioes)
    print(f"  {time.time() - t0:.0f}s, {len(dobras_et)} dobras")
    resultados["espaco_temporal"] = analisar("espaco_temporal", y, oof_et, regioes)
    resultados["espaco_temporal"]["dobras"] = dobras_et
    resultados["espaco_temporal"]["segundos"] = round(time.time() - t0, 1)

    # As predicoes fora da amostra custam 22 minutos, dos quais 17 sao os 105
    # ajustes do bloco espaco-temporal. Salvas aqui para que qualquer analise
    # posterior - calibracao por regiao, mapa de erro, recorte por bioma - saia
    # delas em segundos, em vez de exigir a reexecucao inteira.
    predicoes = pd.DataFrame({
        "cod_mun": painel["cod_mun"], "ano": painel["ano"],
        "uf": painel["uf"], "regiao": painel["regiao"], "y": y,
        "proba_temporal": oof_t, "proba_espaco_temporal": oof_et,
    })
    caminho_pred = cfg.PROCESSADO / "f3_02_predicoes_oof.parquet"
    predicoes.to_parquet(caminho_pred, index=False)
    print(f"\npredicoes fora da amostra em {caminho_pred}")
    print(f"\n{'=' * 92}\nSINTESE\n{'=' * 92}")
    cab = (f"{'esquema':18s} {'AUC':>7s} {'IC 95% AUC':>18s} {'PR-AUC':>8s} "
           f"{'IC 95% PR-AUC':>20s} {'lim. uteis':>11s}")
    print(cab)
    for nome, r in resultados.items():
        a, p = r["ic_roc_auc"], r["ic_pr_auc"]
        ic_a = f"[{a['ic_inferior']:.3f}, {a['ic_superior']:.3f}]"
        ic_p = f"[{p['ic_inferior']:.4f}, {p['ic_superior']:.4f}]"
        print(f"{nome:18s} {r['metricas']['roc_auc']:7.4f} {ic_a:>18s} "
              f"{r['metricas']['pr_auc']:8.4f} {ic_p:>20s} "
              f"{r['limiares_uteis']:11d}")

    cfg.garantir_diretorios()
    SAIDA.write_text(json.dumps({
        "prevalencia": float(y.mean()),
        "n_variaveis": len(colunas),
        "variaveis": colunas,
        "ano_inicial_teste": ANO_INICIAL_TESTE,
        "n_reamostragens": N_REAMOSTRAGENS,
        "macrorregioes": sorted(pd.unique(regioes).tolist()),
        "resultados": resultados,
        "segundos_total": round(time.time() - t_inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmetricas em {SAIDA}")
    print(f"total: {(time.time() - t_inicio) / 60:.1f} min")


if __name__ == "__main__":
    executar()
