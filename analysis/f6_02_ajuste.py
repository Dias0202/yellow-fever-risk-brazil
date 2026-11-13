"""Modelo de previsao, passo 2 - Por que o modelo perde para a regra sem aprendizado.

O passo 1 produziu um resultado que precisa ser explicado antes de qualquer outra
coisa: a forca de infeccao crua a 100 km, sem nenhum ajuste, da PR-AUC de 5,3x o
acaso, e o melhor modelo treinado da 4,5x. O aprendizado esta destruindo sinal.

Quatro suspeitas, todas testaveis, e este script as separa.

  A. PESO DE CLASSE. Com 1.254 eventos em 172.670 linhas, `balanced_subsample` e
     `scale_pos_weight` multiplicam o positivo por ~140. A arvore passa a dividir
     para isolar positivos, e com ruido ambiental abundante isso e memorizacao.
     O peso melhora a calibracao de um classificador; nao melhora ordenacao, que
     e o que PR-AUC e listas de prioridade medem.

  B. DILUICAO DE VARIAVEIS. Com `max_features="sqrt"` sobre 86 variaveis, cada
     divisao sorteia nove. As 21 de propagacao sao 24% do conjunto, de modo que
     boa parte das divisoes nao tem nenhuma delas disponivel. E a explicacao mais
     direta para o modelo completo perder para um dos seus proprios insumos.

  C. IMPUTACAO DE AUSENTE. `projecao_na_frente` e `avanco_da_frente` faltam em
     22,6% das linhas - as temporadas sem frente definida. O imputador por mediana
     inventa uma direcao de onda onde nao houve onda. Arvores impulsionadas tratam
     ausente nativamente e nao precisam disso.

  D. NAO ESTACIONARIEDADE. Treinar em tudo desde 1994 mistura o regime anterior a
     2016 com o da onda da Mata Atlantica. Ponderar por recencia pode ajudar.

Testa-se tambem uma forma funcional alternativa: regressao logistica sobre o
logaritmo das variaveis de propagacao. E a forma que a epidemiologia sugere - o
risco de invasao cresce com o logaritmo da pressao de infeccao - e nao tem como
sobreajustar do jeito que uma floresta tem.

Rodar:
    PYTHONPATH=src python analysis/f6_02_ajuste.py
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

from sklearn.ensemble import RandomForestClassifier            # noqa: E402
from sklearn.impute import SimpleImputer                        # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.pipeline import Pipeline                           # noqa: E402
from sklearn.preprocessing import (                             # noqa: E402
    FunctionTransformer, StandardScaler)
from xgboost import XGBClassifier                               # noqa: E402

from yfvrisk import config as cfg                               # noqa: E402
from yfvrisk.features import painel_temporada as pt             # noqa: E402
from yfvrisk.modeling import prospectiva as pv                  # noqa: E402

warnings.filterwarnings("ignore")

ALVO = "yf_event"
SAIDA = cfg.METRICAS / "f6_02_ajuste.json"
LOG = cfg.METRICAS / "f6_02_ajuste.log"

_log: list[str] = []


def diga(t=""):
    print(t)
    _log.append(t)


# --------------------------------------------------------------------------
# Conjunto compacto, escolhido pela biologia e nao por busca
# --------------------------------------------------------------------------
# A escolha e deliberada e declarada. Cada bloco corresponde a um mecanismo, e
# variaveis redundantes dentro de um bloco foram cortadas: manter as quatro
# escalas do nucleo euclidiano e as quatro do florestal duplicaria a mesma
# informacao e agravaria a diluicao que este script investiga.
PROPAGACAO = [
    "forca_k25", "forca_k50", "forca_k100", "forca_k200",
    "forca_flor_k50", "forca_flor_k100",
    "forca_epiz_k50", "forca_epiz_k100",
    "dist_foco_mais_proximo", "dist_foco_3temporadas",
    "dist_epizootia_mais_proxima",
    "focos_em_50km", "focos_em_100km", "focos_em_200km",
    "pressao_acumulada", "projecao_na_frente",
]
SUSCETIBILIDADE = [
    "temporadas_desde_evento", "eventos_locais_5t", "eventos_locais_10t",
    "evento_temporada_anterior", "epizootia_temporada_anterior",
]
# Adequabilidade estrutural: floresta e perda (Hansen, a fonte que ordena o
# territorio corretamente), relevo, populacao e a normal climatica das quatro
# variaveis com maior sustentacao na analise de explicabilidade (f4_01).
ESTRUTURAL = [
    "cobertura_florestal_t1", "perda_acumulada_km2_t1",
    "altitude_media", "amplitude_altimetrica",
    "populacao_log_t1", "area_km2",
    "ppt_sazonalidade_normal", "tmin_temporada_normal",
    "tmax_temporada_normal", "vpd_temporada_normal",
]
# Gatilho: quanto a temporada passada desviou do normal do proprio municipio.
GATILHO = [
    "ppt_pre_estacao_anomalia", "ppt_transmissao_anomalia",
    "tmax_transmissao_anomalia", "vpd_temporada_anomalia",
]

COMPACTO = PROPAGACAO + SUSCETIBILIDADE + ESTRUTURAL + GATILHO


def _peso_recencia(treino, meia_vida):
    """Peso que decai pela metade a cada `meia_vida` temporadas."""
    idade = treino["temporada"].max() - treino["temporada"]
    return np.power(0.5, idade / meia_vida).to_numpy()


def fabrica_xgb(*, peso_classe, meia_vida=None, profundidade=5, arvores=500):
    def ajustar(X, y, treino=None):
        pos = float(np.asarray(y).sum())
        spw = (len(y) - pos) / max(pos, 1) if peso_classe else 1.0
        m = XGBClassifier(
            n_estimators=arvores, learning_rate=0.04, max_depth=profundidade,
            min_child_weight=8, subsample=0.85, colsample_bytree=0.8,
            reg_lambda=2.0, scale_pos_weight=spw, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=42)
        w = None
        if meia_vida and treino is not None:
            w = _peso_recencia(treino, meia_vida)
        return m.fit(X, y, sample_weight=w)
    return ajustar


def fabrica_rf(*, peso_classe, max_features, folha=5, arvores=400):
    def ajustar(X, y, treino=None):
        m = Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("modelo", RandomForestClassifier(
                n_estimators=arvores, min_samples_leaf=folha,
                max_features=max_features,
                class_weight="balanced_subsample" if peso_classe else None,
                n_jobs=-1, random_state=42)),
        ])
        return m.fit(X, y)
    return ajustar


def fabrica_logistica(C=0.5):
    """Logistica sobre o log das variaveis de propagacao.

    O logaritmo e a transformacao que a epidemiologia sugere: o risco de invasao
    responde a ordens de grandeza da pressao de infeccao, nao a diferencas
    absolutas. `log1p` sobre as variaveis nao negativas; as demais entram como
    estao.

    A transformacao entra dentro do `Pipeline`, e nao antes dele, para que valha
    identicamente no ajuste e na predicao. Aplica-la so no ajuste seria um erro
    silencioso: o modelo aprenderia em escala logaritmica e prediria em escala
    linear.
    """
    nao_negativas = set(PROPAGACAO + SUSCETIBILIDADE) - {"projecao_na_frente"}
    transformador = FunctionTransformer(
        lambda X: _transforma(X, nao_negativas), feature_names_out="one-to-one")

    def ajustar(X, y, treino=None):
        m = Pipeline([
            ("log", transformador),
            ("imputar", SimpleImputer(strategy="median")),
            ("escala", StandardScaler()),
            ("modelo", LogisticRegression(C=C, max_iter=3000)),
        ])
        return m.fit(X, y)
    return ajustar


def _transforma(X, nao_negativas):
    Z = X.copy()
    for c in Z.columns:
        if c in nao_negativas:
            Z[c] = np.log1p(np.clip(Z[c].to_numpy(dtype=float), 0, None))
    return Z


def rodar(painel, colunas, ajustar, precisa_treino=False):
    """Como `prospectiva.rodar`, mas passando o quadro de treino ao ajustador
    quando ele precisa (peso por recencia) e transformando quando preciso."""
    testes = pv.temporadas_de_teste(painel)
    por_temporada, oof = [], []
    for t in testes:
        treino = painel[painel["temporada"] < t]
        teste = painel[painel["temporada"] == t]
        if treino[ALVO].sum() < 10 or teste[ALVO].sum() == 0:
            continue
        modelo = ajustar(treino[colunas], treino[ALVO], treino)
        proba = modelo.predict_proba(teste[colunas])[:, 1]
        m = pv.metricas(teste[ALVO], proba)
        m["temporada"] = int(t)
        por_temporada.append(m)
        oof.append(pd.DataFrame({
            "cod_mun": teste["cod_mun"].to_numpy(), "temporada": t,
            "regiao": teste["regiao"].to_numpy(),
            "y": teste[ALVO].to_numpy(), "proba": proba}))
    oof = pd.concat(oof, ignore_index=True)
    curva = pv.curva_de_decisao(oof["y"], oof["proba"])
    return {"por_temporada": por_temporada,
            "agregado": pv.metricas(oof["y"], oof["proba"]),
            "curva_de_decisao": curva,
            "limiares_uteis": pv.limiares_uteis(curva),
            "oof": oof}


def resumo(r):
    pt_ = [m for m in r["por_temporada"] if "pr_auc_sobre_prevalencia" in m]
    w = np.array([m["positivos"] for m in pt_], float)
    x = np.array([m["pr_auc_sobre_prevalencia"] for m in pt_], float)
    r500 = np.array([m.get("recall_top500", np.nan) for m in pt_], float)
    return {
        "pooled_vezes": r["agregado"]["pr_auc_sobre_prevalencia"],
        "media_pond_vezes": float(np.average(x, weights=w)),
        "roc": r["agregado"]["roc_auc"],
        "recall500_pond": float(np.average(r500, weights=w)),
        "limiares_uteis": r["limiares_uteis"],
    }


def linha(nome, s):
    return (f"{nome:38s} {s['roc']:.3f} {s['pooled_vezes']:7.2f} "
            f"{s['media_pond_vezes']:9.2f} {s['recall500_pond']:9.3f} "
            f"{s['limiares_uteis']:6d}")


def main():
    cfg.garantir_diretorios()
    inicio = time.time()
    painel = pt.carregar()
    grupos = pt.grupos()
    todas = [c for cols in grupos.values() for c in cols]

    faltando = [c for c in COMPACTO if c not in painel.columns]
    if faltando:
        raise SystemExit(f"variaveis ausentes do painel: {faltando}")

    diga(f"painel {len(painel):,} linhas | compacto {len(COMPACTO)} variaveis | "
         f"completo {len(todas)}")
    diga("")
    cab = f"{'configuracao':38s} ROC   pooled  med_pond  rec@500  limiares"
    resultados = {}

    def executa(nome, colunas, ajustar):
        r = rodar(painel, colunas, ajustar)
        s = resumo(r)
        resultados[nome] = {"resumo": s, "por_temporada": r["por_temporada"],
                            "n_variaveis": len(colunas)}
        diga(linha(nome, s))
        return r, s

    # Referencia: a regra sem aprendizado que o passo 1 nao conseguiu superar.
    diga("=" * 92)
    diga("REFERENCIA")
    diga("=" * 92)
    diga(cab)
    r = pv.rodar(painel, [], pv.BASES["vizinhanca"], alvo=ALVO, verboso=False)
    s = resumo(r)
    resultados["base_vizinhanca"] = {"resumo": s,
                                     "por_temporada": r["por_temporada"],
                                     "n_variaveis": 1}
    diga(linha("base: forca de infeccao 100 km", s))

    diga("")
    diga("=" * 92)
    diga("A. PESO DE CLASSE  |  B. DILUICAO DE VARIAVEIS")
    diga("=" * 92)
    diga(cab)
    for peso in (True, False):
        rot = "com peso" if peso else "sem peso"
        executa(f"xgb completo, {rot}", todas,
                fabrica_xgb(peso_classe=peso))
        executa(f"xgb compacto, {rot}", COMPACTO,
                fabrica_xgb(peso_classe=peso))
    for peso in (True, False):
        rot = "com peso" if peso else "sem peso"
        for mf in ("sqrt", 0.5):
            executa(f"rf compacto mf={mf}, {rot}", COMPACTO,
                    fabrica_rf(peso_classe=peso, max_features=mf))

    diga("")
    diga("=" * 92)
    diga("C. FORMA FUNCIONAL  |  D. PESO POR RECENCIA")
    diga("=" * 92)
    diga(cab)
    executa("logistica log-propagacao", COMPACTO, fabrica_logistica())
    for mv in (4, 8):
        executa(f"xgb compacto, meia-vida {mv} temporadas", COMPACTO,
                fabrica_xgb(peso_classe=False, meia_vida=mv))
    for prof in (3, 4):
        executa(f"xgb compacto sem peso, profundidade {prof}", COMPACTO,
                fabrica_xgb(peso_classe=False, profundidade=prof))

    melhor = max(resultados.items(),
                 key=lambda kv: kv[1]["resumo"]["media_pond_vezes"])
    diga("")
    diga(f"melhor por media ponderada: {melhor[0]} "
         f"({melhor[1]['resumo']['media_pond_vezes']:.2f}x)")

    SAIDA.write_text(json.dumps(
        {"compacto": COMPACTO, "resultados": resultados,
         "melhor": melhor[0], "segundos": round(time.time() - inicio, 1)},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    LOG.write_text("\n".join(_log), encoding="utf-8")


if __name__ == "__main__":
    main()
