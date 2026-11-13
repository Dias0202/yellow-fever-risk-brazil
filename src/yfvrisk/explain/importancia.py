"""Explicabilidade: importancia por permutacao, SHAP e curvas ALE.

Usa permutacao fora da amostra, e nao importancia de Gini (MDI). O problema com
MDI neste caso especifico nao e teorico, e concreto:

  1. E enviesada para variaveis continuas de alta cardinalidade. Area municipal e
     densidade populacional aparecem alto por serem continuas e finamente
     divididas, nao por carregarem sinal.

  2. Fica fragmentada entre variaveis colineares. Com tres transformacoes
     deterministicas da mesma cobertura vacinal no modelo, a importancia da
     vacinacao se divide em tres e cada parte parece menor do que e.

  3. E calculada no treino, nao em dado nao visto. Uma floresta totalmente
     crescida atribui importancia a ruido que memorizou.

  4. Nao vem com incerteza. Uma distribuicao quase plana de MDI, sem barras de
     erro, tem ordenacao que nao e robusta.

Importancia por permutacao NEGATIVA em dado nao visto significa que embaralhar a
variavel MELHORA o desempenho, ou seja, ela nao carrega sinal generalizavel - e
o MDI nao tem como mostrar isso.

Aqui a importancia e calculada em dado nao visto, com intervalo por reamostragem,
e agrupada por familia de variaveis para nao dividir o credito entre colineares.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from yfvrisk import config as cfg


def importancia_por_permutacao(modelo, X, y, n_repeticoes: int = 20,
                               semente: int = 42, metrica=None) -> pd.DataFrame:
    """Importancia por permutacao em dado nao visto, com dispersao.

    A metrica padrao e a PR-AUC, nao a acuracia nem o ROC-AUC: com prevalencia de
    0,74% ao ano, acuracia e insensivel e o ROC-AUC e otimista. O que interessa e
    quanto a variavel contribui para ordenar os poucos positivos no topo.
    """
    def marcador(est, Xv, yv):
        return average_precision_score(yv, est.predict_proba(Xv)[:, 1])

    resultado = permutation_importance(
        modelo, X, y, scoring=metrica or marcador,
        n_repeats=n_repeticoes, random_state=semente, n_jobs=-1)

    return (pd.DataFrame({
        "variavel": list(X.columns),
        "importancia_media": resultado.importances_mean,
        "importancia_desvio": resultado.importances_std,
        "ic_inferior": np.percentile(resultado.importances, 2.5, axis=1),
        "ic_superior": np.percentile(resultado.importances, 97.5, axis=1),
    }).sort_values("importancia_media", ascending=False).reset_index(drop=True))


def importancia_por_grupo(modelo, X, y, grupos: dict[str, list[str]],
                          n_repeticoes: int = 10, semente: int = 42) -> pd.DataFrame:
    """Permuta todas as variaveis de um grupo em conjunto.

    Resolve o problema da colinearidade: permutar vac_coverage sozinha nao reduz o
    desempenho se not_vaccinated_pct continua no lugar, porque uma e funcao da
    outra. Permutando o grupo inteiro mede-se a contribuicao real da familia.
    """
    rng = np.random.default_rng(semente)
    base = average_precision_score(y, modelo.predict_proba(X)[:, 1])

    linhas = []
    for nome, colunas in grupos.items():
        presentes = [c for c in colunas if c in X.columns]
        if not presentes:
            continue
        quedas = []
        for _ in range(n_repeticoes):
            Xp = X.copy()
            ordem = rng.permutation(len(Xp))
            Xp[presentes] = Xp[presentes].to_numpy()[ordem]
            quedas.append(base - average_precision_score(
                y, modelo.predict_proba(Xp)[:, 1]))
        linhas.append({
            "grupo": nome,
            "n_variaveis": len(presentes),
            "queda_media_pr_auc": float(np.mean(quedas)),
            "queda_desvio": float(np.std(quedas)),
            "queda_relativa": float(np.mean(quedas) / base) if base > 0 else None,
        })
    return (pd.DataFrame(linhas)
            .sort_values("queda_media_pr_auc", ascending=False)
            .reset_index(drop=True))


def valores_shap(modelo, X, amostra: int = 5000, semente: int = 42):
    """TreeSHAP sobre uma amostra, devolvendo valores e importancia media.

    Amostra porque TreeSHAP e caro e 178 mil linhas nao acrescentam nada a leitura
    em relacao a alguns milhares bem distribuidos.
    """
    import shap

    if len(X) > amostra:
        X = X.sample(amostra, random_state=semente)

    estimador = modelo.named_steps["modelo"] if hasattr(modelo, "named_steps") else modelo
    Xt = X
    if hasattr(modelo, "named_steps"):
        for nome, passo in modelo.named_steps.items():
            if nome != "modelo":
                Xt = passo.transform(Xt)
        Xt = pd.DataFrame(Xt, columns=X.columns, index=X.index)

    explicador = shap.TreeExplainer(estimador)
    valores = explicador.shap_values(Xt, check_additivity=False)
    if isinstance(valores, list):
        valores = valores[1]
    elif valores.ndim == 3:
        valores = valores[:, :, 1]

    resumo = (pd.DataFrame({
        "variavel": list(X.columns),
        "shap_medio_absoluto": np.abs(valores).mean(axis=0),
    }).sort_values("shap_medio_absoluto", ascending=False).reset_index(drop=True))
    return valores, Xt, resumo


def curva_ale(modelo, X, variavel: str, n_intervalos: int = 20) -> pd.DataFrame:
    """Efeito local acumulado (ALE) para uma variavel.

    ALE em vez de dependencia parcial (PDP) porque as variaveis sao correlacionadas
    - chuva com deficit de pressao de vapor, floresta com pastagem. O PDP calcula a
    predicao em combinacoes que nao existem na natureza, como cobertura florestal
    de 80% junto com pastagem de 60%, e o resultado nao descreve nenhum municipio
    real. O ALE so usa variacao local dentro de cada intervalo.
    """
    valores = X[variavel].dropna()
    bordas = np.unique(np.quantile(valores, np.linspace(0, 1, n_intervalos + 1)))
    if len(bordas) < 3:
        return pd.DataFrame()

    efeitos, centros, contagens = [], [], []
    for i in range(len(bordas) - 1):
        sel = (X[variavel] >= bordas[i]) & (X[variavel] <= bordas[i + 1])
        if sel.sum() < 10:
            continue
        Xb = X[sel].copy()
        Xi, Xs = Xb.copy(), Xb.copy()
        Xi[variavel] = bordas[i]
        Xs[variavel] = bordas[i + 1]
        delta = (modelo.predict_proba(Xs)[:, 1] - modelo.predict_proba(Xi)[:, 1])
        efeitos.append(float(np.mean(delta)))
        centros.append(float((bordas[i] + bordas[i + 1]) / 2))
        contagens.append(int(sel.sum()))

    acumulado = np.cumsum(efeitos)
    acumulado = acumulado - np.average(acumulado, weights=contagens)
    return pd.DataFrame({"variavel": variavel, "valor": centros,
                         "efeito_ale": acumulado, "n": contagens})


def salvar(nome: str, tabela: pd.DataFrame, metadados: dict | None = None) -> None:
    cfg.garantir_diretorios()
    tabela.to_csv(cfg.TABELAS / f"{nome}.csv", index=False, encoding="utf-8")
    if metadados:
        (cfg.METRICAS / f"{nome}.json").write_text(
            json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8")
