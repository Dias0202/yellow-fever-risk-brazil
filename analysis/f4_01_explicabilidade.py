"""Explicabilidade - Determinantes ecologicos, medidos fora da amostra.

Nenhuma afirmacao sobre determinantes ecologicos deve vir de importancia de Gini
medida no treino. A comparacao aqui e direta: a mesma floresta aleatoria, o
mesmo painel, avaliada por MDI (no treino) e por permutacao fora da amostra.

A separacao treino/explicacao e temporal, nao aleatoria. Uma importancia medida
por reamostragem aleatoria herdaria a interpolacao temporal descrita em
`yfvrisk/modeling/validacao.py`: mediria o que o modelo memorizou, nao o que ele
generaliza.

O corte do treino e argumento de linha de comando, e o script roda duas vezes.
Ver o comentario de ANO_CORTE: um corte so nao distingue "a variavel nao carrega
sinal" de "este modelo nao aprendeu nada", e no corte de 2014 o modelo fica em
AUC 0,53 - praticamente sorteio. O que interessa e o que sobrevive aos dois.

Rodar:
    PYTHONPATH=src python \\
        analysis/f4_01_explicabilidade.py 2014
    PYTHONPATH=src python \\
        analysis/f4_01_explicabilidade.py 2019
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

from scipy.stats import spearmanr                            # noqa: E402
from sklearn.ensemble import RandomForestClassifier          # noqa: E402
from sklearn.impute import SimpleImputer                     # noqa: E402
from sklearn.pipeline import Pipeline                        # noqa: E402

from yfvrisk import config as cfg                            # noqa: E402
from yfvrisk.explain.importancia import (                    # noqa: E402
    curva_ale, importancia_por_grupo, importancia_por_permutacao, valores_shap,
)
from yfvrisk.features.painel import GRUPOS                   # noqa: E402
from yfvrisk.modeling.metricas import avaliar                # noqa: E402

warnings.filterwarnings("ignore")

ALVO = "yf_event"

# Ultimo ano do treino, passado na linha de comando. Sao dois cortes, e a
# comparacao entre eles e parte do resultado.
#
#   2014  pre-onda. O treino tem 319 positivos e a explicacao 997. E o teste
#         mais duro, e o modelo resultante fica em AUC 0,53 - praticamente
#         sorteio. Explicar so este corte seria explicar ruido: uma importancia
#         negativa aqui nao distingue "a variavel nao carrega sinal" de "este
#         modelo nao aprendeu nada".
#
#   2019  pos-onda. O treino tem 1.051 positivos, incluindo 2017 e 2018, e o
#         desempenho corresponde ao que a janela expansiva de f3_01 alcanca
#         nos anos de 2020 em diante. E o modelo que de fato seria implantado.
#
# As variaveis que aparecem nos dois cortes sao as unicas sobre as quais se
# pode afirmar determinacao ecologica: sobreviveram a mudanca de
# regime de 2016-2019.
ANO_CORTE = int(sys.argv[1]) if len(sys.argv) > 1 else 2014

SAIDA = cfg.METRICAS / f"f4_01_explicabilidade_corte{ANO_CORTE}.json"
SUFIXO = f"corte{ANO_CORTE}"

# Grupos usados. HISTORICO fica de fora: f3_01 mediu que ele acrescenta pouco
# (PR-AUC 0,029 -> 0,033) e ele nao existe no horizonte de projecao, de modo que
# uma explicacao construida sobre ele nao se transfere para o uso pretendido.
GRUPOS_USADOS = ["CLIMA", "CLIMA_DEFASADO", "PAISAGEM", "IMUNIDADE", "DEMOGRAFIA"]

N_REPETICOES_PERMUTACAO = 20
N_REPETICOES_GRUPO = 10
AMOSTRA_SHAP = 5000
N_VARIAVEIS_ALE = 6


def floresta_aleatoria():
    """Identica a de f3_01, para que a explicacao descreva aquele modelo."""
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=42)),
    ])


def montar_variaveis(painel):
    colunas = [c for g in GRUPOS_USADOS for c in GRUPOS[g] if c in painel.columns]
    faltando = [c for g in GRUPOS_USADOS for c in GRUPOS[g]
                if c not in painel.columns]
    if faltando:
        raise SystemExit(
            f"variaveis declaradas em GRUPOS e ausentes do painel: {faltando}")
    return colunas


def importancia_mdi(modelo, colunas) -> pd.DataFrame:
    """Importancia de Gini, calculada no treino. E o ranking convencional."""
    est = modelo.named_steps["modelo"]
    return (pd.DataFrame({"variavel": colunas, "mdi": est.feature_importances_})
            .sort_values("mdi", ascending=False).reset_index(drop=True))


def descrever_forma(curva) -> str:
    """Resume a curva ALE sem afirmar monotonicidade que ela nao tem.

    A primeira versao rotulava a curva de "crescente" ou "decrescente"
    comparando so o primeiro ponto com o ultimo. E enganoso: em tmax_medio o
    ultimo intervalo tem poucos municipios e um repique isolado invertia o
    rotulo de uma curva que decresce em todo o resto do dominio. Aqui a
    tendencia vem da correlacao de posto sobre a curva inteira, e o pico e
    reportado a parte, para que uma resposta unimodal - que e o caso de
    ppt_sazonalidade - apareca como unimodal e nao como monotona.
    """
    e = curva["efeito_ale"].to_numpy()
    amplitude = float(e.max() - e.min())
    rho = float(spearmanr(curva["valor"], e).statistic)

    # Unimodalidade antes de monotonicidade. Uma resposta em corcova com o ramo
    # descendente mais longo produz rho alto e seria rotulada de monotona, o que
    # apaga justamente o optimo ecologico que interessa - o caso de
    # ppt_sazonalidade. So conta como corcova se os dois ramos forem
    # substantivos em relacao a amplitude.
    i = int(e.argmax())
    subida, descida = e[i] - e[0], e[i] - e[-1]
    if 0 < i < len(e) - 1 and min(subida, descida) > 0.25 * amplitude:
        tendencia = f"unimodal, pico em {curva['valor'].iloc[i]:.3g}"
    elif abs(rho) >= 0.7:
        tendencia = "monotona crescente" if rho > 0 else "monotona decrescente"
    else:
        tendencia = "tendencia fraca"
    return f"amplitude {amplitude:.4f}  rho {rho:+.3f}  {tendencia}"


def executar():
    t_inicio = time.time()
    painel = pd.read_parquet(cfg.PROCESSADO / "painel_analitico.parquet")
    painel = painel.sort_values(["cod_mun", "ano"]).reset_index(drop=True)

    colunas = montar_variaveis(painel)
    X = painel[colunas]
    y = painel[ALVO].to_numpy()
    anos = painel["ano"].to_numpy()

    treino = np.where(anos <= ANO_CORTE)[0]
    explicar = np.where(anos > ANO_CORTE)[0]

    print("=" * 92)
    print("EXPLICABILIDADE - Determinantes ecologicos fora da amostra")
    print("=" * 92)
    print(f"\npainel      {len(painel):,} linhas, {len(colunas)} variaveis")
    print(f"treino      {int(anos[treino].min())}-{ANO_CORTE}, "
          f"{len(treino):,} linhas, {int(y[treino].sum())} positivos")
    print(f"explicacao  {ANO_CORTE + 1}-{int(anos.max())}, "
          f"{len(explicar):,} linhas, {int(y[explicar].sum())} positivos")

    print("\n[1] Ajustando a floresta aleatoria no treino...")
    t0 = time.time()
    modelo = floresta_aleatoria().fit(X.iloc[treino], y[treino])
    p_explicar = modelo.predict_proba(X.iloc[explicar])[:, 1]
    desempenho = avaliar(y[explicar], p_explicar)
    print(f"    {time.time() - t0:.0f}s | fora da amostra: "
          f"AUC {desempenho['roc_auc']:.4f}  PR-AUC {desempenho['pr_auc']:.4f}  "
          f"prec {desempenho['precisao']:.3f}")

    # ------------------------------------------------------ MDI (no treino)
    print("\n[2] Importancia de Gini (MDI), no treino")
    mdi = importancia_mdi(modelo, colunas)
    print(f"    amplitude {mdi['mdi'].min():.4f} a {mdi['mdi'].max():.4f} "
          f"(razao {mdi['mdi'].max() / max(mdi['mdi'].min(), 1e-12):.1f}x)")
    for _, r in mdi.head(8).iterrows():
        print(f"      {r['variavel']:32s} {r['mdi']:.4f}")

    # --------------------------------------- permutacao fora da amostra
    print(f"\n[3] Importancia por permutacao em {ANO_CORTE + 1}-{int(anos.max())}, "
          f"metrica PR-AUC, {N_REPETICOES_PERMUTACAO} repeticoes")
    t0 = time.time()
    perm = importancia_por_permutacao(
        modelo, X.iloc[explicar], y[explicar],
        n_repeticoes=N_REPETICOES_PERMUTACAO)
    print(f"    {time.time() - t0:.0f}s")
    for _, r in perm.head(10).iterrows():
        marca = " " if r["ic_inferior"] > 0 else " *"
        print(f"      {r['variavel']:32s} {r['importancia_media']:+.5f}  "
              f"[{r['ic_inferior']:+.5f}, {r['ic_superior']:+.5f}]{marca}")

    nao_distinguiveis = perm[perm["ic_inferior"] <= 0]
    negativas = perm[perm["importancia_media"] < 0]
    print(f"\n    variaveis cujo IC 95% inclui zero: {len(nao_distinguiveis)}"
          f" de {len(perm)}")
    print(f"    variaveis com importancia media negativa: {len(negativas)}")
    if len(negativas):
        print("      " + ", ".join(negativas["variavel"].head(8)))

    # ------------------------------------------- permutacao no treino
    # O ponto 3 da critica ao MDI: uma floresta totalmente crescida atribui
    # importancia a ruido que memorizou. Medir a mesma coisa dentro do treino
    # mostra o tamanho desse efeito.
    print("\n[4] A mesma permutacao dentro do treino, para dimensionar o vies")
    t0 = time.time()
    perm_treino = importancia_por_permutacao(
        modelo, X.iloc[treino], y[treino], n_repeticoes=5)
    print(f"    {time.time() - t0:.0f}s")

    # ------------------------------------------------- concordancia de ordem
    juntas = (mdi.merge(perm[["variavel", "importancia_media"]], on="variavel")
                 .merge(perm_treino[["variavel", "importancia_media"]],
                        on="variavel", suffixes=("_fora", "_treino")))
    rho_fora = spearmanr(juntas["mdi"], juntas["importancia_media_fora"])
    rho_treino = spearmanr(juntas["mdi"], juntas["importancia_media_treino"])
    print(f"\n    Spearman MDI x permutacao fora da amostra: "
          f"rho {rho_fora.statistic:+.3f} (p {rho_fora.pvalue:.3g})")
    print(f"    Spearman MDI x permutacao no treino:       "
          f"rho {rho_treino.statistic:+.3f} (p {rho_treino.pvalue:.3g})")

    juntas["posto_mdi"] = juntas["mdi"].rank(ascending=False)
    juntas["posto_perm"] = juntas["importancia_media_fora"].rank(ascending=False)
    juntas["deslocamento"] = juntas["posto_mdi"] - juntas["posto_perm"]
    piores = juntas.reindex(juntas["deslocamento"].abs()
                            .sort_values(ascending=False).index).head(6)
    print("\n    maiores discordancias de posto (MDI - permutacao):")
    for _, r in piores.iterrows():
        print(f"      {r['variavel']:32s} MDI #{int(r['posto_mdi']):2d}  "
              f"perm #{int(r['posto_perm']):2d}  ({r['deslocamento']:+.0f})")

    # -------------------------------------------------- importancia por grupo
    print(f"\n[5] Importancia por familia de variaveis, "
          f"{N_REPETICOES_GRUPO} repeticoes")
    grupos = {g: GRUPOS[g] for g in GRUPOS_USADOS}
    t0 = time.time()
    por_grupo = importancia_por_grupo(
        modelo, X.iloc[explicar], y[explicar], grupos,
        n_repeticoes=N_REPETICOES_GRUPO)
    print(f"    {time.time() - t0:.0f}s")
    for _, r in por_grupo.iterrows():
        print(f"      {r['grupo']:18s} {int(r['n_variaveis']):2d} var  "
              f"queda PR-AUC {r['queda_media_pr_auc']:+.5f} "
              f"+-{r['queda_desvio']:.5f}  ({r['queda_relativa']:+.1%})")

    # ------------------------------------------------------------------ SHAP
    print(f"\n[6] TreeSHAP sobre {AMOSTRA_SHAP} linhas de "
          f"{ANO_CORTE + 1}-{int(anos.max())}")
    t0 = time.time()
    try:
        _, _, shap_resumo = valores_shap(modelo, X.iloc[explicar],
                                         amostra=AMOSTRA_SHAP)
        print(f"    {time.time() - t0:.0f}s")
        for _, r in shap_resumo.head(8).iterrows():
            print(f"      {r['variavel']:32s} {r['shap_medio_absoluto']:.5f}")
        rho_shap = spearmanr(
            shap_resumo.set_index("variavel").loc[perm["variavel"],
                                                  "shap_medio_absoluto"],
            perm["importancia_media"])
        print(f"\n    Spearman SHAP x permutacao fora da amostra: "
              f"rho {rho_shap.statistic:+.3f}")
    except Exception as e:                                   # noqa: BLE001
        print(f"    SHAP indisponivel: {type(e).__name__}: {e}")
        shap_resumo, rho_shap = pd.DataFrame(), None

    # ------------------------------------------------------------------- ALE
    print(f"\n[7] Curvas ALE das {N_VARIAVEIS_ALE} variaveis mais importantes")
    principais = perm.head(N_VARIAVEIS_ALE)["variavel"].tolist()
    ales = []
    for v in principais:
        t0 = time.time()
        curva = curva_ale(modelo, X.iloc[explicar], v)
        if curva.empty:
            print(f"    {v:32s} sem intervalos suficientes")
            continue
        ales.append(curva)
        print(f"    {v:32s} {descrever_forma(curva)}  "
              f"({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- saidas
    cfg.garantir_diretorios()
    mdi.to_csv(cfg.TABELAS / f"f4_01_importancia_mdi_{SUFIXO}.csv",
               index=False, encoding="utf-8")
    perm.to_csv(cfg.TABELAS / f"f4_01_importancia_permutacao_{SUFIXO}.csv",
                index=False, encoding="utf-8")
    perm_treino.to_csv(cfg.TABELAS / f"f4_01_importancia_permutacao_treino_{SUFIXO}.csv",
                       index=False, encoding="utf-8")
    por_grupo.to_csv(cfg.TABELAS / f"f4_01_importancia_por_grupo_{SUFIXO}.csv",
                     index=False, encoding="utf-8")
    if not shap_resumo.empty:
        shap_resumo.to_csv(cfg.TABELAS / f"f4_01_shap_resumo_{SUFIXO}.csv",
                           index=False, encoding="utf-8")
    if ales:
        pd.concat(ales).to_csv(cfg.TABELAS / f"f4_01_curvas_ale_{SUFIXO}.csv",
                               index=False, encoding="utf-8")

    SAIDA.write_text(json.dumps({
        "ano_corte": ANO_CORTE,
        "n_variaveis": len(colunas),
        "variaveis": colunas,
        "grupos_usados": GRUPOS_USADOS,
        "n_treino": int(len(treino)),
        "positivos_treino": int(y[treino].sum()),
        "n_explicacao": int(len(explicar)),
        "positivos_explicacao": int(y[explicar].sum()),
        "desempenho_fora_da_amostra": desempenho,
        "mdi_amplitude": [float(mdi["mdi"].min()), float(mdi["mdi"].max())],
        "spearman_mdi_permutacao_fora": float(rho_fora.statistic),
        "spearman_mdi_permutacao_fora_p": float(rho_fora.pvalue),
        "spearman_mdi_permutacao_treino": float(rho_treino.statistic),
        "spearman_shap_permutacao": (float(rho_shap.statistic)
                                     if rho_shap is not None else None),
        "n_ic_inclui_zero": int(len(nao_distinguiveis)),
        "n_importancia_negativa": int(len(negativas)),
        "variaveis_negativas": negativas["variavel"].tolist(),
        "importancia_permutacao": perm.to_dict("records"),
        "importancia_mdi": mdi.to_dict("records"),
        "importancia_por_grupo": por_grupo.to_dict("records"),
        "shap_resumo": (shap_resumo.to_dict("records")
                        if not shap_resumo.empty else []),
        "segundos_total": round(time.time() - t_inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 92}")
    print(f"metricas em {SAIDA}")
    print(f"tabelas em  {cfg.TABELAS}")
    print(f"total: {(time.time() - t_inicio) / 60:.1f} min")


if __name__ == "__main__":
    executar()
