"""Ablacao da vacinacao - Qual das escolhas na variavel move o desempenho.

O diagnostico (f5_00) leva a quatro mudancas simultaneas: defasar a imunidade em
dois anos, acrescentar o contraste transversal, acrescentar os anos de registro
no PNI e retirar `suscetivel_pct`, que e colinear. Com elas, o corte de 2014 vai
de ROC-AUC 0,5318 para 0,5006 - exatamente o acaso.

A leitura natural e que o pouco que aquele modelo tinha acima do acaso vinha da
imunidade contemporanea, que o estudo de evento mostra estar a jusante do
desfecho. Mas quatro mudancas simultaneas nao permitem atribuir a
queda a nenhuma delas. Este script separa.

Sao ajustes e avaliacoes, sem permutacao, SHAP nem ALE: dezenas de segundos por
variante, contra vinte minutos da explicabilidade completa. O que se quer aqui e
so o desempenho fora da amostra de cada conjunto de variaveis.

Variantes
---------
  modelo_final                 o painel de modelagem: so `imunidade_defasada2`
  sem_imunidade                nenhuma variavel de imunidade
  imunidade_a_jusante          a CONTEMPORANEA e o seu complemento colinear
  com_construtos_de_registro   a defasada mais `imunidade_relativa2` e
                               `anos_com_registro_pni`, versao intermediaria
                               rejeitada por admissibilidade

A comparacao que interessa e `sem_imunidade` contra `imunidade_a_jusante`. Se a
segunda for melhor que a primeira, a imunidade contemporanea estava de fato
carregando o desempenho - e carregando-o por vazamento, porque a variavel e
consequencia do surto que ela aparenta prever.

Sobre a incerteza destas diferencas
-----------------------------------
Com 265 positivos no teste do corte de 2019, uma diferenca de 0,02 em ROC-AUC
entre duas variantes pode ser inteiramente ruido de amostra. Comparar quatro
numeros de ponto sem intervalo nao sustenta nenhuma conclusao.

Cada diferenca em relacao a `sem_imunidade` vem, portanto, com intervalo por
reamostragem PAREADA: a mesma reamostragem do conjunto de teste e aplicada as
duas variantes, e o que se reamostra sao MUNICIPIOS, nao linhas. Pareado porque
o que interessa e a diferenca, e boa parte da variabilidade e comum as duas
variantes e cancela; por municipio porque as linhas de um mesmo municipio nao
sao independentes.

Rodar:
    PYTHONPATH=src python \\
        analysis/f5_01_ablacao_vacinacao.py
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

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from yfvrisk import config as cfg                            # noqa: E402
from yfvrisk.features.painel import GRUPOS                   # noqa: E402
from yfvrisk.modeling.metricas import avaliar                # noqa: E402
from yfvrisk.scenarios.projetar import floresta_aleatoria    # noqa: E402

warnings.filterwarnings("ignore")

ALVO = "yf_event"
CORTES = [2014, 2019]
GRUPOS_USADOS = ["CLIMA", "CLIMA_DEFASADO", "PAISAGEM", "IMUNIDADE", "DEMOGRAFIA"]

REFERENCIA = "sem_imunidade"
N_REAMOSTRAGENS = 500

SAIDA = cfg.METRICAS / "f5_01_ablacao_vacinacao.json"
LOG = cfg.METRICAS / "f5_01_ablacao_vacinacao.log"

_linhas_log: list[str] = []


def diga(texto: str = "") -> None:
    print(texto)
    _linhas_log.append(texto)


def variantes(painel: pd.DataFrame) -> dict[str, list[str]]:
    """As quatro variantes, com as colunas de imunidade declaradas uma a uma.

    Declaradas, e nao derivadas de `GRUPOS["IMUNIDADE"]`, de proposito: o que
    este script compara sao decisoes sobre esse grupo, e uma variante que mudasse
    junto com a decisao nao poderia testa-la. Foi o que aconteceu na primeira
    versao, em que `modelo_final` seguia GRUPOS e ficou identica a variante que
    deveria servir de contraste.
    """
    base = [c for g in GRUPOS_USADOS if g != "IMUNIDADE"
            for c in GRUPOS[g] if c in painel.columns]
    return {
        # O que o painel de modelagem usa.
        "modelo_final": base + ["imunidade_defasada2"],
        # Nenhuma imunidade, para medir o que o grupo inteiro acrescenta.
        "sem_imunidade": base,
        # A imunidade contemporanea e o seu complemento colinear: a construcao
        # ingenua da variavel.
        "imunidade_a_jusante": base + ["imunidade_acumulada", "suscetivel_pct"],
        # A versao intermediaria, com os construtos sobre o processo de registro.
        # Rejeitada por admissibilidade - sob projecao os dois teriam de ser
        # congelados e virariam codigo de identidade do municipio -, e esta aqui
        # para que a rejeicao tenha numero e nao so argumento.
        "com_construtos_de_registro": base + ["imunidade_defasada2",
                                              "imunidade_relativa2",
                                              "anos_com_registro_pni"],
    }


def diferencas_com_intervalo(y, proba_por_variante, municipios,
                             n=N_REAMOSTRAGENS, semente=42) -> list[dict]:
    """IC 95% da diferenca de cada variante contra a referencia, pareada.

    A mesma reamostragem de municipios e aplicada a todas as variantes, de modo
    que a variabilidade comum a elas cancele e o intervalo descreva so a
    diferenca.
    """
    rng = np.random.default_rng(semente)
    codigos = np.unique(municipios)
    indices = {c: np.where(municipios == c)[0] for c in codigos}
    nomes = [k for k in proba_por_variante if k != REFERENCIA]

    acumulado = {nome: {"roc": [], "pr": []} for nome in nomes}
    for _ in range(n):
        escolhidos = rng.choice(codigos, size=len(codigos), replace=True)
        linhas = np.concatenate([indices[c] for c in escolhidos])
        yb = y[linhas]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ref_roc = roc_auc_score(yb, proba_por_variante[REFERENCIA][linhas])
        ref_pr = average_precision_score(yb, proba_por_variante[REFERENCIA][linhas])
        for nome in nomes:
            p = proba_por_variante[nome][linhas]
            acumulado[nome]["roc"].append(roc_auc_score(yb, p) - ref_roc)
            acumulado[nome]["pr"].append(average_precision_score(yb, p) - ref_pr)

    saida = []
    for nome in nomes:
        d = acumulado[nome]
        saida.append({
            "variante": nome, "referencia": REFERENCIA,
            "delta_roc_auc": float(np.mean(d["roc"])),
            "ic95_roc_auc": [float(np.percentile(d["roc"], 2.5)),
                             float(np.percentile(d["roc"], 97.5))],
            "delta_pr_auc": float(np.mean(d["pr"])),
            "ic95_pr_auc": [float(np.percentile(d["pr"], 2.5)),
                            float(np.percentile(d["pr"], 97.5))],
        })
    return saida


def main() -> None:
    cfg.garantir_diretorios()
    inicio = time.time()
    painel = pd.read_parquet(cfg.PROCESSADO / "painel_analitico.parquet")
    conjuntos = variantes(painel)

    diga(f"painel: {len(painel):,} linhas, {int(painel[ALVO].sum()):,} eventos")
    diga("")

    resultados, intervalos = [], {}
    for corte in CORTES:
        treino = painel[painel["ano"] <= corte]
        teste = painel[painel["ano"] > corte]
        diga("=" * 74)
        diga(f"corte {corte}: treino {len(treino):,} linhas "
             f"({int(treino[ALVO].sum())} eventos), "
             f"teste {len(teste):,} ({int(teste[ALVO].sum())} eventos)")
        diga("=" * 74)
        diga("variante              n_var   ROC-AUC   PR-AUC   PR-AUC/prevalencia")

        probas = {}
        for nome, colunas in conjuntos.items():
            modelo = floresta_aleatoria(42)
            modelo.fit(treino[colunas].to_numpy(float), treino[ALVO].to_numpy(int))
            proba = modelo.predict_proba(teste[colunas].to_numpy(float))[:, 1]
            probas[nome] = proba
            m = avaliar(teste[ALVO].to_numpy(int), proba)

            resultados.append({"corte": corte, "variante": nome,
                               "n_variaveis": len(colunas),
                               "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                               "pr_auc_sobre_prevalencia":
                                   m["pr_auc_sobre_prevalencia"]})
            diga(f"{nome:20s} {len(colunas):5d}   {m['roc_auc']:.4f}   "
                 f"{m['pr_auc']:.4f}   {m['pr_auc_sobre_prevalencia']:8.3f}")

        diga("")
        diga(f"diferencas contra `{REFERENCIA}`, IC 95% por reamostragem pareada "
             f"de municipios:")
        diga("variante              d ROC-AUC  IC 95%              "
             "d PR-AUC   IC 95%")
        intervalos[corte] = diferencas_com_intervalo(
            teste[ALVO].to_numpy(int), probas, teste["cod_mun"].to_numpy())
        for r in intervalos[corte]:
            marca = "" if (r["ic95_pr_auc"][0] < 0 < r["ic95_pr_auc"][1]) else "  *"
            diga(f"{r['variante']:20s} {r['delta_roc_auc']:+.4f}   "
                 f"[{r['ic95_roc_auc'][0]:+.4f}, {r['ic95_roc_auc'][1]:+.4f}]   "
                 f"{r['delta_pr_auc']:+.4f}  "
                 f"[{r['ic95_pr_auc'][0]:+.4f}, {r['ic95_pr_auc'][1]:+.4f}]{marca}")
        diga("  * = IC da diferenca em PR-AUC nao inclui zero")
        diga("")

    tab = pd.DataFrame(resultados)
    diga("=" * 74)
    diga("O contraste que importa: a imunidade a jusante contra nenhuma imunidade")
    diga("=" * 74)
    for corte in CORTES:
        sub = tab[tab["corte"] == corte].set_index("variante")
        ganho = (sub.loc["imunidade_a_jusante", "roc_auc"]
                 - sub.loc["sem_imunidade", "roc_auc"])
        ganho_novo = (sub.loc["modelo_final", "roc_auc"]
                      - sub.loc["sem_imunidade", "roc_auc"])
        diga(f"corte {corte}: a contemporanea acrescenta {ganho:+.4f} de ROC-AUC; "
             f"a defasada, {ganho_novo:+.4f}")

    resultado = {"cortes": CORTES, "variantes": {k: v for k, v in conjuntos.items()},
                 "referencia": REFERENCIA, "n_reamostragens": N_REAMOSTRAGENS,
                 "resultados": resultados,
                 "diferencas_com_intervalo": intervalos,
                 "segundos_total": round(time.time() - inicio, 1)}
    SAIDA.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    LOG.write_text("\n".join(_linhas_log), encoding="utf-8")
    print(f"\nmetricas em {SAIDA}")


if __name__ == "__main__":
    main()
