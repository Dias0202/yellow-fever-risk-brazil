"""Modelo de previsao, passo 1 - O modelo preve? Validacao prospectiva por temporada.

Compara, sob o mesmo esquema de janela expansiva, as tres linhas de base triviais
e quatro conjuntos de variaveis. A comparacao com as linhas de base e obrigatoria:
um modelo que nao supera "quem teve evento na temporada passada tera de novo" nao
tem valor operacional.

Os conjuntos, em ordem de acrescimo:

    ambiental      so clima, paisagem, relevo e demografia: adequabilidade
                   estatica de habitat, transposta para a temporada.
    suscetibilidade  acrescenta historico local e refratariedade
    propagacao     acrescenta forca de infeccao, distancia ao foco e frente
    completo       tudo

A separacao existe para atribuir o ganho. Se o salto estiver entre `ambiental`
e `propagacao`, esta demonstrado que o que falta a um modelo de adequabilidade de
habitat e a estrutura de contagio, e nao mais variaveis ambientais.

Rodar:
    PYTHONPATH=src python analysis/f6_01_previsao.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import RandomForestClassifier            # noqa: E402
from sklearn.impute import SimpleImputer                        # noqa: E402
from sklearn.pipeline import Pipeline                           # noqa: E402
from xgboost import XGBClassifier                               # noqa: E402

from yfvrisk import config as cfg                               # noqa: E402
from yfvrisk.features import painel_temporada as pt             # noqa: E402
from yfvrisk.modeling import prospectiva as pv                  # noqa: E402

warnings.filterwarnings("ignore")

ALVO = "yf_event"
SAIDA = cfg.METRICAS / "f6_01_previsao.json"
LOG = cfg.METRICAS / "f6_01_previsao.log"
OOF = cfg.PROCESSADO / "f6_01_predicoes_oof.parquet"

_log: list[str] = []


def diga(t=""):
    print(t)
    _log.append(t)


def floresta(X, y):
    """A mesma floresta das analises de validacao, para que a comparacao seja de
    variaveis e nao de algoritmo."""
    m = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
    ])
    return m.fit(X, y)


def gradiente(X, y):
    """Arvores impulsionadas. Lidam com ausentes sem imputacao, o que importa
    aqui: `dist_foco_mais_proximo` na primeira temporada e ausente por
    construcao, e imputar pela mediana inventaria um foco que nao existiu."""
    pos = float(y.sum())
    m = XGBClassifier(
        n_estimators=600, learning_rate=0.04, max_depth=5,
        min_child_weight=8, subsample=0.85, colsample_bytree=0.7,
        reg_lambda=2.0, scale_pos_weight=(len(y) - pos) / max(pos, 1),
        eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=42,
    )
    return m.fit(X, y)


def conjuntos(grupos: dict) -> dict[str, list[str]]:
    amb = (grupos["CLIMA_NORMAL"] + grupos["CLIMA_RECENTE"]
           + grupos["CLIMA_ANOMALIA"] + grupos["PAISAGEM"]
           + grupos["RELEVO"] + grupos["DEMOGRAFIA"])
    return {
        "ambiental": amb,
        "suscetibilidade": amb + grupos["SUSCETIBILIDADE"],
        "propagacao": amb + grupos["PROPAGACAO"],
        "completo": amb + grupos["SUSCETIBILIDADE"] + grupos["PROPAGACAO"],
        "so_propagacao": grupos["PROPAGACAO"] + grupos["SUSCETIBILIDADE"],
    }


def linha(nome, r):
    a = r["agregado"]
    return (f"{nome:28s} {a['roc_auc']:.3f}  {a['pr_auc']:.4f}  "
            f"{a['pr_auc_sobre_prevalencia']:6.1f}x  "
            f"{a.get('recall_top100', float('nan')):.3f}  "
            f"{a.get('investigados_por_surto_top100', float('nan')):6.1f}  "
            f"{r['limiares_uteis']:4d}")


def main():
    cfg.garantir_diretorios()
    inicio = time.time()

    painel = pt.carregar()
    grupos = pt.grupos()
    confs = conjuntos(grupos)

    diga(f"painel: {len(painel):,} linhas, {painel['cod_mun'].nunique():,} municipios, "
         f"temporadas {painel['temporada'].min()}-{painel['temporada'].max()}")
    diga(f"prevalencia de {ALVO}: {painel[ALVO].mean():.4%}")
    diga(f"temporadas previstas: {pv.temporadas_de_teste(painel)[0]}-"
         f"{pv.temporadas_de_teste(painel)[-1]}")
    diga("")

    resultados = {}

    diga("=" * 92)
    diga("LINHAS DE BASE (sem aprendizado)")
    diga("=" * 92)
    diga(f"{'':28s} ROC    PR-AUC   vezes   rec@100  inv/surto  limiares uteis")
    for nome, fn in pv.BASES.items():
        r = pv.rodar(painel, [], fn, alvo=ALVO, verboso=False)
        resultados[f"base_{nome}"] = r
        diga(linha(f"base: {nome}", r))

    diga("")
    diga("=" * 92)
    diga("FLORESTA ALEATORIA, por conjunto de variaveis")
    diga("=" * 92)
    diga(f"{'':28s} ROC    PR-AUC   vezes   rec@100  inv/surto  limiares uteis")
    for nome, cols in confs.items():
        r = pv.rodar(painel, cols, floresta, alvo=ALVO, verboso=False)
        resultados[f"rf_{nome}"] = r
        diga(linha(f"floresta: {nome}", r))

    diga("")
    diga("=" * 92)
    diga("ARVORES IMPULSIONADAS, por conjunto de variaveis")
    diga("=" * 92)
    diga(f"{'':28s} ROC    PR-AUC   vezes   rec@100  inv/surto  limiares uteis")
    melhor, melhor_nome = None, None
    for nome, cols in confs.items():
        r = pv.rodar(painel, cols, gradiente, alvo=ALVO, verboso=False)
        resultados[f"xgb_{nome}"] = r
        diga(linha(f"impulsionadas: {nome}", r))
        if melhor is None or (r["agregado"]["pr_auc"] > melhor["agregado"]["pr_auc"]):
            melhor, melhor_nome = r, f"xgb_{nome}"

    # Detalhe por temporada do melhor conjunto: a media esconde que prever a
    # chegada da onda e diferente de prever a sua continuacao.
    diga("")
    diga("=" * 92)
    diga(f"DESEMPENHO POR TEMPORADA - {melhor_nome}")
    diga("=" * 92)
    diga("temporada  eventos    ROC   PR-AUC   vezes  rec@100  rec@500")
    for m in melhor["por_temporada"]:
        diga(f"  {m['temporada']}/{str(m['temporada'] + 1)[2:]}   {m['positivos']:5d}"
             f"  {m['roc_auc']:.3f}  {m['pr_auc']:.4f}  {m['pr_auc_sobre_prevalencia']:6.1f}x"
             f"   {m.get('recall_top100', float('nan')):.3f}"
             f"   {m.get('recall_top500', float('nan')):.3f}")

    melhor["oof"].to_parquet(OOF, index=False)

    saida = {
        "alvo": ALVO,
        "primeira_temporada_prevista": pv.PRIMEIRA_PREVISTA,
        "conjuntos": {k: len(v) for k, v in confs.items()},
        "melhor": melhor_nome,
        "resultados": {
            k: {"agregado": v["agregado"], "limiares_uteis": v["limiares_uteis"],
                "por_temporada": v["por_temporada"]}
            for k, v in resultados.items()},
        "curva_de_decisao_do_melhor": melhor["curva_de_decisao"],
        "segundos_total": round(time.time() - inicio, 1),
    }
    SAIDA.write_text(json.dumps(saida, indent=2, ensure_ascii=False,
                                default=float), encoding="utf-8")
    LOG.write_text("\n".join(_log), encoding="utf-8")
    print(f"\nmetricas em {SAIDA}")


if __name__ == "__main__":
    main()
