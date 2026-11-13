"""Validacao prospectiva por temporada: o unico teste que corresponde ao uso.

O esquema
---------
Para cada temporada T a partir de um minimo: treina em tudo ate T-1, preve T,
avanca. Nenhuma linha da temporada prevista, nem de nenhuma posterior, participa
do ajuste. E exatamente a operacao real - ao fim de junho, com o historico
fechado, dizer onde a febre amarela vai aparecer na temporada que comeca.

Por que nao o bloco espaco-temporal
-----------------------------------
O bloco espaco-temporal retira do treino uma macrorregiao inteira alem do ano.
E o teste certo para uma afirmacao de transferibilidade ecologica - que as
relacoes aprendidas num lugar se transferem para outro.

Para um modelo de contagio, e o teste errado, e por um motivo simples: em uso
real sabe-se, sim, o que aconteceu nos municipios vizinhos. Esconder a
macrorregiao inteira remove justamente a informacao que o modelo existe para usar,
e mediria a capacidade de prever uma invasao sem saber que a onda esta chegando.
Isso nao e conservadorismo, e uma pergunta diferente.

O bloco espaco-temporal continua sendo reportado, em `transferencia()`, como
medida de transferibilidade ecologica.

Linhas de base obrigatorias
---------------------------
Nenhum modelo tem valor se nao superar as regras triviais:

    persistencia   quem teve evento na temporada passada tera de novo
    vizinhanca     risco proporcional a forca de infeccao, sem nenhum aprendizado
    historico      quem teve mais eventos nos ultimos dez anos

Metricas
--------
PR-AUC sobre prevalencia e a metrica principal: com desfecho em 0,7% a ROC-AUC
permanece alta mesmo quando quase todo alerta e falso.

Acrescentam-se duas metricas operacionais (ver `metricas.py`) que dizem
mais a um gestor que qualquer AUC: quantos municipios e preciso investigar para
encontrar um surto, e que fracao dos surtos esta nos K municipios de maior risco.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Primeira temporada prevista. Antes disso o treino tem poucas temporadas e
# nenhum evento suficiente para ajustar qualquer coisa.
PRIMEIRA_PREVISTA = 2003

# Tamanhos de lista de prioridade para recall@K. Sao ordens de grandeza
# operacionais: 50 municipios e uma forca-tarefa, 500 e quase 10% do pais.
TOPOS = [50, 100, 200, 500]


def temporadas_de_teste(painel: pd.DataFrame,
                        primeira: int = PRIMEIRA_PREVISTA) -> list[int]:
    ts = sorted(painel["temporada"].unique())
    return [t for t in ts if t >= primeira]


def metricas(y, proba, topos=TOPOS) -> dict:
    """Metricas de uma temporada, ou do agregado."""
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n, pos = len(y), int(y.sum())
    if pos == 0 or pos == n:
        return {"n": n, "positivos": pos, "erro": "temporada sem ambas as classes"}

    prev = pos / n
    pr = average_precision_score(y, proba)
    saida = {
        "n": n, "positivos": pos, "prevalencia": prev,
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": pr, "pr_auc_sobre_prevalencia": pr / prev,
    }

    ordem = np.argsort(-proba)
    for k in topos:
        if k > n:
            continue
        capturados = int(y[ordem[:k]].sum())
        saida[f"recall_top{k}"] = capturados / pos
        saida[f"precisao_top{k}"] = capturados / k
        saida[f"investigados_por_surto_top{k}"] = (
            k / capturados if capturados else float("nan"))
    return saida


def curva_de_decisao(y, proba, limiares=None) -> list[dict]:
    """Beneficio liquido por limiar, contra investigar todos e nenhum.

    beneficio = VP/n - FP/n * p/(1-p), com p o limiar. E a formulacao padrao de
    curva de decisao clinica: o segundo termo e a razao de troca implicita entre
    um falso positivo e um verdadeiro positivo naquele limiar.
    """
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n = len(y)
    prev = y.mean()
    if limiares is None:
        limiares = np.unique(np.round(np.geomspace(1e-4, 0.5, 60), 6))

    saida = []
    for p in limiares:
        if p >= 1:
            continue
        sinal = proba >= p
        vp = int((sinal & (y == 1)).sum())
        fp = int((sinal & (y == 0)).sum())
        troca = p / (1 - p)
        saida.append({
            "limiar": float(p),
            "beneficio_liquido": vp / n - fp / n * troca,
            "beneficio_tratar_todos": prev - (1 - prev) * troca,
            "sinalizados": int(sinal.sum()),
        })
    return saida


def limiares_uteis(curva) -> int:
    """Quantos limiares tem beneficio estritamente acima das duas triviais."""
    return sum(1 for c in curva
               if c["beneficio_liquido"] > max(c["beneficio_tratar_todos"], 0)
               + 1e-12)


# --------------------------------------------------------------------------
# Linhas de base
# --------------------------------------------------------------------------
def base_persistencia(treino, teste) -> np.ndarray:
    """Quem teve evento na temporada anterior. A regra mais simples que existe."""
    return teste["evento_temporada_anterior"].fillna(0).to_numpy(dtype=float)


def base_vizinhanca(treino, teste) -> np.ndarray:
    """Forca de infeccao a 100 km, sem nenhum ajuste. So a epidemiologia."""
    return teste["forca_k100"].fillna(0).to_numpy(dtype=float)


def base_historico(treino, teste) -> np.ndarray:
    """Contagem de eventos locais nas dez temporadas anteriores."""
    return teste["eventos_locais_10t"].fillna(0).to_numpy(dtype=float)


BASES = {
    "persistencia": base_persistencia,
    "vizinhanca": base_vizinhanca,
    "historico_local": base_historico,
}


# --------------------------------------------------------------------------
# Execucao prospectiva
# --------------------------------------------------------------------------
def rodar(painel: pd.DataFrame, colunas: list[str], ajustar, alvo="yf_event",
          primeira=PRIMEIRA_PREVISTA, verboso=True) -> dict:
    """Janela expansiva: treina em < T, preve T, para toda temporada T.

    `ajustar(X_treino, y_treino)` devolve um objeto com `predict_proba`, ou uma
    funcao que recebe (treino, teste) e devolve o vetor de probabilidades - o que
    permite passar linhas de base pelo mesmo caminho.
    """
    testes = temporadas_de_teste(painel, primeira)
    por_temporada, oof = [], []

    for t in testes:
        treino = painel[painel["temporada"] < t]
        teste = painel[painel["temporada"] == t]
        if treino[alvo].sum() < 10 or teste[alvo].sum() == 0:
            continue

        if callable(ajustar) and getattr(ajustar, "__eh_base__", False):
            proba = ajustar(treino, teste)
        else:
            modelo = ajustar(treino[colunas], treino[alvo])
            proba = modelo.predict_proba(teste[colunas])[:, 1]

        m = metricas(teste[alvo], proba)
        m["temporada"] = int(t)
        m["n_treino"] = int(len(treino))
        m["positivos_treino"] = int(treino[alvo].sum())
        por_temporada.append(m)

        oof.append(pd.DataFrame({
            "cod_mun": teste["cod_mun"].to_numpy(),
            "temporada": t,
            "regiao": teste["regiao"].to_numpy(),
            "y": teste[alvo].to_numpy(),
            "proba": proba,
        }))
        if verboso:
            print(f"  {t}/{str(t + 1)[2:]}  n={m['n']:,} pos={m['positivos']:4d}"
                  f"  ROC {m['roc_auc']:.3f}  PR {m['pr_auc']:.4f}"
                  f"  {m['pr_auc_sobre_prevalencia']:6.1f}x")

    oof = pd.concat(oof, ignore_index=True)
    agregado = metricas(oof["y"], oof["proba"])
    curva = curva_de_decisao(oof["y"], oof["proba"])

    return {
        "por_temporada": por_temporada,
        "agregado": agregado,
        "curva_de_decisao": curva,
        "limiares_uteis": limiares_uteis(curva),
        "oof": oof,
    }


def marcar_base(fn):
    fn.__eh_base__ = True
    return fn


for _nome, _fn in BASES.items():
    marcar_base(_fn)
