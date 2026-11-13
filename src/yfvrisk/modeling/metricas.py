"""Metricas de avaliacao, sempre reportadas em conjunto.

Um ROC-AUC alto pode coexistir com precisao de 0,33 - dois em cada tres alertas
falsos -, e para uma ferramenta que orienta alocacao de vacina e de vigilancia
essa e a informacao operacional mais importante.

Aqui nenhuma metrica sai sozinha. Toda avaliacao devolve discriminacao,
calibracao e utilidade de decisao ao mesmo tempo, porque as tres respondem a
perguntas diferentes:

  discriminacao  o modelo ordena bem os municipios?          ROC-AUC, PR-AUC
  calibracao     a probabilidade de 0,3 significa 30%?       Brier, calibracao
  decisao        vale a pena agir sobre este alerta?         curva de decisao

Um modelo pode ter ROC-AUC alto e ser inutil na pratica: com prevalencia de 0,74%
ao ano, a PR-AUC e a precisao sao o que separa uma ferramenta de vigilancia de um
gerador de alarme falso.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def limiar_por_f_beta(y, proba, beta: float = 2.0):
    """Limiar que maximiza F-beta, e as metricas nesse limiar.

    Beta 2 pesa recall quatro vezes mais que precisao, refletindo que deixar de
    detectar um surto custa mais que investigar um alerta falso.

    IMPORTANTE: escolher o limiar no mesmo conjunto em que se reporta o
    desempenho produz vies otimista. Por isso a funcao existe separada, para ser
    chamada num conjunto de validacao distinto do de teste.
    """
    prec, rec, lim = precision_recall_curve(y, proba)
    b2 = beta ** 2
    den = b2 * prec[:-1] + rec[:-1]
    f = np.divide((1 + b2) * prec[:-1] * rec[:-1], den,
                  out=np.zeros_like(den), where=den > 0)
    if len(f) == 0:
        return 0.5
    return float(lim[int(np.argmax(f))])


def curva_de_decisao(y, proba, limiares=None):
    """Beneficio liquido por limiar (decision curve analysis).

    Responde a pergunta que o gestor de fato faz: se eu agir sobre todos os
    municipios acima deste limiar, quanto ganho em relacao a nao agir sobre
    nenhum, ou a agir sobre todos?

    beneficio_liquido = (VP/n) - (FP/n) * (p / (1 - p))

    em que p e o limiar, interpretado como a razao de custo entre falso positivo
    e falso negativo que torna a acao indiferente.
    """
    y = np.asarray(y)
    n = len(y)
    limiares = limiares if limiares is not None else np.linspace(0.001, 0.5, 60)

    linhas = []
    for p in limiares:
        pred = (proba >= p).astype(int)
        vp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        beneficio = vp / n - (fp / n) * (p / (1 - p))
        # Referencia: tratar todos.
        prev = y.mean()
        beneficio_todos = prev - (1 - prev) * (p / (1 - p))
        linhas.append({
            "limiar": float(p),
            "beneficio_liquido": float(beneficio),
            "beneficio_tratar_todos": float(beneficio_todos),
            "beneficio_tratar_nenhum": 0.0,
            "municipios_sinalizados": int(pred.sum()),
        })
    return linhas


def calibracao(y, proba, n_faixas: int = 10):
    """Probabilidade prevista contra frequencia observada, por faixa.

    Sem isto, a rampa de cor de um mapa de risco nao significa nada: uma cor
    rotulada como risco 0,3 pode corresponder a 3% ou a 60% de eventos reais.
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    # Faixas por quantil, porque com prevalencia de 0,74% faixas uniformes
    # deixariam quase tudo na primeira.
    bordas = np.unique(np.quantile(proba, np.linspace(0, 1, n_faixas + 1)))
    if len(bordas) < 3:
        return []
    idx = np.clip(np.digitize(proba, bordas[1:-1]), 0, len(bordas) - 2)

    faixas = []
    for i in range(len(bordas) - 1):
        sel = idx == i
        if sel.sum() == 0:
            continue
        faixas.append({
            "faixa": i,
            "n": int(sel.sum()),
            "probabilidade_media_prevista": float(proba[sel].mean()),
            "frequencia_observada": float(y[sel].mean()),
            "eventos": int(y[sel].sum()),
        })
    return faixas


def avaliar(y, proba, limiar: float | None = None, beta: float = 2.0) -> dict:
    """Conjunto completo de metricas para um par (verdade, probabilidade)."""
    y = np.asarray(y)
    proba = np.asarray(proba)

    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "positivos": int(y.sum()),
                "erro": "conjunto sem ambas as classes"}

    limiar_usado = limiar if limiar is not None else limiar_por_f_beta(y, proba, beta)
    pred = (proba >= limiar_usado).astype(int)
    tn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    precisao = vp / (vp + fp) if (vp + fp) else 0.0
    recall = vp / (vp + fn) if (vp + fn) else 0.0
    b2 = beta ** 2
    f_beta = ((1 + b2) * precisao * recall / (b2 * precisao + recall)
              if (b2 * precisao + recall) else 0.0)

    prevalencia = float(y.mean())
    return {
        "n": int(len(y)),
        "positivos": int(y.sum()),
        "prevalencia": prevalencia,
        # discriminacao
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        # A PR-AUC precisa ser lida contra a prevalencia: um classificador
        # aleatorio tem PR-AUC igual a prevalencia, nao 0,5.
        "pr_auc_sobre_prevalencia": float(average_precision_score(y, proba) / prevalencia)
        if prevalencia > 0 else None,
        # limiar e matriz
        "limiar": float(limiar_usado),
        "vp": int(vp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precisao": float(precisao),
        "recall": float(recall),
        f"f{int(beta)}": float(f_beta),
        # Numero necessario para detectar um evento: quantos municipios e preciso
        # investigar para encontrar um surto. E a metrica operacional direta.
        "municipios_investigados_por_evento": float((vp + fp) / vp) if vp else None,
        # calibracao
        "brier": float(brier_score_loss(y, proba)),
        # Referencia: prever a prevalencia para todos.
        "brier_referencia_prevalencia": float(np.mean((y - prevalencia) ** 2)),
    }
