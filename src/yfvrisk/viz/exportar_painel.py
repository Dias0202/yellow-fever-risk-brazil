"""Exporta para o painel o que as analises mediram.

O painel publico e uma saida do projeto, nao uma fonte propria. Este modulo e a
unica ponte entre os dois: le `results` e `data/processed` e escreve
JSON em `dashboard/dados`. Nenhum numero e digitado a mao no HTML.

A razao e concreta: numeros escritos a mao num HTML nao tem como ser reproduzidos
nem conferidos, e divergem em silencio dos arquivos de metrica. Com este modulo,
atualizar o painel e reexecutar a analise.

O que o painel PODE mostrar
---------------------------
A restricao vem das proprias medicoes, e e severa:

- Nao pode haver mapa municipal de risco do modelo ambiental. A validacao em
  bloco mediu que a calibracao superestima o decil de topo por fator de 2,4x (temporal) a 4,7x
  (espaco-temporal), de modo que a rampa de cor nao pode ser lida como
  probabilidade.
- Nao pode haver mapa de cenario. A projecao mediu que em 83% dos municipios a
  dispersao entre cenarios e menor que a dispersao entre reajustes do modelo.
- Nao pode haver alavanca de vacinacao. A variavel nao carrega sinal
  generalizavel, nem antes nem depois de removido o vazamento.

O que sobra e mais interessante do que o que sai: como o desempenho depende da
particao, onde o modelo transfere, o que ele de fato usa, e por que a cobertura
vacinal registrada e resposta ao surto e nao protecao contra ele.

Rodar:
    python -m yfvrisk.viz.exportar_painel
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from yfvrisk import config as cfg
from yfvrisk.features.painel import GRUPOS

warnings.filterwarnings("ignore")

DESTINO = cfg.RAIZ / "dashboard" / "dados"
ALVO = "yf_event"

# Corte de 2019: treino ate 2019, explicacao em 2020-2025. E o modelo que de fato
# seria implantado, e o unico dos dois cortes com discriminacao real.
CORTE_SHAP = 2019

# Pontos por variavel no enxame de SHAP. Mais que isso nao acrescenta leitura e
# so pesa o arquivo; menos comeca a mostrar buracos na nuvem.
PONTOS_SHAP = 1200
VARIAVEIS_SHAP = 14


def _ler_json(nome: str) -> dict:
    caminho = cfg.METRICAS / nome
    if not caminho.exists():
        raise SystemExit(f"{caminho} nao existe. Rode a analise correspondente.")
    return json.loads(caminho.read_text(encoding="utf-8"))


def _escrever(nome: str, dados) -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    caminho = DESTINO / nome
    caminho.write_text(json.dumps(dados, ensure_ascii=False, separators=(",", ":")),
                       encoding="utf-8")
    print(f"[ok] {nome:24s} {caminho.stat().st_size / 1024:7.1f} kB")


def _arred(x, casas=5):
    """Arredonda para o painel. Um JSON com 17 digitos por numero e ruido."""
    if isinstance(x, dict):
        return {k: _arred(v, casas) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_arred(v, casas) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), casas)
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


# --------------------------------------------------------------------------
# 1. Validacao: o desempenho depende da particao
# --------------------------------------------------------------------------
def exportar_validacao() -> None:
    f3a = _ler_json("f3_01_comparacao_validacao.json")
    f3b = _ler_json("f3_02_calibracao_decisao_bloco.json")

    # So as linhas da floresta com o conjunto ambiental, o modelo de
    # adequabilidade usado nas projecoes. As referencias e as variantes ficam de fora do grafico
    # principal e entram como contexto.
    def _achar(rotulo, esquema):
        for r in f3a["resultados"]:
            if r["rotulo"] == rotulo and r["esquema"] == esquema:
                return r
        return None

    esquemas = []
    for rotulo, chave, nome, explicacao in [
        ("floresta aleatoria | so ambiental", "A_aleatorio", "Aleatória",
         "Linhas sorteadas ao acaso. É o atalho mais comum, e coloca o mesmo "
         "município e anos adjacentes dos dois lados da partição."),
        ("floresta aleatoria | so ambiental", "C_espacial", "Espacial",
         "Nenhum município da UF de teste aparece no treino. Mede transferência "
         "para território não visto."),
        ("floresta aleatoria | so ambiental", "B_temporal", "Temporal",
         "Nenhuma linha do ano de teste aparece no treino. É a partição que "
         "corresponde à pergunta “qual será o risco no próximo ano”."),
    ]:
        r = _achar(rotulo, chave)
        if r:
            esquemas.append({"nome": nome, "explicacao": explicacao,
                             "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                             "vezes_o_acaso": r["pr_auc_sobre_prevalencia"],
                             "investigados_por_evento":
                                 r["municipios_investigados_por_evento"]})

    bloco = f3b["resultados"]["espaco_temporal"]
    m = bloco["metricas"]
    esquemas.append({
        "nome": "Espaço-temporal",
        "explicacao": "Nem a macrorregião nem o ano de teste aparecem no treino. "
                      "É o esquema que corresponde a projetar o risco para onde não há história.",
        "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
        "vezes_o_acaso": m["pr_auc_sobre_prevalencia"],
        "investigados_por_evento": m["municipios_investigados_por_evento"],
        "ic_roc_auc": [bloco["ic_roc_auc"]["ic_inferior"],
                       bloco["ic_roc_auc"]["ic_superior"]],
    })

    dados = {
        "prevalencia": f3a["prevalencia"],
        "esquemas": esquemas,
        "por_regiao": [
            {"regiao": r["regiao"], "roc_auc": r["roc_auc"],
             "ic": [r["ic_roc_auc"]["ic_inferior"], r["ic_roc_auc"]["ic_superior"]],
             "pr_auc": r["pr_auc"], "vezes_o_acaso": r["pr_auc_sobre_prevalencia"],
             "positivos": r["positivos"], "n": r["n"]}
            for r in bloco["por_regiao"]],
        "calibracao": [
            {"faixa": c["faixa"], "previsto": c["probabilidade_media_prevista"],
             "observado": c["frequencia_observada"], "n": c["n"]}
            for c in bloco["calibracao"]],
        "curva_de_decisao": [
            {"limiar": c["limiar"], "modelo": c["beneficio_liquido"],
             "todos": c["beneficio_tratar_todos"]}
            for c in bloco["curva_de_decisao"]],
        "limiares_uteis": bloco["limiares_uteis"],
    }
    _escrever("validacao.json", _arred(dados, 6))


# --------------------------------------------------------------------------
# 2. Importancia: MDI no treino contra permutacao fora da amostra
# --------------------------------------------------------------------------
def exportar_importancia() -> None:
    cortes = {}
    for corte in (2014, 2019):
        d = _ler_json(f"f4_01_explicabilidade_corte{corte}.json")
        perm = {r["variavel"]: r for r in d["importancia_permutacao"]}
        mdi = {r["variavel"]: r for r in d["importancia_mdi"]}
        posto_perm = {v: i + 1 for i, v in enumerate(perm)}
        posto_mdi = {v: i + 1 for i, v in enumerate(mdi)}

        cortes[str(corte)] = {
            "roc_auc": d["desempenho_fora_da_amostra"]["roc_auc"],
            "pr_auc": d["desempenho_fora_da_amostra"]["pr_auc"],
            "vezes_o_acaso":
                d["desempenho_fora_da_amostra"]["pr_auc_sobre_prevalencia"],
            "rho_fora": d["spearman_mdi_permutacao_fora"],
            "rho_fora_p": d["spearman_mdi_permutacao_fora_p"],
            "rho_treino": d["spearman_mdi_permutacao_treino"],
            "variaveis": [
                {"nome": v,
                 "grupo": next((g for g, cols in GRUPOS.items() if v in cols), "?"),
                 "permutacao": perm[v]["importancia_media"],
                 "ic": [perm[v]["ic_inferior"], perm[v]["ic_superior"]],
                 "mdi": mdi[v]["mdi"],
                 "posto_permutacao": posto_perm[v],
                 "posto_mdi": posto_mdi[v]}
                for v in perm],
            "grupos": [{"grupo": g["grupo"], "n": g["n_variaveis"],
                        "queda_relativa": g["queda_relativa"]}
                       for g in d["importancia_por_grupo"]],
        }

    sobreviventes = sorted(
        {v["nome"] for v in cortes["2014"]["variaveis"] if v["ic"][0] > 0}
        & {v["nome"] for v in cortes["2019"]["variaveis"] if v["ic"][0] > 0})

    _escrever("importancia.json",
              _arred({"cortes": cortes, "sobrevivem_aos_dois": sobreviventes}, 6))


# --------------------------------------------------------------------------
# 3. SHAP por ponto, para o enxame
# --------------------------------------------------------------------------
def exportar_shap() -> None:
    """Recalcula o SHAP guardando os valores por ponto, e nao so a media.

    A analise de explicabilidade salva apenas `shap_medio_absoluto`, que basta para
    ordenar mas nao para desenhar. O enxame precisa do par (contribuicao, valor da
    variavel) de cada observacao, que e o que mostra a DIRECAO do efeito - um
    grafico de barras de importancia nao tem como expressar isso.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    from yfvrisk.explain.importancia import valores_shap

    painel = pd.read_parquet(cfg.PROCESSADO / "painel_analitico.parquet")
    colunas = [c for g in ["CLIMA", "CLIMA_DEFASADO", "PAISAGEM", "IMUNIDADE",
                           "DEMOGRAFIA"] for c in GRUPOS[g] if c in painel.columns]

    treino = painel[painel["ano"] <= CORTE_SHAP]
    fora = painel[painel["ano"] > CORTE_SHAP]

    modelo = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=42)),
    ])
    print(f"[shap] ajustando ate {CORTE_SHAP} ({len(treino):,} linhas)")
    modelo.fit(treino[colunas], treino[ALVO])

    # O SHAP e medido FORA da amostra, pelo mesmo motivo que a permutacao: no
    # treino ele descreveria o que a floresta memorizou.
    print(f"[shap] explicando {len(fora):,} linhas nao vistas")
    valores, Xt, resumo = valores_shap(modelo, fora[colunas], amostra=PONTOS_SHAP)

    ordem = list(resumo["variavel"][:VARIAVEIS_SHAP])
    indices = {c: i for i, c in enumerate(colunas)}

    variaveis = []
    for nome in ordem:
        j = indices[nome]
        contrib = valores[:, j]
        bruto = Xt[nome].to_numpy(dtype=float)
        # Valor da variavel normalizado por posto percentual: a cor do enxame
        # tem de ser legivel em variaveis de escalas incomparaveis, e o posto
        # nao se deixa dominar por caudas como perda_acumulada_km2.
        posto = pd.Series(bruto).rank(pct=True).to_numpy()
        variaveis.append({
            "nome": nome,
            "grupo": next((g for g, cols in GRUPOS.items() if nome in cols), "?"),
            "medio_absoluto": float(np.abs(contrib).mean()),
            "unidade_min": float(np.nanmin(bruto)),
            "unidade_max": float(np.nanmax(bruto)),
            # Pares achatados, para o JSON nao carregar 1200 objetos por variavel.
            "contribuicao": [round(float(v), 5) for v in contrib],
            "valor_relativo": [round(float(v), 3) for v in posto],
        })

    _escrever("shap.json", {
        "corte": CORTE_SHAP,
        "n_pontos": int(len(valores)),
        "n_variaveis_mostradas": len(variaveis),
        "n_variaveis_no_modelo": len(colunas),
        "variaveis": variaveis,
    })


# --------------------------------------------------------------------------
# 4. Curvas ALE: a forma da resposta
# --------------------------------------------------------------------------
def exportar_ale() -> None:
    cortes = {}
    for corte in (2014, 2019):
        caminho = cfg.TABELAS / f"f4_01_curvas_ale_corte{corte}.csv"
        if not caminho.exists():
            continue
        d = pd.read_csv(caminho)
        cortes[str(corte)] = [
            {"variavel": v,
             "pontos": [{"x": float(a), "y": float(b)}
                        for a, b in zip(g["valor"], g["efeito_ale"])]}
            for v, g in d.groupby("variavel")]
    _escrever("ale.json", _arred({"cortes": cortes}, 6))


# --------------------------------------------------------------------------
# 5. Vacinacao: o estudo de evento
# --------------------------------------------------------------------------
def exportar_vacinacao() -> None:
    d = _ler_json("f5_00_diagnostico_vacinacao.json")
    h6 = d["h6_causalidade_reversa"]
    _escrever("vacinacao.json", _arred({
        "estudo_de_evento": {
            "todos": h6["estudo_de_evento_cobertura"],
            "primeiro_surto": h6["estudo_de_evento_primeiro_surto"],
        },
        "por_defasagem": [
            {"defasagem": int(k), **v}
            for k, v in h6["associacao_por_defasagem"].items()],
        "dentro_do_municipio": {
            "bruta": d["h5_dentro_do_municipio"]["rho_bruta"],
            "centrada_no_ano": d["h5_dentro_do_municipio"]["rho_centrada_no_ano"],
            "dupla_centragem": d["h5_dentro_do_municipio"]["rho_dupla_centragem"],
            "ic": d["h5_dentro_do_municipio"]["ic95_dupla_centragem"],
        },
        "saturacao": [
            {"ano": r["ano"], "media": r["media"], "p10": r["p10"],
             "p90": r["p90"], "amplitude": r["amplitude_interdecil"]}
            for r in d["h4_saturacao"]["por_ano"]],
    }, 5))


# --------------------------------------------------------------------------
# 6. Clima: o ensemble e a projecao
# --------------------------------------------------------------------------
def exportar_clima() -> None:
    ens = _ler_json("f5_clima_futuro_diagnostico.json")
    proj = _ler_json("f5_01_projecao.json")

    combos = pd.DataFrame(ens["por_combinacao"])
    cen = pd.DataFrame(proj["por_cenario"])
    base = proj["risco_linha_de_base_medio"]
    cen["aumento_relativo"] = cen["risco_medio"] / base - 1

    _escrever("clima.json", _arred({
        "gcms": ens["gcms_pedidos"], "ssps": ens["ssps_pedidos"],
        "periodo": ens["periodo_futuro"],
        "aquecimento": combos[["gcm", "ssp", "delta_tmax_medio",
                               "razao_ppt_mediana"]].to_dict("records"),
        "dispersao_entre_gcms": float(
            combos.groupby("gcm")["delta_tmax_medio"].mean().std()),
        "dispersao_entre_ssps": float(
            combos.groupby("ssp")["delta_tmax_medio"].mean().std()),
        "risco_linha_de_base": base,
        "projecao": cen[["gcm", "ssp", "risco_medio",
                         "aumento_relativo"]].to_dict("records"),
        "contrastes": proj["contrastes_pareados"],
        "por_regiao": proj["por_regiao"],
        "razao_sinal_ruido_mediana": proj["razao_sinal_ruido_mediana"],
        "municipios_com_sinal_acima_do_ruido":
            proj["municipios_com_sinal_acima_do_ruido"],
        "municipios_projetados": proj["municipios_projetados"],
    }, 6))


# --------------------------------------------------------------------------
# 7. A serie observada
# --------------------------------------------------------------------------
def exportar_serie() -> None:
    """O que de fato aconteceu. E a unica camada do painel que nao e modelo."""
    p = pd.read_parquet(cfg.PROCESSADO / "painel_analitico.parquet")
    macro = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}
    p["regiao"] = (p["cod_uf"].astype(int) // 10).map(macro)

    por_ano = (p.groupby("ano")
               .agg(eventos=(ALVO, "sum"), casos=("caso_humano", "sum"),
                    epizootias=("epizootia", "sum"),
                    municipios=("cod_mun", "nunique"))
               .reset_index())

    por_ano_regiao = (p.groupby(["ano", "regiao"])[ALVO].sum()
                      .unstack(fill_value=0).reset_index())

    _escrever("serie.json", _arred({
        "anos": [int(a) for a in por_ano["ano"]],
        "eventos": [int(v) for v in por_ano["eventos"]],
        "casos_humanos": [int(v) for v in por_ano["casos"]],
        "epizootias": [int(v) for v in por_ano["epizootias"]],
        "regioes": list(por_ano_regiao.columns[1:]),
        "por_regiao": {c: [int(v) for v in por_ano_regiao[c]]
                       for c in por_ano_regiao.columns[1:]},
        "sem_vigilancia_epizootias": sorted(cfg.ANOS_SEM_VIGILANCIA_EPIZOOTIAS),
        "total_eventos": int(p[ALVO].sum()),
        "municipios": int(p["cod_mun"].nunique()),
        "linhas": int(len(p)),
    }, 4))


# --------------------------------------------------------------------------
# 8. O modelo que preve a proxima temporada
# --------------------------------------------------------------------------
def exportar_previsao() -> None:
    """Desempenho prospectivo, calibracao e a lista da proxima temporada.

    E a unica camada do painel que faz afirmacao sobre o futuro, e por isso a que
    mais precisa vir acompanhada do proprio desempenho. Exporta os dois juntos,
    de proposito: a lista sem o numero seria uma promessa sem lastro.
    """
    d = _ler_json("f6_05_previsao_operacional.json")
    reg = _ler_json("f6_06_regularizacao.json")
    ganho = _ler_json("f6_01_previsao.json")
    diag = _ler_json("f6_temporada_diagnostico.json")

    tabela = pd.read_csv(cfg.TABELAS / "f6_05_previsao_proxima_temporada.csv")
    topo = tabela.nsmallest(120, "posto_logistica")[
        ["cod_mun", "nome_mun", "uf", "regiao", "probabilidade",
         "posto_logistica", "posto_forca", "dist_foco_mais_proximo",
         "focos_em_100km", "temporadas_desde_evento",
         "cobertura_florestal_t1", "altitude_media"]]

    melhor = reg["resultados"].get(reg["resultados"].get("melhor", ""), {})
    conjuntos = {k: v["agregado"]["pr_auc_sobre_prevalencia"]
                 for k, v in ganho["resultados"].items()
                 if k.startswith("rf_")}

    _escrever("previsao.json", _arred({
        "temporada_prevista": d["temporada_prevista"],
        "ultima_observada": d["ultima_observada"],
        "variaveis": d["variaveis_operacionais"],
        "desempenho": d["desempenho_prospectivo"],
        "calibracao": d["calibracao_fora_da_amostra"],
        "coeficientes": reg["resultados"].get("coeficientes_do_melhor", []),
        "monotonicidade": {
            "taxas_por_decil": melhor.get("taxas_por_decil", []),
            "correlacao_posto": melhor.get("correlacao_posto_decil"),
        },
        "ganho_por_conjunto": conjuntos,
        "sazonalidade_casos": diag["casos_humanos"]["por_mes"],
        "sazonalidade_epizootias": diag["epizootias"]["por_mes"],
        "eventos_por_temporada": diag["eventos_por_temporada"],
        "topo": topo.to_dict("records"),
        "total_municipios": int(len(tabela)),
        "recomendacao": d["recomendacao"],
    }, 6))


def main() -> None:
    print(f"[destino] {DESTINO}")
    exportar_serie()
    exportar_validacao()
    exportar_importancia()
    exportar_ale()
    exportar_vacinacao()
    exportar_clima()
    exportar_shap()
    print("\npainel: dados exportados")


if __name__ == "__main__":
    main()
