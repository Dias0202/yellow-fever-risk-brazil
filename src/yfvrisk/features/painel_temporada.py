"""Painel de modelagem por temporada, para previsao da proxima temporada.

Decisoes de desenho, todas motivadas pelo objetivo de PREVER e nao de descrever:

1. **Unidade temporal e a temporada julho-junho**, nao o ano civil. 97,6% dos
   casos humanos ocorrem entre dezembro e maio, e o ano civil corta a temporada
   ao meio.

2. **Tudo e defasado.** A previsao e feita ao fim de junho, para a temporada que
   comeca em julho. Nesse instante conhece-se o desfecho ate a temporada t-1 e o
   ambiente ate junho. O clima da temporada que vai comecar NAO e conhecido, e
   por isso nao entra: entra o clima das temporadas anteriores e a normal
   climatologica do municipio.

3. **A propagacao entra.** Vinte e sete variaveis de forca de infeccao,
   refratariedade e frente de onda, do modulo `epidemia`. Sao elas que traduzem o
   fato de a febre amarela silvestre ser uma onda que se propaga, e nao um
   atributo estatico de habitat.

4. **A anomalia entra ao lado do nivel.** Um municipio quente e diferente de um
   municipio que esteve anormalmente quente na temporada passada. O primeiro e
   adequabilidade estrutural; o segundo e gatilho. Sao mecanismos distintos e
   entram como variaveis distintas.

Grupos de variaveis
-------------------
    PROPAGACAO     forca de infeccao, distancia ao foco, frente de onda
    SUSCETIBILIDADE refratariedade e historico local
    CLIMA_NORMAL   normal climatologica do municipio: adequabilidade estrutural
    CLIMA_RECENTE  clima observado nas temporadas anteriores
    CLIMA_ANOMALIA desvio do recente em relacao a normal
    PAISAGEM       floresta e perda florestal, defasadas
    RELEVO         altitude e aspereza do terreno
    DEMOGRAFIA     populacao e densidade, defasadas
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yfvrisk import config as cfg

MES_INICIO = 7

# Janelas dentro da temporada, em mes ordinal (julho = 0).
#   pre-estacao  set-nov: as primeiras chuvas, que constroem a populacao de vetor
#                antes da janela de transmissao
#   transmissao  dez-mai: onde estao 97,6% dos casos humanos
JANELA_PRE = [2, 3, 4]
JANELA_TRANSMISSAO = [5, 6, 7, 8, 9, 10]

# Numero de temporadas usadas para a normal climatologica de cada municipio.
# Trinta anos e a convencao da OMM; aqui a serie inteira disponivel se aproxima
# disso e a normal fica fixa, calculada uma vez sobre todo o periodo.
SAIDA = cfg.PROCESSADO / "painel_modelagem_temporada.parquet"
DIAGNOSTICO = cfg.METRICAS / "f6_painel_diagnostico.json"

GRUPOS: dict[str, list[str]] = {}


def _clima_por_temporada() -> pd.DataFrame:
    """Agrega o mensal do TerraClimate em temporadas julho-junho."""
    m = pd.read_parquet(cfg.PROCESSADO / "clima_muni_mes.parquet")
    m["temporada"] = np.where(m["mes"] >= MES_INICIO, m["ano"], m["ano"] - 1)
    m["ordinal"] = (m["mes"] - MES_INICIO) % 12

    # So temporadas completas: uma temporada com meses faltando produziria total
    # de chuva artificialmente baixo, que o modelo leria como seca.
    completas = m.groupby(["cod_mun", "temporada"])["mes"].nunique()
    validas = completas[completas == 12].index
    m = m.set_index(["cod_mun", "temporada"]).loc[validas].reset_index()

    g = m.groupby(["cod_mun", "temporada"])
    fora = m[m["ordinal"].isin(JANELA_PRE)].groupby(["cod_mun", "temporada"])
    trans = m[m["ordinal"].isin(JANELA_TRANSMISSAO).tolist()].groupby(
        ["cod_mun", "temporada"])

    anual = pd.DataFrame({
        "ppt_temporada": g["ppt"].sum(),
        "tmax_temporada": g["tmax"].mean(),
        "tmin_temporada": g["tmin"].mean(),
        "vpd_temporada": g["vpd"].mean(),
        "ws_temporada": g["ws"].mean(),
        "ppt_sazonalidade": g["ppt"].std() / g["ppt"].mean().replace(0, np.nan),
        "ppt_pre_estacao": fora["ppt"].sum(),
        "ppt_transmissao": trans["ppt"].sum(),
        "tmax_transmissao": trans["tmax"].mean(),
        "tmin_transmissao": trans["tmin"].mean(),
        "vpd_transmissao": trans["vpd"].mean(),
    }).reset_index()

    # Chuva do trimestre mais seco da temporada.
    m = m.sort_values(["cod_mun", "temporada", "ordinal"])
    tri = (m.groupby(["cod_mun", "temporada"])["ppt"]
           .rolling(3, min_periods=3).sum().groupby(level=[0, 1]).min()
           .reset_index(name="ppt_trimestre_seco"))
    return anual.merge(tri, on=["cod_mun", "temporada"], how="left")


def construir() -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()

    epi = pd.read_parquet(cfg.PROCESSADO / "epi_temporada.parquet")
    prop = pd.read_parquet(cfg.PROCESSADO / "epidemia_temporada.parquet")
    painel = epi.merge(prop, on=["cod_mun", "temporada"], how="left")

    # ------------------------------------------------------------- clima
    clima = _clima_por_temporada()
    vars_clima = [c for c in clima.columns if c not in ("cod_mun", "temporada")]

    # Normal climatologica: media do municipio sobre toda a serie. E a
    # adequabilidade estrutural, e nao muda de temporada para temporada.
    normal = clima.groupby("cod_mun")[vars_clima].mean()
    normal.columns = [f"{c}_normal" for c in vars_clima]
    painel = painel.merge(normal.reset_index(), on="cod_mun", how="left")

    # Clima recente: as duas temporadas anteriores. Defasado porque o clima da
    # temporada a prever ainda nao aconteceu quando a previsao e feita.
    clima = clima.sort_values(["cod_mun", "temporada"])
    for k in (1, 2):
        desl = clima.copy()
        desl["temporada"] = desl["temporada"] + k
        desl = desl.rename(columns={c: f"{c}_t{k}" for c in vars_clima})
        painel = painel.merge(desl, on=["cod_mun", "temporada"], how="left")

    # Anomalia da temporada anterior em relacao a normal do proprio municipio.
    for c in vars_clima:
        painel[f"{c}_anomalia"] = painel[f"{c}_t1"] - painel[f"{c}_normal"]

    # ---------------------------------------------------------- paisagem
    amb = pd.read_parquet(
        cfg.PROCESSADO / "painel_analitico.parquet",
        columns=["cod_mun", "ano", "cobertura_florestal", "perda_ano_km2",
                 "perda_acumulada_km2", "populacao", "densidade_hab_km2",
                 "area_km2", "populacao_log", "uf", "cod_uf"])
    # Ano civil t corresponde a temporada t: a temporada t comeca em julho de t,
    # de modo que o estado da paisagem no ano civil t e o que vigora quando ela
    # comeca. Ainda assim entra defasado de uma temporada, porque em junho de t o
    # dado anual de t nao esta publicado.
    amb = amb.rename(columns={"ano": "temporada"})
    estatico = amb[["cod_mun", "uf", "cod_uf", "area_km2"]].drop_duplicates("cod_mun")
    dinamico = ["cobertura_florestal", "perda_ano_km2", "perda_acumulada_km2",
                "populacao", "densidade_hab_km2", "populacao_log"]
    desl = amb[["cod_mun", "temporada"] + dinamico].copy()
    desl["temporada"] = desl["temporada"] + 1
    desl = desl.rename(columns={c: f"{c}_t1" for c in dinamico})
    painel = painel.merge(desl, on=["cod_mun", "temporada"], how="left")
    painel = painel.merge(estatico, on="cod_mun", how="left")

    # ------------------------------------------------------------ relevo
    relevo = pd.read_parquet(cfg.PROCESSADO / "relevo_muni.parquet")
    painel = painel.merge(relevo, on="cod_mun", how="left")

    macro = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}
    painel["regiao"] = (painel["cod_uf"].astype(int) // 10).map(macro)

    # ------------------------------------------------------------ grupos
    prop_cols = [c for c in prop.columns if c not in ("cod_mun", "temporada")]
    suscet = ["temporadas_desde_evento", "eventos_locais_3t", "eventos_locais_5t",
              "eventos_locais_10t", "evento_temporada_anterior",
              "epizootia_temporada_anterior"]
    GRUPOS.clear()
    GRUPOS.update({
        "PROPAGACAO": [c for c in prop_cols if c not in suscet],
        "SUSCETIBILIDADE": suscet,
        "CLIMA_NORMAL": [f"{c}_normal" for c in vars_clima],
        "CLIMA_RECENTE": [f"{c}_t1" for c in vars_clima]
                         + [f"{c}_t2" for c in vars_clima],
        "CLIMA_ANOMALIA": [f"{c}_anomalia" for c in vars_clima],
        "PAISAGEM": ["cobertura_florestal_t1", "perda_ano_km2_t1",
                     "perda_acumulada_km2_t1"],
        "RELEVO": ["altitude_media", "altitude_min", "altitude_max",
                   "altitude_desvio", "amplitude_altimetrica"],
        "DEMOGRAFIA": ["populacao_log_t1", "densidade_hab_km2_t1", "area_km2"],
    })

    painel = painel.sort_values(["cod_mun", "temporada"]).reset_index(drop=True)
    painel.to_parquet(SAIDA, index=False)

    todas = [c for cols in GRUPOS.values() for c in cols]
    faltando = [c for c in todas if c not in painel.columns]
    ausentes = painel[todas].isna().mean().sort_values(ascending=False)

    diag = {
        "linhas": int(len(painel)),
        "municipios": int(painel["cod_mun"].nunique()),
        "temporadas": [int(painel["temporada"].min()),
                       int(painel["temporada"].max())],
        "grupos": {g: len(c) for g, c in GRUPOS.items()},
        "total_variaveis": len(todas),
        "variaveis_declaradas_ausentes_do_painel": faltando,
        "prevalencia": {
            "yf_event": float(painel["yf_event"].mean()),
            "caso_humano": float(painel["caso_humano"].mean()),
            "epizootia": float(painel["epizootia"].mean())},
        "eventos_por_temporada": {
            int(t): int(v) for t, v in
            painel.groupby("temporada")["yf_event"].sum().items()},
        "fracao_ausente": {k: round(float(v), 4)
                           for k, v in ausentes.head(12).items()},
    }
    DIAGNOSTICO.write_text(json.dumps(diag, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return painel, diag


def carregar() -> pd.DataFrame:
    if not SAIDA.exists():
        construir()
    return pd.read_parquet(SAIDA)


def grupos() -> dict[str, list[str]]:
    """Os grupos, recalculados sem refazer o painel."""
    if not GRUPOS:
        construir()
    return dict(GRUPOS)


if __name__ == "__main__":
    p, d = construir()
    print(f"painel de modelagem: {len(p):,} linhas x {d['total_variaveis']} variaveis")
    print(f"temporadas {d['temporadas'][0]}-{d['temporadas'][1]}, "
          f"{d['municipios']:,} municipios")
    for g, n in d["grupos"].items():
        print(f"  {g:16s} {n:3d}")
    print(f"\nprevalencia: yf_event {d['prevalencia']['yf_event']:.4%}, "
          f"caso humano {d['prevalencia']['caso_humano']:.4%}, "
          f"epizootia {d['prevalencia']['epizootia']:.4%}")
    if d["variaveis_declaradas_ausentes_do_painel"]:
        print("AUSENTES:", d["variaveis_declaradas_ausentes_do_painel"])
    print("\nmaior fracao ausente:")
    for k, v in list(d["fracao_ausente"].items())[:6]:
        print(f"  {k:34s} {v:.1%}")
