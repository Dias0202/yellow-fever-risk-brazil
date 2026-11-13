"""Montagem do painel analitico municipio-ano.

Junta os componentes construidos na ingestao e deriva as variaveis de modelagem.

Tres decisoes de desenho
------------------------

1. Resolucao anual, nao periodos plurianuais de largura desigual. Com periodos
   largos presentes nos dois lados de qualquer particao aleatoria, o modelo
   nunca precisa extrapolar e interpola no tempo em vez de prever. Com
   resolucao anual, a validacao pode retirar do treino tudo que vem depois do
   corte.

2. Precipitacao anual e SOMA dos meses, nao media. Uma "precipitacao anual" de
   112 mm e na verdade uma media mensal - o acumulado anual brasileiro fica
   entre 1.000 e 3.000 mm.

3. Defasagens explicitas. Clima do ano anterior e do retrasado entram como
   variaveis proprias, porque a resposta do vetor e do hospedeiro a chuva e
   temperatura nao e instantanea.

Variaveis de historico
----------------------
`evento_ano_anterior`, `eventos_ultimos_5_anos` e `anos_desde_ultimo_evento` sao
legitimas para previsao de curto prazo, e sao a referencia honesta a ser batida.
Mas nao existem para uma projecao de 2021-2040, porque dependem do desfecho que
se quer projetar. Ficam separadas no grupo HISTORICO para que
o modelo possa ser avaliado com e sem elas.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yfvrisk import config as cfg

CLIMA_BASE = ["ppt", "tmax", "tmin", "vpd", "ws"]
DEFASAGENS = [1, 2]

# Variaveis climaticas que entram defasadas. Declaradas aqui, e nao dentro de
# construir(), porque GRUPOS precisa estar completo no momento do import: um
# script que faz "from painel import GRUPOS" sem construir o painel recebia
# CLIMA_DEFASADO vazio e treinava sem as oito defasagens, silenciosamente.
BASES_DEFASADAS = ["ppt_total", "tmax_medio", "vpd_medio", "ppt_trimestre_seco"]

# Grupos de variaveis, para que o modelo possa ser montado por grupo e a
# contribuicao de cada um medida separadamente.
GRUPOS = {
    "CLIMA": ["ppt_total", "tmax_medio", "tmin_medio", "vpd_medio", "ws_medio",
              "ppt_sazonalidade", "ppt_trimestre_seco", "tmax_mes_mais_quente",
              "amplitude_termica"],
    "CLIMA_DEFASADO": [f"{base}_lag{k}"
                       for base in BASES_DEFASADAS for k in DEFASAGENS],
    "PAISAGEM": ["floresta_frac", "silvicultura_frac", "pastagem_frac",
                 "urbano_frac", "areas_umidas_frac", "savana_frac",
                 "mosaico_agropecuario_frac", "interface_floresta_antropico",
                 "cobertura_florestal", "perda_ano_km2", "perda_acumulada_km2"],
    # Uma variavel so, e defasada em 2 anos. Tres decisoes, todas medidas.
    #
    # `imunidade_acumulada` sai por estar a jusante do desfecho: a cobertura do
    # PNI e registrada no ano do surto, em resposta a ele - salta 4,5x em k = 0 e
    # continua elevada em k = +1. `suscetivel_pct` sai por ser exatamente o seu
    # complemento, duas colunas colineares dividindo a mesma importancia.
    #
    # `imunidade_relativa2` e `anos_com_registro_pni` ficam no painel, e fora do
    # modelo. O criterio nao e desempenho, e admissibilidade para o uso pretendido:
    # as duas sao construtos sobre o processo de REGISTRO, nao sobre o estado do
    # municipio, e sob projecao teriam de ser congeladas - viram entao um codigo
    # de identidade do municipio, isto e, uma variavel memorizavel. Congelar
    # cobertura florestal e uma hipotese substantiva; congelar "anos de presenca
    # na serie do PNI" nao significa nada.
    # A medicao concorda: no corte de 2019 as duas sao #1 e #2 no MDI e #14 e #28
    # na permutacao, com IC cruzando zero. Sao memorizacao.
    #
    # Ver ingest/vacinacao.py e analysis/f5_00_diagnostico_vacinacao.py.
    "IMUNIDADE": ["imunidade_defasada2"],
    "DEMOGRAFIA": ["populacao_log", "densidade_hab_km2", "area_km2"],
    "HISTORICO": ["evento_ano_anterior", "eventos_ultimos_5_anos",
                  "anos_desde_ultimo_evento"],
}


def agregar_clima_anual(m: pd.DataFrame) -> pd.DataFrame:
    """Agrega clima mensal para o ano, preservando informacao de sazonalidade.

    Recebe o mensal em vez de le-lo do disco para que o modulo de cenarios derive
    as variaveis anuais do clima FUTURO exatamente com estas definicoes. Se o
    futuro fosse derivado por outro caminho (ex.: bioclimaticas `Bio1...Bio15`
    com outra convencao), as colunas projetadas poderiam nao ser as que o modelo
    le, e um cenario so de clima sairia identico a linha de base sem nenhum
    erro visivel. Com uma funcao so, a divergencia deixa de ser possivel.

    Espera colunas cod_mun, ano, mes, ppt, tmax, tmin, vpd, ws.
    """
    g = m.groupby(["cod_mun", "ano"])
    anual = pd.DataFrame({
        # Soma, nao media: e o acumulado anual que tem sentido hidrologico.
        "ppt_total": g["ppt"].sum(),
        "tmax_medio": g["tmax"].mean(),
        "tmin_medio": g["tmin"].mean(),
        "vpd_medio": g["vpd"].mean(),
        "ws_medio": g["ws"].mean(),
        # Coeficiente de variacao mensal da chuva: distingue regime sazonal de
        # regime uniforme com o mesmo total anual.
        "ppt_sazonalidade": g["ppt"].std() / g["ppt"].mean().replace(0, np.nan),
        "tmax_mes_mais_quente": g["tmax"].max(),
        "amplitude_termica": g["tmax"].mean() - g["tmin"].mean(),
    }).reset_index()

    # Precipitacao do trimestre mais seco: limita a sobrevivencia de larvas em
    # criadouros naturais e e a variavel bioclimatica classica para arboviroses.
    m = m.sort_values(["cod_mun", "ano", "mes"])
    tri = (m.groupby(["cod_mun", "ano"])["ppt"]
           .rolling(3, min_periods=3).sum().groupby(level=[0, 1]).min()
           .reset_index(name="ppt_trimestre_seco"))
    return anual.merge(tri, on=["cod_mun", "ano"], how="left")


def _clima_anual() -> pd.DataFrame:
    """O clima anual observado, do TerraClimate ja agregado por municipio e mes."""
    return agregar_clima_anual(
        pd.read_parquet(cfg.PROCESSADO / "clima_muni_mes.parquet"))


def _historico(epi: pd.DataFrame) -> pd.DataFrame:
    """Variaveis derivadas do proprio desfecho em anos anteriores.

    Todas usam APENAS o passado: shift(1) antes de qualquer acumulacao. Nenhuma
    linha enxerga o proprio ano nem os seguintes.
    """
    d = epi[["cod_mun", "ano", "yf_event"]].sort_values(["cod_mun", "ano"]).copy()
    g = d.groupby("cod_mun", sort=False)["yf_event"]

    d["evento_ano_anterior"] = g.shift(1).fillna(0).astype(int)
    d["eventos_ultimos_5_anos"] = (
        g.shift(1).rolling(5, min_periods=1).sum()
        .reset_index(level=0, drop=True).fillna(0).astype(int))

    # Anos desde o ultimo evento, com teto para municipios que nunca tiveram.
    def _desde(s: pd.Series) -> pd.Series:
        desde, contador = [], 99
        for v in s.shift(1).fillna(0):
            contador = 0 if v == 1 else min(contador + 1, 99)
            desde.append(contador)
        return pd.Series(desde, index=s.index)

    d["anos_desde_ultimo_evento"] = g.transform(_desde)
    return d.drop(columns="yf_event")


def construir() -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()

    epi = pd.read_parquet(cfg.PROCESSADO / "painel_epi_muni_ano.parquet")
    painel = epi.copy()

    clima = _clima_anual()
    painel = painel.merge(clima, on=["cod_mun", "ano"], how="left")

    # Defasagens do clima: a resposta ecologica a chuva e temperatura nao e
    # instantanea.
    painel = painel.sort_values(["cod_mun", "ano"])
    for base in BASES_DEFASADAS:
        for k in DEFASAGENS:
            painel[f"{base}_lag{k}"] = (
                painel.groupby("cod_mun", sort=False)[base].shift(k))

    vac = pd.read_parquet(cfg.PROCESSADO / "vacinacao_muni_ano.parquet")
    painel = painel.merge(
        vac[["cod_mun", "ano", "imunidade_acumulada", "suscetivel_pct",
             "imunidade_defasada2", "imunidade_relativa2",
             "anos_com_registro_pni", "cobertura_disponivel"]],
        on=["cod_mun", "ano"], how="left")

    pop = pd.read_parquet(cfg.PROCESSADO / "populacao_muni_ano.parquet")
    painel = painel.merge(
        pop[["cod_mun", "ano", "populacao", "densidade_hab_km2", "area_km2",
             "populacao_observada"]], on=["cod_mun", "ano"], how="left")
    painel["populacao_log"] = np.log1p(painel["populacao"].astype(float))

    solo = pd.read_parquet(cfg.PROCESSADO / "uso_solo_muni_ano.parquet")
    painel = painel.merge(solo.drop(columns=[c for c in ["uso_solo_observado"]
                                             if c in solo.columns], errors="ignore"),
                          on=["cod_mun", "ano"], how="left")

    caminho_floresta = cfg.PROCESSADO / "floresta_muni_ano.parquet"
    if caminho_floresta.exists():
        flo = pd.read_parquet(caminho_floresta)
        painel = painel.merge(
            flo[["cod_mun", "ano", "cobertura_florestal", "perda_ano_km2",
                 "perda_acumulada_km2", "hansen_cobertura"]],
            on=["cod_mun", "ano"], how="left")

    painel = painel.merge(_historico(epi), on=["cod_mun", "ano"], how="left")

    ref = pd.read_parquet(cfg.PROCESSADO / "municipios_referencia.parquet",
                          columns=["cod_mun", "uf", "cod_uf", "nome_mun"])
    painel = painel.merge(ref, on="cod_mun", how="left")
    painel = painel.sort_values(["cod_mun", "ano"]).reset_index(drop=True)

    disponiveis = {g: [c for c in cols if c in painel.columns]
                   for g, cols in GRUPOS.items()}
    todas = [c for cols in disponiveis.values() for c in cols]

    diag = {
        "linhas": int(len(painel)),
        "municipios": int(painel["cod_mun"].nunique()),
        "anos": [int(painel["ano"].min()), int(painel["ano"].max())],
        "grupos_de_variaveis": {g: len(c) for g, c in disponiveis.items()},
        "variaveis": disponiveis,
        "total_de_variaveis": len(todas),
        "prevalencia": {
            "yf_event": float(painel["yf_event"].mean()),
            "caso_humano": float(painel["caso_humano"].mean()),
            "epizootia": float(painel["epizootia"].mean()),
        },
        "ausentes_por_variavel": {
            c: int(painel[c].isna().sum()) for c in todas
            if painel[c].isna().any()
        },
    }

    painel.to_parquet(cfg.PROCESSADO / "painel_analitico.parquet", index=False)
    (cfg.METRICAS / "f2_painel_analitico.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    return painel, diag


if __name__ == "__main__":
    p, d = construir()
    print(f"painel analitico: {d['linhas']:,} linhas x {d['total_de_variaveis']} variaveis")
    print(f"{d['municipios']} municipios, anos {d['anos'][0]}-{d['anos'][1]}\n")
    for g, n in d["grupos_de_variaveis"].items():
        print(f"  {g:16s} {n:3d} variaveis")
    print("\nprevalencia anual:")
    for k, v in d["prevalencia"].items():
        print(f"  {k:14s} {v:.4%}")
    if d["ausentes_por_variavel"]:
        print("\nvariaveis com ausentes:")
        for c, n in sorted(d["ausentes_por_variavel"].items(),
                           key=lambda x: -x[1])[:12]:
            print(f"  {c:32s} {n:7,d} ({100*n/d['linhas']:.1f}%)")
