"""Uso e cobertura do solo por municipio e ano.

A armadilha
-----------
O conjunto derivado do MapBiomas traz colunas `X-mean_normalized` que parecem
fracoes de area, mas estao na ordem de 1e-4 e praticamente sem variancia
("0,00 +/- 0,00" numa tabela de estatisticas). Usadas como variavel, nao movem
o modelo, e qualquer cenario de mudanca de uso construido sobre elas fica dentro
do ruido.

A fracao certa
--------------
O conjunto traz, para cada uma das 25 classes, uma coluna `X-count` com o numero
de pixels da classe no municipio. A fracao da area municipal e simplesmente

    fracao_da_classe = X-count / soma de todos os counts

Medido em 2020, a diferenca e de tres ordens de grandeza:

    Forest_formation   fracao por count:    mediana 0,1534, maximo 0,794
                       -mean_normalized:    mediana 1,07e-04
    Pasture            fracao por count:    mediana 0,0947, maximo 0,672
                       -mean_normalized:    mediana 1,71e-04

Com a fracao por count o uso do solo volta a ser uma variavel com amplitude real, e
os cenarios de mudanca de uso passam a ter efeito interpretavel.
"""

from __future__ import annotations

import json

import pandas as pd

from yfvrisk import config as cfg
from yfvrisk.ingest.municipios import carregar_referencia

ARQUIVO = cfg.FONTE_MANUAL / "KrempserData" / "01-stats-1985--2020-land_use_cover-BR.csv"

# Classes relevantes para o ciclo silvestre da febre amarela. As demais entram
# apenas no denominador.
CLASSES = {
    "Forest_formation": "floresta_frac",
    "Forest_plantation": "silvicultura_frac",
    "Pasture": "pastagem_frac",
    "Urban_area": "urbano_frac",
    "Wetlands": "areas_umidas_frac",
    "Savanna_formation": "savana_frac",
    "Mosaic_agriculture_and_pasture": "mosaico_agropecuario_frac",
    "Mangrove": "manguezal_frac",
    "Wooded_restinga": "restinga_arborea_frac",
}


def _validar_classes(colunas_disponiveis: list[str]) -> None:
    """Falha alto se um nome de classe nao existir na fonte.

    Sem esta checagem, um nome errado vira uma coluna de zeros que atravessa o
    pipeline inteiro sem alarme - foi o que aconteceu na primeira versao com
    'Mosaic_of_Uses', que nao existe neste conjunto.
    """
    ausentes = [c for c in CLASSES if f"{c}-count" not in colunas_disponiveis]
    if ausentes:
        raise ValueError(
            f"classes inexistentes na fonte: {ausentes}. "
            f"Disponiveis: {sorted(c[:-6] for c in colunas_disponiveis if c.endswith('-count'))}")


def construir() -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()
    ref = carregar_referencia(com_geometria=False)
    mapa = {c[:6]: c for c in ref["cod_mun"]}

    todas = pd.read_csv(ARQUIVO, nrows=0).columns.tolist()
    counts = [c for c in todas if c.endswith("-count")]
    _validar_classes(todas)

    df = pd.read_csv(ARQUIVO, usecols=["geocode", "datetime"] + counts)
    df["cod6"] = df["geocode"].astype(int).astype(str).str.zfill(6)
    df["cod_mun"] = df["cod6"].map(mapa)
    orfaos = sorted(df.loc[df["cod_mun"].isna(), "cod6"].unique())
    df = df.dropna(subset=["cod_mun"])
    df["ano"] = df["datetime"].astype(int)

    total = df[counts].sum(axis=1)
    saida = pd.DataFrame({"cod_mun": df["cod_mun"], "ano": df["ano"]})
    for classe, nome in CLASSES.items():
        coluna = f"{classe}-count"
        saida[nome] = df[coluna] / total

    # Fragmentacao aproximada da paisagem: interface entre floresta e uso
    # antropico. E onde o contato entre humanos e o ciclo silvestre acontece.
    saida["interface_floresta_antropico"] = (
        saida["floresta_frac"] * (saida["pastagem_frac"] + saida["urbano_frac"]
                                  + saida["mosaico_agropecuario_frac"])
    )

    ano_min, ano_max = int(saida["ano"].min()), int(saida["ano"].max())

    # Painel completo. Antes e depois do periodo da fonte o valor e carregado do
    # ano mais proximo disponivel, marcado como extrapolado.
    anos = range(cfg.ANO_INICIO, cfg.ANO_FIM + 1)
    painel = pd.MultiIndex.from_product(
        [ref["cod_mun"], anos], names=["cod_mun", "ano"]).to_frame(index=False)
    painel = painel.merge(saida, on=["cod_mun", "ano"], how="left")
    painel["uso_solo_observado"] = painel["floresta_frac"].notna().astype(int)

    colunas = list(CLASSES.values()) + ["interface_floresta_antropico"]
    painel = painel.sort_values(["cod_mun", "ano"])
    painel[colunas] = (painel.groupby("cod_mun", sort=False)[colunas]
                       .transform(lambda s: s.ffill().bfill()))
    painel = painel.reset_index(drop=True)

    diag = {
        "arquivo": ARQUIVO.name,
        "classes_no_denominador": len(counts),
        "classes_extraidas": list(CLASSES.keys()),
        "anos_na_fonte": [ano_min, ano_max],
        "anos_extrapolados": sorted(set(anos) - set(range(ano_min, ano_max + 1))),
        "codigos_nao_encontrados_na_malha": orfaos,
        "linhas": int(len(painel)),
        "observado_pct": float(painel["uso_solo_observado"].mean()),
        "estatisticas_2020": {
            nome: {
                "mediana": round(float(painel.loc[painel["ano"] == 2020, nome].median()), 4),
                "maximo": round(float(painel.loc[painel["ano"] == 2020, nome].max()), 4),
                "desvio": round(float(painel.loc[painel["ano"] == 2020, nome].std()), 4),
            } for nome in colunas
        },
        "nota_escala": (
            "As colunas -mean_normalized do conjunto derivado tem mediana "
            "da ordem de 1e-4 e amplitude nula. As fracoes por count tem mediana "
            "de 0,15 para floresta e 0,09 para pastagem, com amplitude de 0 a 0,8."
        ),
    }

    painel.to_parquet(cfg.PROCESSADO / "uso_solo_muni_ano.parquet", index=False)
    (cfg.METRICAS / "f2_uso_solo_diagnostico.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    return painel, diag


if __name__ == "__main__":
    p, d = construir()
    print(f"uso do solo: {len(p):,} linhas, {d['classes_no_denominador']} classes "
          f"no denominador")
    print(f"anos na fonte: {d['anos_na_fonte']}  extrapolados: {d['anos_extrapolados']}")
    print(f"observado em {d['observado_pct']:.1%} das linhas\n")
    print(f"{'variavel':34s} {'mediana':>9s} {'maximo':>9s} {'desvio':>9s}")
    for nome, e in d["estatisticas_2020"].items():
        print(f"{nome:34s} {e['mediana']:9.4f} {e['maximo']:9.4f} {e['desvio']:9.4f}")
