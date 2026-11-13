"""Referencia municipal canonica.

Estabelece a malha unica do projeto — IBGE 2022, EPSG:4674 — e produz a tabela de
referencia contra a qual todo o restante do pipeline e validado.

Tres regras:

  1.  A malha do IBGE traz 5.572 feicoes, das quais duas nao sao municipios:
      Lagoa Mirim (4300001) e Lagoa dos Patos (4300002), corpos d'agua no RS.
      Truncados a seis digitos ambos viram 430000, o que produz uma contagem
      intermediaria de 5.571. Aqui os dois sao removidos explicitamente e o
      resultado e 5.570.

  2.  Area e distancia sao calculadas em CRS projetado, nao sobre lat/lon (que
      daria area em grau quadrado e distancia em graus).

  3.  Nenhum caminho absoluto.

Codigos IBGE sao mantidos com SETE digitos em todo o pipeline. A truncagem a seis
digitos descarta o digito verificador e permite a fusao silenciosa dos dois
corpos d'agua.
"""

from __future__ import annotations

import pandas as pd
import pyogrio

from yfvrisk import config as cfg


def caminho_malha():
    candidatos = sorted(cfg.MALHA_MUNICIPIOS.glob("*.shp"))
    if not candidatos:
        raise FileNotFoundError(f"malha nao encontrada em {cfg.MALHA_MUNICIPIOS}")
    return candidatos[0]


def carregar_referencia(com_geometria: bool = False):
    """Devolve a referencia municipal canonica, com exatamente 5.570 linhas."""
    shp = caminho_malha()

    if com_geometria:
        import geopandas as gpd

        gdf = gpd.read_file(shp, engine="pyogrio")
        gdf["cod_mun"] = gdf["CD_MUN"].astype(str).str.zfill(7)
        gdf = gdf[~gdf["cod_mun"].isin(cfg.NAO_MUNICIPIOS)].copy()

        # Area em projecao EQUAL-AREA. A policonica, usada para distancia, nao
        # preserva area e superestimaria o territorio em 1,13%.
        gdf["area_km2"] = gdf.to_crs(cfg.CRS_AREA).geometry.area / 1e6
        # Atributo oficial do IBGE, mantido para conferencia cruzada.
        gdf["area_km2_ibge"] = gdf["AREA_KM2"]
        gdf = gdf.rename(columns={"NM_MUN": "nome_mun", "SIGLA_UF": "uf"})
        gdf["cod_uf"] = gdf["cod_mun"].str[:2]
        return gdf[["cod_mun", "nome_mun", "uf", "cod_uf", "area_km2",
                    "area_km2_ibge", "geometry"]] \
            .sort_values("cod_mun").reset_index(drop=True)

    df = pyogrio.read_dataframe(shp, read_geometry=False)
    df["cod_mun"] = df["CD_MUN"].astype(str).str.zfill(7)
    df = df[~df["cod_mun"].isin(cfg.NAO_MUNICIPIOS)].copy()
    df = df.rename(columns={"NM_MUN": "nome_mun", "SIGLA_UF": "uf",
                            "AREA_KM2": "area_km2_ibge"})
    df["cod_uf"] = df["cod_mun"].str[:2]
    return df[["cod_mun", "nome_mun", "uf", "cod_uf", "area_km2_ibge"]] \
        .sort_values("cod_mun").reset_index(drop=True)


def validar(df: pd.DataFrame) -> list[str]:
    """Checagens que devem passar para a referencia ser aceita."""
    problemas = []
    if len(df) != cfg.N_MUNICIPIOS:
        problemas.append(f"esperava {cfg.N_MUNICIPIOS} municipios, obteve {len(df)}")
    if df["cod_mun"].duplicated().any():
        problemas.append("ha codigos municipais duplicados")
    if not df["cod_mun"].str.fullmatch(r"\d{7}").all():
        problemas.append("ha codigos que nao tem sete digitos")
    if df["cod_mun"].isin(cfg.NAO_MUNICIPIOS).any():
        problemas.append("corpos d'agua da malha nao foram removidos")
    n_uf = df["uf"].nunique()
    if n_uf != 27:
        problemas.append(f"esperava 27 unidades federativas, obteve {n_uf}")
    if "area_km2" in df and "area_km2_ibge" in df:
        # A area calculada deve bater com o atributo oficial do IBGE dentro de 1%.
        # E o teste que detecta uso acidental de projecao nao equivalente em area.
        desvio = abs(df["area_km2"].sum() - df["area_km2_ibge"].sum())
        desvio /= df["area_km2_ibge"].sum()
        if desvio > 0.01:
            problemas.append(
                f"area calculada difere do atributo do IBGE em {desvio:.2%} "
                "(projecao provavelmente nao e equal-area)")
    return problemas


def construir() -> pd.DataFrame:
    cfg.garantir_diretorios()
    ref = carregar_referencia(com_geometria=True)

    problemas = validar(ref)
    if problemas:
        raise ValueError("referencia municipal invalida: " + "; ".join(problemas))

    # A tabela salva nao carrega geometria: repetir poligonos em cada parquet
    # derivado multiplica o tamanho em disco sem acrescentar informacao.
    tabela = ref.drop(columns="geometry")
    destino = cfg.PROCESSADO / "municipios_referencia.parquet"
    tabela.to_parquet(destino, index=False)

    return tabela


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(cfg.RAIZ / "src"))
    t = construir()
    print(f"referencia municipal: {len(t)} municipios, {t['uf'].nunique()} UFs")
    print(f"area total: {t['area_km2'].sum():,.0f} km2")
    print(t.head().to_string(index=False))
