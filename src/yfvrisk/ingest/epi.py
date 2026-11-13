"""Painel epidemiologico anual: casos humanos e epizootias em primatas nao humanos.

A resolucao e ANUAL, nao agregada em periodos plurianuais de largura desigual.
Periodos largos que aparecem dos dois lados de qualquer particao deixam o modelo
interpolar no tempo em vez de prever; com resolucao anual a validacao pode ser
prospectiva, treinando so com o passado.

Alem disso, casos humanos e epizootias sao mantidos SEPARADOS. Agrega-los num
unico desfecho e o que faz a vacinacao parecer inocua: vacinar humanos nao muda a
probabilidade de epizootia em primatas, que permanecem plenamente suscetiveis.

Tratamento da lacuna de vigilancia
----------------------------------
A serie de epizootias comeca em 2000, nao em 1994, e nao tem NENHUM registro
nacional em 2004-2006 e 2010-2013. Sao lacunas de vigilancia, nao ausencia de
circulacao viral. Preencher tudo com zero por produto cartesiano ensinaria ao
modelo que aqueles anos foram livres de virus.

Aqui cada linha carrega marcadores explicitos do que e zero observado e do que e
ausencia de observacao, para que a modelagem possa tratar os dois de forma
diferente.
"""

from __future__ import annotations

import pandas as pd

from yfvrisk import config as cfg
from yfvrisk.ingest.municipios import carregar_referencia

ARQ_HUMANOS = cfg.FONTE_MANUAL / "fa_casoshumanos_1994-2025.csv"
ARQ_EPIZOOTIAS = cfg.FONTE_MANUAL / "fa_epizpnh_1994-2025.csv"


def _mapa_seis_para_sete(ref: pd.DataFrame) -> dict[str, str]:
    """Os arquivos do Ministerio da Saude usam codigo IBGE de seis digitos."""
    m = {c[:6]: c for c in ref["cod_mun"]}
    if len(m) != len(ref):
        raise ValueError("prefixos de seis digitos nao sao unicos na referencia")
    return m


def _ler(arquivo, col_mun, col_ano, ref, rotulo):
    df = pd.read_csv(arquivo, sep=";", encoding="latin-1")
    n_bruto = len(df)

    sem_mun = int(df[col_mun].isna().sum())
    df = df.dropna(subset=[col_mun]).copy()

    df["cod6"] = df[col_mun].astype(float).astype(int).astype(str).str.zfill(6)
    df["ano"] = df[col_ano].astype(int)

    mapa = _mapa_seis_para_sete(ref)
    df["cod_mun"] = df["cod6"].map(mapa)

    orfaos = df[df["cod_mun"].isna()]
    cods_orfaos = sorted(orfaos["cod6"].unique())
    df = df.dropna(subset=["cod_mun"])

    agregado = (df.groupby(["cod_mun", "ano"]).size()
                .reset_index(name=f"n_{rotulo}"))

    diag = {
        "arquivo": arquivo.name,
        "registros_brutos": n_bruto,
        "sem_codigo_municipal": sem_mun,
        "codigos_nao_encontrados_na_malha": cods_orfaos,
        "registros_descartados_por_codigo_orfao": int(len(orfaos)),
        "registros_aproveitados": int(len(df)),
        "ano_min": int(df["ano"].min()),
        "ano_max": int(df["ano"].max()),
        "municipios_afetados": int(df["cod_mun"].nunique()),
    }
    return agregado, diag


def construir() -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()
    ref = carregar_referencia(com_geometria=False)

    humanos, diag_h = _ler(ARQ_HUMANOS, "COD_MUN_LPI", "ANO_IS", ref, "casos_humanos")
    epiz, diag_e = _ler(ARQ_EPIZOOTIAS, "COD_MUN_OCOR", "ANO_OCOR", ref, "epizootias")

    anos = range(cfg.ANO_INICIO, cfg.ANO_FIM + 1)
    painel = pd.MultiIndex.from_product(
        [ref["cod_mun"], anos], names=["cod_mun", "ano"]).to_frame(index=False)

    painel = painel.merge(humanos, on=["cod_mun", "ano"], how="left")
    painel = painel.merge(epiz, on=["cod_mun", "ano"], how="left")
    painel[["n_casos_humanos", "n_epizootias"]] = \
        painel[["n_casos_humanos", "n_epizootias"]].fillna(0).astype(int)

    painel["caso_humano"] = (painel["n_casos_humanos"] > 0).astype(int)
    painel["epizootia"] = (painel["n_epizootias"] > 0).astype(int)
    painel["yf_event"] = ((painel["caso_humano"] == 1)
                          | (painel["epizootia"] == 1)).astype(int)

    # --- Marcadores de observabilidade -------------------------------------
    # Distinguem zero observado de ausencia de observacao. Nenhum modelo deve
    # tratar as duas coisas como a mesma evidencia.
    anos_sem_registro = set(
        a for a in anos
        if a >= cfg.ANO_INICIO_EPIZOOTIAS
        and int(painel.loc[painel["ano"] == a, "n_epizootias"].sum()) == 0
    )
    painel["epizootia_observavel"] = (
        (painel["ano"] >= cfg.ANO_INICIO_EPIZOOTIAS)
        & (~painel["ano"].isin(anos_sem_registro))
    ).astype(int)

    # Desfecho restrito ao que era efetivamente observavel naquele ano.
    painel["yf_event_observavel"] = painel["caso_humano"].where(
        painel["epizootia_observavel"] == 0, painel["yf_event"]).astype(int)

    diagnostico = {
        "humanos": diag_h,
        "epizootias": diag_e,
        "linhas_painel": int(len(painel)),
        "municipios": int(painel["cod_mun"].nunique()),
        "anos": [cfg.ANO_INICIO, cfg.ANO_FIM],
        "anos_sem_registro_nacional_de_epizootia": sorted(anos_sem_registro),
        "prevalencia_yf_event": float(painel["yf_event"].mean()),
        "prevalencia_caso_humano": float(painel["caso_humano"].mean()),
        "prevalencia_epizootia": float(painel["epizootia"].mean()),
    }

    painel.to_parquet(cfg.PROCESSADO / "painel_epi_muni_ano.parquet", index=False)
    return painel, diagnostico


if __name__ == "__main__":
    import json

    p, d = construir()
    print(json.dumps(d, indent=2, ensure_ascii=False))
    print()
    print(f"painel: {len(p)} linhas = {p['cod_mun'].nunique()} municipios "
          f"x {p['ano'].nunique()} anos")
    resumo = p.groupby("ano")[["caso_humano", "epizootia", "yf_event"]].sum()
    print("\nmunicipios com evento, por ano:")
    print(resumo.to_string())
