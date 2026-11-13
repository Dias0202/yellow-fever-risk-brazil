"""Populacao municipal anual, a partir das estimativas do IBGE.

Duas armadilhas neste conjunto, ambas capazes de corromper o painel em silencio.

Armadilha 1 - o formato do codigo muda entre os anos
-----------------------------------------------------
Nas planilhas antigas (1994, 2000) a coluna COD. MUNIC traz o codigo sequencial de
quatro digitos, SEM digito verificador: Alta Floresta D'Oeste aparece como 1.
Nas recentes (2020) traz os cinco digitos COM verificador: a mesma cidade aparece
como 00015. Concatenar cegamente com a UF produz 1100001 num caso e 1100015 no
outro - dois codigos diferentes para o mesmo municipio, e so o segundo existe.

O modulo detecta o formato de cada arquivo testando as duas interpretacoes contra
a referencia municipal e adotando a de maior taxa de correspondencia, registrando
a decisao e a taxa no diagnostico.

Armadilha 2 - anos ausentes
----------------------------
Nao ha planilha para 1996, 2007, 2010, 2022 e 2024. Sao anos de Contagem ou de
Censo, em que o IBGE publica em outro formato. Os valores sao interpolados
linearmente entre os anos vizinhos e marcados como interpolados, nunca
apresentados como observados.
"""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd

from yfvrisk import config as cfg
from yfvrisk.ingest.municipios import carregar_referencia

DIRETORIO = cfg.FONTE_MANUAL / "População ibge"


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", t.lower())


SIGLAS_UF = {
    "RO", "AC", "AM", "RR", "PA", "AP", "TO", "MA", "PI", "CE", "RN", "PB", "PE",
    "AL", "SE", "BA", "MG", "ES", "RJ", "SP", "PR", "SC", "RS", "MS", "MT", "GO", "DF",
}


def _para_inteiro(coluna: pd.Series) -> pd.Series:
    """Converte texto do IBGE para inteiro: separador de milhar e nota de rodape."""
    return pd.to_numeric(
        coluna.astype(str)
        .str.replace(r"\(.*?\)", "", regex=True)   # "12.345 (1)"
        .str.replace(".", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce")


def _ler_planilha(caminho) -> pd.DataFrame:
    """Le uma planilha do IBGE identificando as colunas por CONTEUDO.

    As planilhas mudam de layout ao longo dos anos: cabecalho em uma ou duas
    linhas, coluna de populacao chamada 'POPULACAO ESTIMADA', 'POP-2001' ou
    'POPULACAO', coluna de nome chamada 'MUNICIPIOS' ou 'NOMES'. Casar texto de
    cabecalho quebra em oito dos vinte e sete arquivos.

    A identificacao por conteudo e estavel: a coluna de UF e a que contem siglas
    validas, a de codigo de UF a que so tem inteiros entre 11 e 53, e a de
    populacao a coluna numerica de maior magnitude.
    """
    bruto = pd.read_excel(caminho, sheet_name=0, header=None, dtype=str)

    # Linhas de dados sao as que trazem uma sigla de UF valida em alguma coluna.
    marca = bruto.apply(
        lambda col: col.astype(str).str.strip().str.upper().isin(SIGLAS_UF))
    linhas_dados = marca.any(axis=1)
    if not linhas_dados.any():
        raise ValueError(f"nenhuma linha com sigla de UF em {caminho.name}")

    dados = bruto[linhas_dados].copy()

    numericas = {c: _para_inteiro(dados[c]) for c in dados.columns}
    validas = {c: s for c, s in numericas.items() if s.notna().mean() > 0.9}
    if len(validas) < 3:
        raise ValueError(f"colunas numericas insuficientes em {caminho.name}")

    # Codigo de UF: inteiros no intervalo oficial 11-53.
    col_uf = next(
        (c for c, s in validas.items()
         if s.dropna().between(11, 53).mean() > 0.99 and s.dropna().max() <= 53),
        None)
    if col_uf is None:
        raise ValueError(f"coluna de codigo de UF nao identificada em {caminho.name}")

    # Populacao: a coluna numerica de maior mediana entre as restantes.
    restantes = {c: s for c, s in validas.items() if c != col_uf}
    col_pop = max(restantes, key=lambda c: restantes[c].median())

    # Codigo do municipio: a coluna numerica restante de maior mediana.
    sobra = {c: s for c, s in restantes.items() if c != col_pop}
    if not sobra:
        raise ValueError(f"coluna de codigo municipal nao identificada em {caminho.name}")
    col_mun = max(sobra, key=lambda c: sobra[c].median())

    df = pd.DataFrame({
        "cod_uf": validas[col_uf].astype("Int64").astype(str).str.zfill(2),
        "cod_munic": sobra[col_mun].astype("Int64").astype(str),
        "populacao": restantes[col_pop],
    }).dropna()
    df = df[df["populacao"] > 0]
    df["populacao"] = df["populacao"].astype(np.int64)
    return df


def _resolver_codigo(df: pd.DataFrame, validos7: set, prefixo6: dict):
    """Testa as duas convencoes de codigo e adota a de maior correspondencia."""
    # Convencao moderna: UF + cinco digitos com verificador.
    cod_moderno = df["cod_uf"] + df["cod_munic"].str.zfill(5)
    acerto_moderno = cod_moderno.isin(validos7).mean()

    # Convencao antiga: UF + quatro digitos sem verificador, resolvida via prefixo.
    cod_antigo6 = df["cod_uf"] + df["cod_munic"].str.zfill(4).str[-4:]
    cod_antigo = cod_antigo6.map(prefixo6)
    acerto_antigo = cod_antigo.notna().mean()

    if acerto_moderno >= acerto_antigo:
        return cod_moderno, "moderno_7_digitos", float(acerto_moderno)
    return cod_antigo, "antigo_4_digitos_sem_verificador", float(acerto_antigo)


def construir() -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()
    ref = carregar_referencia(com_geometria=False)
    validos7 = set(ref["cod_mun"])
    prefixo6 = {c[:6]: c for c in ref["cod_mun"]}

    registros, por_arquivo = [], []
    for caminho in sorted(DIRETORIO.glob("*.xls*")):
        ano_txt = caminho.stem
        if not ano_txt.isdigit():
            continue  # ignora planilhas cujo nome nao e um ano
        ano = int(ano_txt)

        try:
            df = _ler_planilha(caminho)
        except Exception as e:
            por_arquivo.append({"arquivo": caminho.name, "erro": f"{type(e).__name__}: {e}"})
            continue

        cod, convencao, acerto = _resolver_codigo(df, validos7, prefixo6)
        df = df.assign(cod_mun=cod, ano=ano).dropna(subset=["cod_mun"])
        df = df[df["cod_mun"].isin(validos7)]

        registros.append(df[["cod_mun", "ano", "populacao"]])
        por_arquivo.append({
            "arquivo": caminho.name, "ano": ano, "convencao": convencao,
            "taxa_correspondencia": round(acerto, 4),
            "municipios_aproveitados": int(len(df)),
        })

    observado = pd.concat(registros, ignore_index=True)
    observado = observado.drop_duplicates(subset=["cod_mun", "ano"])

    anos = range(cfg.ANO_INICIO, cfg.ANO_FIM + 1)
    painel = pd.MultiIndex.from_product(
        [ref["cod_mun"], anos], names=["cod_mun", "ano"]).to_frame(index=False)
    painel = painel.merge(observado, on=["cod_mun", "ano"], how="left")
    painel["populacao_observada"] = painel["populacao"].notna().astype(int)

    painel = painel.sort_values(["cod_mun", "ano"])
    painel["populacao"] = (painel.groupby("cod_mun", sort=False)["populacao"]
                           .transform(lambda s: s.interpolate(limit_direction="both")))
    painel["populacao"] = painel["populacao"].round().astype("Int64")

    # A area equal-area vem da referencia ja construida; a leitura sem geometria
    # so traz o atributo do IBGE.
    areas = pd.read_parquet(cfg.PROCESSADO / "municipios_referencia.parquet",
                            columns=["cod_mun", "area_km2"])
    painel = painel.merge(areas, on="cod_mun", how="left")
    painel["densidade_hab_km2"] = painel["populacao"] / painel["area_km2"]
    painel = painel.reset_index(drop=True)

    anos_observados = sorted(observado["ano"].unique().tolist())
    diag = {
        "arquivos": por_arquivo,
        "anos_com_planilha": anos_observados,
        "anos_interpolados": [a for a in anos if a not in anos_observados],
        "linhas": int(len(painel)),
        "populacao_observada_pct": float(painel["populacao_observada"].mean()),
        "populacao_ausente_apos_interpolacao": int(painel["populacao"].isna().sum()),
        "populacao_total_por_ano": {
            int(a): int(g["populacao"].sum())
            for a, g in painel.groupby("ano") if a in (1994, 2000, 2010, 2020, 2025)
        },
    }

    painel.to_parquet(cfg.PROCESSADO / "populacao_muni_ano.parquet", index=False)
    (cfg.METRICAS / "f2_populacao_diagnostico.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    return painel, diag


if __name__ == "__main__":
    p, d = construir()
    print(f"populacao: {len(p):,} linhas")
    print(f"observada em {d['populacao_observada_pct']:.1%} das linhas")
    print(f"anos interpolados: {d['anos_interpolados']}")
    print(f"ausentes apos interpolacao: {d['populacao_ausente_apos_interpolacao']}")
    print("\nconvencao de codigo detectada por arquivo:")
    for r in d["arquivos"]:
        if "erro" in r:
            print(f"  {r['arquivo']:16s} ERRO {r['erro']}")
        else:
            print(f"  {r['arquivo']:16s} {r['convencao']:34s} "
                  f"acerto {r['taxa_correspondencia']:.3f}  "
                  f"n={r['municipios_aproveitados']}")
    print("\npopulacao total do Brasil por ano:")
    for a, v in d["populacao_total_por_ano"].items():
        print(f"  {a}: {v:,}")
