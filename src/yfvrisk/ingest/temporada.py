"""Painel epidemiologico por TEMPORADA de monitoramento, nao por ano civil.

Agregar por ano civil corta a temporada de transmissao exatamente ao meio.

    casos humanos por mes de inicio de sintomas, 1994-2025

        jan   fev   mar   abr   mai   jun   jul   ago   set   out   nov   dez
       1294   670   438   189    49    25    11     9     9     4    10   167

    dezembro a maio concentra 97,6% dos casos humanos e 76,8% das epizootias.

Dezembro e janeiro da MESMA onda caem em unidades diferentes sob ano civil. Sob
validacao temporal, isso e pior do que parece: a metade de dezembro de uma
epidemia pode estar no treino e a metade de janeiro no teste, o que e vazamento;
ou o inverso, o que destroi o sinal que o modelo deveria aprender.

A fonte ja resolve isso
-----------------------
Os dois arquivos do Ministerio da Saude trazem o campo `MONITORAMENTO`, que e a
temporada de vigilancia julho-junho definida pelo proprio programa - "1994/1995",
"1995/1996" e assim por diante. Esta completo em 100% dos registros das duas
bases.

Aqui ele passa a ser a unidade de analise. A temporada e identificada pelo ano de
INICIO: a temporada "2016/2017", que e a da onda da Mata Atlantica, recebe o
rotulo 2016.

Preferencia pela data sobre o campo declarado
---------------------------------------------
Quando ha data de inicio de sintomas ou de ocorrencia, a temporada e derivada
dela, e nao do campo `MONITORAMENTO`. Os dois concordam em quase toda parte, e a
divergencia entra no diagnostico - mas a data e o dado primario e o campo e uma
classificacao administrativa, que pode ter sido atribuida por lote.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yfvrisk import config as cfg
from yfvrisk.ingest.municipios import carregar_referencia

ARQ_HUMANOS = cfg.FONTE_MANUAL / "fa_casoshumanos_1994-2025.csv"
ARQ_EPIZOOTIAS = cfg.FONTE_MANUAL / "fa_epizpnh_1994-2025.csv"

# Mes em que comeca a temporada de monitoramento. Julho: esta no fundo do vale
# sazonal (11 casos humanos em toda a serie contra 1.294 em janeiro), de modo que
# o corte separa ondas em vez de parti-las.
MES_INICIO_TEMPORADA = 7

# Temporadas cobertas. A de 1994 comeca em julho de 1994; a ultima completa
# termina em junho de 2025.
TEMPORADA_INICIO = 1994
TEMPORADA_FIM = 2024

# Temporadas acrescentadas ALEM da ultima observada, para que o painel possa
# carregar a linha da temporada a prever.
#
# Sem isso nao ha como emitir previsao de verdade: as variaveis de propagacao da
# temporada T+1 sao funcao do desfecho de T, que existe, mas so podem ser
# calculadas se houver linha T+1 onde grava-las. A linha extra tem desfecho
# AUSENTE, e nao zero - marcar como zero ensinaria ao modelo que nao houve surto
# numa temporada que ainda nao aconteceu, o mesmo erro de tratar lacuna de
# vigilancia como zero observado.
TEMPORADAS_FUTURAS = 1

SAIDA = cfg.PROCESSADO / "epi_temporada.parquet"
DIAGNOSTICO = cfg.METRICAS / "f6_temporada_diagnostico.json"


def temporada_de(ano: pd.Series, mes: pd.Series) -> pd.Series:
    """Ano de inicio da temporada julho-junho a que a data pertence."""
    return np.where(mes >= MES_INICIO_TEMPORADA, ano, ano - 1)


def _rotulo_para_ano(rotulo: pd.Series) -> pd.Series:
    """"2016/2017" -> 2016."""
    return pd.to_numeric(rotulo.astype(str).str.slice(0, 4), errors="coerce")


def _ler(arquivo, col_mun, col_ano, col_mes, col_monit, ref, rotulo):
    df = pd.read_csv(arquivo, sep=";", encoding="latin-1")
    diag = {"arquivo": arquivo.name, "registros_brutos": int(len(df))}

    diag["sem_codigo_municipal"] = int(df[col_mun].isna().sum())
    df = df.dropna(subset=[col_mun]).copy()

    df["cod6"] = df[col_mun].astype(float).astype(int).astype(str).str.zfill(6)
    mapa = {c[:6]: c for c in ref["cod_mun"]}
    df["cod_mun"] = df["cod6"].map(mapa)
    diag["codigos_orfaos"] = sorted(df.loc[df["cod_mun"].isna(), "cod6"].unique())
    df = df.dropna(subset=["cod_mun"])

    ano = pd.to_numeric(df[col_ano], errors="coerce")
    mes = pd.to_numeric(df[col_mes], errors="coerce")

    # Temporada pela data quando ela existe; pelo campo declarado quando nao.
    pela_data = pd.Series(temporada_de(ano, mes), index=df.index)
    pela_data = pela_data.where(ano.notna() & mes.notna())
    pelo_campo = _rotulo_para_ano(df[col_monit])

    df["temporada"] = pela_data.fillna(pelo_campo)
    diag["sem_mes"] = int(mes.isna().sum())
    diag["recuperados_pelo_campo_monitoramento"] = int(
        pela_data.isna().sum() - df["temporada"].isna().sum())

    # Divergencia entre a data e a classificacao administrativa. Nao e erro
    # necessariamente - um caso de inicio em junho pode ser atribuido a temporada
    # seguinte pela investigacao -, mas precisa ficar contado.
    comparavel = pela_data.notna() & pelo_campo.notna()
    diag["divergencia_data_x_campo"] = int(
        (pela_data[comparavel] != pelo_campo[comparavel]).sum())
    diag["comparaveis"] = int(comparavel.sum())

    df = df.dropna(subset=["temporada"])
    df["temporada"] = df["temporada"].astype(int)
    df["ano_civil"] = ano
    df["mes"] = mes

    diag["registros_usados"] = int(len(df))
    diag["temporadas"] = [int(df["temporada"].min()), int(df["temporada"].max())]
    diag["por_mes"] = {int(k): int(v) for k, v in
                       mes.dropna().astype(int).value_counts().sort_index().items()}
    return df, diag


def construir() -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()
    ref = carregar_referencia(com_geometria=False)

    humanos, d_h = _ler(ARQ_HUMANOS, "COD_MUN_LPI", "ANO_IS", "MES_IS",
                        "MONITORAMENTO_IS", ref, "casos humanos")
    epiz, d_e = _ler(ARQ_EPIZOOTIAS, "COD_MUN_OCOR", "ANO_OCOR", "MES_OCOR",
                     "MONITORAMENTO_OCOR", ref, "epizootias")

    temporadas = range(TEMPORADA_INICIO, TEMPORADA_FIM + 1 + TEMPORADAS_FUTURAS)
    painel = pd.MultiIndex.from_product(
        [ref["cod_mun"], temporadas],
        names=["cod_mun", "temporada"]).to_frame(index=False)

    # Marca o que e observado. A temporada acrescentada nao tem desfecho, e a
    # distincao precisa sobreviver a todo o resto do pipeline.
    painel["observado"] = (painel["temporada"] <= TEMPORADA_FIM).astype(int)

    for nome, df, coluna in [("caso_humano", humanos, "n_casos_humanos"),
                             ("epizootia", epiz, "n_epizootias")]:
        cont = (df.groupby(["cod_mun", "temporada"]).size()
                .rename(coluna).reset_index())
        painel = painel.merge(cont, on=["cod_mun", "temporada"], how="left")
        painel[coluna] = painel[coluna].fillna(0)
        painel[nome] = (painel[coluna] > 0).astype(float)
        painel.loc[painel["observado"] == 0, [coluna, nome]] = np.nan

    painel["yf_event"] = np.where(
        painel["observado"] == 1,
        ((painel["caso_humano"] == 1) | (painel["epizootia"] == 1)).astype(float),
        np.nan)

    # Marcadores de observabilidade, os mesmos do painel anual, reexpressos em
    # temporada: uma temporada sem nenhum registro nacional de epizootia e lacuna
    # de vigilancia, nao ausencia de circulacao.
    obs = painel[painel["observado"] == 1]
    por_temporada = obs.groupby("temporada")["n_epizootias"].sum()
    sem_vigilancia = set(por_temporada[por_temporada == 0].index)
    painel["epizootia_observavel"] = (
        (painel["temporada"] >= cfg.ANO_INICIO_EPIZOOTIAS)
        & (~painel["temporada"].isin(sem_vigilancia))).astype(int)
    painel["yf_event_observavel"] = 1

    # Primeiro mes de ocorrencia dentro da temporada. Serve para medir a
    # defasagem entre epizootia e caso humano, que e o passo 6.5 do plano.
    for nome, df, col in [("mes_1o_caso", humanos, "mes"),
                          ("mes_1a_epizootia", epiz, "mes")]:
        # Mes ordinal dentro da temporada: julho = 0, junho = 11.
        d = df.dropna(subset=[col]).copy()
        d["ordinal"] = (d[col].astype(int) - MES_INICIO_TEMPORADA) % 12
        primeiro = (d.groupby(["cod_mun", "temporada"])["ordinal"].min()
                    .rename(nome).reset_index())
        painel = painel.merge(primeiro, on=["cod_mun", "temporada"], how="left")

    painel = painel.sort_values(["cod_mun", "temporada"]).reset_index(drop=True)

    diag = {
        "casos_humanos": d_h, "epizootias": d_e,
        "mes_inicio_temporada": MES_INICIO_TEMPORADA,
        "temporadas": [TEMPORADA_INICIO, TEMPORADA_FIM],
        "linhas": int(len(painel)),
        "municipios": int(painel["cod_mun"].nunique()),
        "temporadas_futuras": TEMPORADAS_FUTURAS,
        "linhas_observadas": int((painel["observado"] == 1).sum()),
        "eventos": int(painel["yf_event"].sum()),
        "prevalencia": float(painel.loc[painel["observado"] == 1,
                                        "yf_event"].mean()),
        "temporadas_sem_vigilancia_epizootias": sorted(int(t) for t in sem_vigilancia),
        "eventos_por_temporada": {int(t): int(v) for t, v in
                                  obs.groupby("temporada")["yf_event"].sum().items()},
    }

    painel.to_parquet(SAIDA, index=False)
    DIAGNOSTICO.write_text(json.dumps(diag, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return painel, diag


if __name__ == "__main__":
    p, d = construir()
    print(f"painel por temporada: {len(p):,} linhas, "
          f"{d['municipios']:,} municipios x {len(d['eventos_por_temporada'])} temporadas")
    print(f"eventos: {d['eventos']:,}  prevalencia {d['prevalencia']:.4%}")
    print(f"divergencia data x campo: casos {d['casos_humanos']['divergencia_data_x_campo']}"
          f"/{d['casos_humanos']['comparaveis']}, "
          f"epizootias {d['epizootias']['divergencia_data_x_campo']}"
          f"/{d['epizootias']['comparaveis']}")
    print(f"temporadas sem vigilancia de epizootia: "
          f"{d['temporadas_sem_vigilancia_epizootias']}")
    print("\ntemporada  eventos")
    for t, v in sorted(d["eventos_por_temporada"].items()):
        if v:
            print(f"  {t}/{str(t + 1)[2:]}   {v:4d}")
