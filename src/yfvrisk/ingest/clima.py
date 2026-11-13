"""Agregacao do TerraClimate por municipio e mes.

Produz, de forma reproduzivel e conferivel, todo o conjunto climatico do
modelo a partir das grades oficiais do TerraClimate.

Estrategia
----------
O TerraClimate e uma grade global de 4320 x 8640 a 1/24 de grau (~4,6 km), com 12
meses por arquivo anual. Ler o mundo inteiro seria 4 GB por arquivo; o recorte do
Brasil sai em 92 MB e menos de um segundo.

A agregacao zonal e feita rasterizando a malha municipal UMA vez sobre a grade do
TerraClimate e depois usando bincount por mes. Fazer zonal_stats poligono a
poligono seria 10 milhoes de operacoes; assim sao 155 leituras e uma soma vetorizada.

Municipios menores que a celula
-------------------------------
A celula tem ~21 km2 e o menor municipio brasileiro tem 3,57 km2. Muitos municipios
nao contem nenhum centro de celula e ficariam sem valor. Para esses, o valor e
amostrado na celula valida mais proxima do ponto representativo do poligono, e a
linha e marcada em `origem_amostragem` para que a modelagem saiba quais valores
vieram por vizinhanca e nao por media zonal.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin

from yfvrisk import config as cfg
from yfvrisk.ingest.municipios import carregar_referencia

VARIAVEIS = ["ppt", "tmax", "tmin", "vpd", "ws"]
DIR_NC = cfg.FONTE_MANUAL / "TerraClimate_raw"
DIR_RECUPERADOS = cfg.BRUTO / "terraclimate"

# Abaixo disso o arquivo esta truncado. tmax_1994.nc chegou com 0 bytes na origem.
BYTES_MINIMOS = 10_000_000


def _arquivos_de(variavel: str) -> list:
    """Arquivos anuais de uma variavel, trocando os truncados pelos recuperados.

    A copia local recebida e preservada como esta. Quando um arquivo dela esta
    truncado, usa-se a versao rebaixada da fonte oficial em
    data/raw/terraclimate, e a substituicao fica registrada no diagnostico.
    """
    escolhidos, substituicoes = [], []
    for arq in sorted(DIR_NC.glob(f"{variavel}_*.nc")):
        if arq.stat().st_size >= BYTES_MINIMOS:
            escolhidos.append(arq)
            continue
        alternativa = DIR_RECUPERADOS / arq.name
        if alternativa.exists() and alternativa.stat().st_size >= BYTES_MINIMOS:
            escolhidos.append(alternativa)
            substituicoes.append(arq.name)
        else:
            raise FileNotFoundError(
                f"{arq.name} esta truncado ({arq.stat().st_size} bytes) e nao ha "
                f"versao recuperada. Rode: python -m yfvrisk.ingest."
                f"baixar_terraclimate --recuperar-truncados")
    return escolhidos, substituicoes

# Recorte com folga sobre o territorio brasileiro.
#
# O limite leste precisa cobrir as ilhas oceanicas, nao so o continente. A ponta
# continental mais a leste e a Ponta do Seixas (-34,79), mas Fernando de Noronha
# fica em -32,4 e e um municipio (2605459). Com o recorte parando em -33,5 ele
# caia fora da grade e saia com valor ausente em todos os 372 meses.
# -28,0 cobre tambem Trindade e Martim Vaz.
LAT_MAX, LAT_MIN = 6.0, -34.5
LON_MIN, LON_MAX = -74.5, -28.0


def _grade_referencia():
    """Coordenadas e transform da grade recortada, lidos de um arquivo real."""
    arq = next(DIR_NC.glob("ppt_*.nc"))
    with xr.open_dataset(arq) as ds:
        lat = ds["lat"].sel(lat=slice(LAT_MAX, LAT_MIN)).values
        lon = ds["lon"].sel(lon=slice(LON_MIN, LON_MAX)).values
    res_lat = abs(float(lat[1] - lat[0]))
    res_lon = abs(float(lon[1] - lon[0]))
    # Origem no canto superior esquerdo do primeiro pixel.
    transform = from_origin(lon[0] - res_lon / 2, lat[0] + res_lat / 2, res_lon, res_lat)
    return lat, lon, transform


def construir_rotulos():
    """Rasteriza a malha municipal sobre a grade do TerraClimate.

    Devolve o raster de rotulos (indice do municipio + 1, com 0 = sem municipio),
    a referencia municipal e o diagnostico de cobertura.
    """
    lat, lon, transform = _grade_referencia()
    ref = carregar_referencia(com_geometria=True)
    ref = ref.to_crs("EPSG:4326").reset_index(drop=True)

    formas = ((geom, i + 1) for i, geom in enumerate(ref.geometry))
    rotulos = rasterize(
        formas, out_shape=(len(lat), len(lon)), transform=transform,
        fill=0, dtype="int32", all_touched=False,
    )

    contagem = np.bincount(rotulos.ravel(), minlength=len(ref) + 1)[1:]
    sem_celula = np.where(contagem == 0)[0]

    # Para os municipios sem centro de celula, guarda o indice de grade do ponto
    # representativo, que sera usado para amostrar o vizinho mais proximo.
    pontos = {}
    if len(sem_celula):
        rep = ref.geometry.representative_point()
        for i in sem_celula:
            p = rep.iloc[i]
            j = int(np.abs(lat - p.y).argmin())
            k = int(np.abs(lon - p.x).argmin())
            pontos[int(i)] = (j, k)

    diag = {
        "grade": {"n_lat": len(lat), "n_lon": len(lon),
                  "res_graus": round(abs(float(lat[1] - lat[0])), 6)},
        "municipios": int(len(ref)),
        "com_media_zonal": int((contagem > 0).sum()),
        "amostrados_por_vizinhanca": int(len(sem_celula)),
        "celulas_por_municipio": {
            "min": int(contagem.min()), "mediana": float(np.median(contagem)),
            "max": int(contagem.max()),
        },
    }
    return rotulos, ref, pontos, diag


def _agregar_arquivo(arq, variavel, rotulos, n_mun, pontos):
    """Media zonal por municipio para os 12 meses de um arquivo anual."""
    with xr.open_dataset(arq) as ds:
        bloco = ds[variavel].sel(lat=slice(LAT_MAX, LAT_MIN),
                                 lon=slice(LON_MIN, LON_MAX)).values

    plano = rotulos.ravel()
    linhas = []
    for mes in range(bloco.shape[0]):
        vals = bloco[mes].ravel()
        valido = (plano > 0) & np.isfinite(vals)

        soma = np.bincount(plano[valido], weights=vals[valido], minlength=n_mun + 1)
        cont = np.bincount(plano[valido], minlength=n_mun + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            media = np.where(cont > 0, soma / np.maximum(cont, 1), np.nan)[1:]

        # Municipios sem celula: vizinho mais proximo valido.
        if pontos:
            grade = bloco[mes]
            for i, (j, k) in pontos.items():
                media[i] = _vizinho_valido(grade, j, k)

        linhas.append(media)
    return np.vstack(linhas)  # (12, n_mun)


def _vizinho_valido(grade, j, k, raio_max=6):
    """Valor valido mais proximo de (j, k), procurando em aneis crescentes."""
    if np.isfinite(grade[j, k]):
        return grade[j, k]
    n_j, n_k = grade.shape
    for r in range(1, raio_max + 1):
        j0, j1 = max(0, j - r), min(n_j, j + r + 1)
        k0, k1 = max(0, k - r), min(n_k, k + r + 1)
        janela = grade[j0:j1, k0:k1]
        if np.isfinite(janela).any():
            return float(np.nanmean(janela))
    return np.nan


def construir() -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()
    rotulos, ref, pontos, diag = construir_rotulos()
    n_mun = len(ref)
    cods = ref["cod_mun"].to_numpy()

    print(f"[grade] {diag['grade']['n_lat']} x {diag['grade']['n_lon']} celulas")
    print(f"[cobertura] {diag['com_media_zonal']} municipios por media zonal, "
          f"{diag['amostrados_por_vizinhanca']} por vizinhanca")

    partes = []
    todas_substituicoes = []
    for var in VARIAVEIS:
        arquivos, substituicoes = _arquivos_de(var)
        todas_substituicoes.extend(substituicoes)
        if substituicoes:
            print(f"[{var}] usando versao recuperada para: {', '.join(substituicoes)}")
        print(f"[{var}] {len(arquivos)} arquivos anuais")
        for arq in arquivos:
            ano = int(arq.stem.split("_")[1])
            m = _agregar_arquivo(arq, var, rotulos, n_mun, pontos)
            d = pd.DataFrame({
                "cod_mun": np.tile(cods, 12),
                "ano": ano,
                "mes": np.repeat(np.arange(1, 13), n_mun),
                var: m.ravel(),
            })
            partes.append(d.set_index(["cod_mun", "ano", "mes"]))
        print(f"[{var}] concluido")

    mensal = pd.concat(
        [pd.concat([p for p in partes if v in p.columns]) for v in VARIAVEIS],
        axis=1).reset_index()

    idx_viz = set(pontos)
    mensal["origem_amostragem"] = np.where(
        mensal["cod_mun"].isin(cods[list(idx_viz)]) if idx_viz else False,
        "vizinho", "media_zonal")

    mensal.to_parquet(cfg.PROCESSADO / "clima_muni_mes.parquet", index=False)

    diag["linhas_mensais"] = int(len(mensal))
    diag["anos"] = [int(mensal["ano"].min()), int(mensal["ano"].max())]
    diag["na_por_variavel"] = {v: int(mensal[v].isna().sum()) for v in VARIAVEIS}
    diag["arquivos_substituidos_por_versao_recuperada"] = todas_substituicoes
    (cfg.METRICAS / "f2_clima_diagnostico.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    return mensal, diag


if __name__ == "__main__":
    m, d = construir()
    print()
    print(json.dumps(d, indent=2, ensure_ascii=False))
    print(f"\nclima mensal: {len(m):,} linhas")
    print(m.head().to_string(index=False))
