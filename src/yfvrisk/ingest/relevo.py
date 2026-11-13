"""Altitude e relevo por municipio.

Por que altitude entra num modelo de febre amarela silvestre
------------------------------------------------------------
*Haemagogus janthinomys* e *Hg. leucocelaenus*, os vetores do ciclo silvestre no
Brasil, tem distribuicao altitudinal caracteristica: sao mosquitos de dossel de
mata e ocorrem sobretudo abaixo de ~1.200 m, com maior abundancia em cotas
intermediarias. A onda de 2016-2019 correu justamente pelos vales e encostas da
Serra do Mar e da Mantiqueira, e nao pelas cotas mais altas nem pela planicie
costeira urbanizada.

Alem da altitude media, entra a AMPLITUDE altimetrica do municipio. Ela e um
indicador barato de terreno acidentado, e terreno acidentado e onde a mata
remanescente sobreviveu no Sudeste - as encostas nao viraram lavoura. E tambem
onde vale encaixado e dossel continuo se combinam, que e o habitat do vetor.

Fonte
-----
Elevacao do WorldClim 2.1 a 2,5 arc-min, derivada do SRTM. E a mesma familia de
dados ja usada nos cenarios climaticos, o que evita introduzir mais uma fonte com
grade e convencao proprias. A resolucao de ~4,6 km e grosseira para um municipio
pequeno, mas o que se pede da variavel - faixa altitudinal e aspereza do terreno -
sobrevive a ela.

Uso:
    python -m yfvrisk.ingest.relevo
"""

from __future__ import annotations

import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from yfvrisk import config as cfg
from yfvrisk.ingest.municipios import carregar_referencia

URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_elev.zip"
DESTINO = cfg.BRUTO / "worldclim" / "elevacao"
SAIDA = cfg.PROCESSADO / "relevo_muni.parquet"
DIAGNOSTICO = cfg.METRICAS / "f6_relevo_diagnostico.json"
BYTES_MINIMOS = 5_000_000


def baixar(forcar: bool = False):
    DESTINO.mkdir(parents=True, exist_ok=True)
    zipado = DESTINO / "wc2.1_2.5m_elev.zip"
    if not zipado.exists() or zipado.stat().st_size < BYTES_MINIMOS or forcar:
        print(f"[baixar] {URL}")
        parcial = zipado.with_suffix(".parcial")
        urllib.request.urlretrieve(URL, parcial)
        if parcial.stat().st_size < BYTES_MINIMOS:
            parcial.unlink()
            raise RuntimeError("download truncado")
        parcial.replace(zipado)
    with zipfile.ZipFile(zipado) as z:
        z.extractall(DESTINO)
    tifs = sorted(DESTINO.glob("*.tif"))
    if not tifs:
        raise RuntimeError("nenhum raster de elevacao extraido")
    return tifs[0]


def construir() -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()
    raster = baixar()

    ref = carregar_referencia(com_geometria=True).to_crs("EPSG:4326")
    ref = ref.sort_values("cod_mun").reset_index(drop=True)

    with rasterio.open(raster) as src:
        elev = src.read(1, masked=True).astype("float64").filled(np.nan)
        rotulos = rasterize(((g, i + 1) for i, g in enumerate(ref.geometry)),
                            out_shape=(src.height, src.width),
                            transform=src.transform, fill=0, dtype="int32")
        transform = src.transform

    plano, vals = rotulos.ravel(), elev.ravel()
    valido = (plano > 0) & np.isfinite(vals)
    n = len(ref)

    soma = np.bincount(plano[valido], weights=vals[valido], minlength=n + 1)[1:]
    soma2 = np.bincount(plano[valido], weights=vals[valido] ** 2,
                        minlength=n + 1)[1:]
    cont = np.bincount(plano[valido], minlength=n + 1)[1:]

    with np.errstate(invalid="ignore", divide="ignore"):
        media = np.where(cont > 0, soma / np.maximum(cont, 1), np.nan)
        var = np.where(cont > 0, soma2 / np.maximum(cont, 1) - media ** 2, np.nan)
    desvio = np.sqrt(np.maximum(var, 0))

    # Minimo e maximo exigem varredura por rotulo; feita so onde ha celulas.
    minimo = np.full(n, np.nan)
    maximo = np.full(n, np.nan)
    ordem = np.argsort(plano[valido], kind="stable")
    p_ord, v_ord = plano[valido][ordem], vals[valido][ordem]
    limites = np.searchsorted(p_ord, np.arange(1, n + 2))
    for i in range(n):
        a, b = limites[i], limites[i + 1]
        if b > a:
            minimo[i] = v_ord[a:b].min()
            maximo[i] = v_ord[a:b].max()

    # Municipios sem celula propria: amostra a celula mais proxima do ponto
    # representativo. Mesma politica dos demais modulos - nunca imputar pela
    # mediana nacional em silencio.
    sem = np.where(cont == 0)[0]
    if len(sem):
        rep = ref.geometry.representative_point()
        for i in sem:
            linha, col = rasterio.transform.rowcol(transform, rep.iloc[i].x,
                                                   rep.iloc[i].y)
            j0, j1 = max(0, linha - 3), min(elev.shape[0], linha + 4)
            k0, k1 = max(0, col - 3), min(elev.shape[1], col + 4)
            janela = elev[j0:j1, k0:k1]
            if np.isfinite(janela).any():
                media[i] = np.nanmean(janela)
                minimo[i] = np.nanmin(janela)
                maximo[i] = np.nanmax(janela)
                desvio[i] = np.nanstd(janela)

    quadro = pd.DataFrame({
        "cod_mun": ref["cod_mun"],
        "altitude_media": media,
        "altitude_min": minimo,
        "altitude_max": maximo,
        "altitude_desvio": desvio,
        "amplitude_altimetrica": maximo - minimo,
    })
    quadro.to_parquet(SAIDA, index=False)

    diag = {
        "fonte": URL, "raster": raster.name,
        "municipios": int(len(quadro)),
        "com_celula_propria": int((cont > 0).sum()),
        "amostrados_por_vizinhanca": int(len(sem)),
        "ausentes": int(quadro["altitude_media"].isna().sum()),
        "altitude_media": {
            "min": float(np.nanmin(media)), "mediana": float(np.nanmedian(media)),
            "max": float(np.nanmax(media))},
        "amplitude_altimetrica_mediana": float(
            np.nanmedian(quadro["amplitude_altimetrica"])),
    }
    DIAGNOSTICO.write_text(json.dumps(diag, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return quadro, diag


if __name__ == "__main__":
    q, d = construir()
    print(f"relevo: {len(q):,} municipios, {d['ausentes']} ausentes")
    print(f"altitude media: mediana {d['altitude_media']['mediana']:.0f} m "
          f"[{d['altitude_media']['min']:.0f}, {d['altitude_media']['max']:.0f}]")
    print(f"amplitude altimetrica mediana: {d['amplitude_altimetrica_mediana']:.0f} m")
    print(f"amostrados por vizinhanca: {d['amostrados_por_vizinhanca']}")
