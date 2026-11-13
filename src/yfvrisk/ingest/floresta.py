"""Cobertura e perda florestal por municipio e ano, a partir do Hansen GFC.

Um levantamento unico 2000-2024 replicado para todos os anos descreveria perda
florestal ocorrida DEPOIS do desfecho que deveria prever - informacao do futuro
entrando no passado.

Aqui a serie e anual e causal: a floresta do municipio no ano t e a cobertura de
2000 menos a perda acumulada ATE t. Nenhuma linha enxerga perda posterior ao seu
proprio ano.

Inclui tambem a faixa de latitude 10N (ver `baixar_hansen.py`), sem a qual 12,4%
do territorio ficaria sem dado.

Metodo
------
Os tiles do Hansen sao de 40.000 x 40.000 pixels a 30 m, grandes demais para
carregar inteiros. O processamento e feito em blocos de linhas: para cada bloco,
a malha municipal e rasterizada na grade daquele recorte e as contagens sao
acumuladas por bincount.

Metricas por municipio e ano:

  floresta_km2         area com cobertura arborea acima do limiar, no ano t
  perda_acumulada_km2  perda acumulada de 2001 ate t
  perda_ano_km2        perda ocorrida no proprio ano t
  cobertura_florestal  fracao da area do municipio coberta por floresta em t
  hansen_cobertura     fracao do municipio efetivamente coberta pelos tiles
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from yfvrisk import config as cfg
from yfvrisk.ingest.municipios import carregar_referencia

DIR_PRINCIPAL = cfg.FONTE_MANUAL / "Hansen_2024_BR"
DIR_COMPLEMENTO = cfg.BRUTO / "hansen"

# Limiar de cobertura arborea para considerar o pixel floresta. 30% e o valor
# convencional na literatura que usa o Hansen e o adotado pela FAO.
LIMIAR_COBERTURA = 30

ANO_BASE = 2000          # ano do levantamento de cobertura
ANO_ULTIMO_HANSEN = 2024  # ultimo ano de perda na colecao GFC-2024-v1.12

BLOCO_LINHAS = 2048


def _tiles_disponiveis(camada: str) -> dict[str, "Path"]:
    """Tiles da camada, preferindo o complemento baixado quando houver duplicata."""
    encontrados = {}
    for diretorio in (DIR_PRINCIPAL, DIR_COMPLEMENTO):
        if not diretorio.exists():
            continue
        for p in diretorio.glob(f"*_{camada}_*.tif"):
            m = re.search(r"_(\d{2}[NS]_\d{3}[EW])\.tif$", p.name)
            if m:
                encontrados[m.group(1)] = p
    return encontrados


def _processar_tile(caminho_cover, caminho_loss, ref, n_mun, acumulador):
    """Acumula contagens de floresta e de perda por ano para um tile."""
    with rasterio.open(caminho_cover) as src_cover:
        src_loss = rasterio.open(caminho_loss) if caminho_loss else None
        try:
            altura, largura = src_cover.height, src_cover.width
            limites = src_cover.bounds

            # So os municipios que intersectam o tile precisam ser rasterizados.
            recorte = ref.cx[limites.left:limites.right, limites.bottom:limites.top]
            if recorte.empty:
                return

            for inicio in range(0, altura, BLOCO_LINHAS):
                n_linhas = min(BLOCO_LINHAS, altura - inicio)
                janela = Window(0, inicio, largura, n_linhas)
                transform = src_cover.window_transform(janela)

                # Subconjunto espacial ANTES de rasterizar. A maioria dos blocos
                # de um tile amazonico ou costeiro nao toca nenhum municipio, e
                # rasterizar 82 milhoes de pixels para depois descobrir isso
                # dominaria o tempo de execucao.
                topo = transform.f
                base = transform.f + transform.e * n_linhas
                no_bloco = recorte.cx[limites.left:limites.right, base:topo]
                if no_bloco.empty:
                    continue

                rotulos = rasterize(
                    ((g, i + 1) for i, g in zip(no_bloco.index, no_bloco.geometry)),
                    out_shape=(n_linhas, largura), transform=transform,
                    fill=0, dtype="int32", all_touched=False)
                if not rotulos.any():
                    continue

                cover = src_cover.read(1, window=janela)
                dentro = rotulos > 0

                # Area do pixel varia com a latitude; usa-se o cosseno da latitude
                # do centro de cada linha para converter contagem em km2.
                lats = np.array([transform.f + transform.e * (r + 0.5)
                                 for r in range(n_linhas)])
                area_linha = (
                    abs(transform.a) * abs(transform.e)
                    * (111.32 ** 2) * np.cos(np.radians(lats))
                )
                pesos = np.repeat(area_linha[:, None], largura, axis=1)

                rot = rotulos[dentro]
                acumulador["area_total"] += np.bincount(
                    rot, weights=pesos[dentro], minlength=n_mun + 1)

                eh_floresta = cover[dentro] >= LIMIAR_COBERTURA
                acumulador["floresta_2000"] += np.bincount(
                    rot[eh_floresta], weights=pesos[dentro][eh_floresta],
                    minlength=n_mun + 1)

                if src_loss is not None:
                    loss = src_loss.read(1, window=janela)[dentro]
                    perdeu = (loss > 0) & eh_floresta
                    if perdeu.any():
                        anos = loss[perdeu].astype(int)
                        for ano_rel in np.unique(anos):
                            sel = anos == ano_rel
                            acumulador["perda_por_ano"][ano_rel] += np.bincount(
                                rot[perdeu][sel], weights=pesos[dentro][perdeu][sel],
                                minlength=n_mun + 1)
        finally:
            if src_loss is not None:
                src_loss.close()


def construir() -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()
    ref = carregar_referencia(com_geometria=True).to_crs("EPSG:4326").reset_index(drop=True)
    n_mun = len(ref)

    tiles_cover = _tiles_disponiveis("treecover2000")
    tiles_loss = _tiles_disponiveis("lossyear")
    print(f"[tiles] treecover2000: {len(tiles_cover)} | lossyear: {len(tiles_loss)}")

    acumulador = {
        "area_total": np.zeros(n_mun + 1),
        "floresta_2000": np.zeros(n_mun + 1),
        "perda_por_ano": {a: np.zeros(n_mun + 1)
                          for a in range(1, ANO_ULTIMO_HANSEN - ANO_BASE + 1)},
    }

    for i, (tile, caminho) in enumerate(sorted(tiles_cover.items()), 1):
        print(f"[{i}/{len(tiles_cover)}] {tile}"
              f"{'' if tile in tiles_loss else '  (sem lossyear)'}")
        _processar_tile(caminho, tiles_loss.get(tile), ref, n_mun, acumulador)

    # --- Montagem do painel anual ------------------------------------------
    area_hansen = acumulador["area_total"][1:]
    floresta_2000 = acumulador["floresta_2000"][1:]

    linhas = []
    perda_acum = np.zeros(n_mun)
    for ano in range(cfg.ANO_INICIO, cfg.ANO_FIM + 1):
        rel = ano - ANO_BASE
        perda_ano = np.zeros(n_mun)
        if 1 <= rel <= ANO_ULTIMO_HANSEN - ANO_BASE:
            perda_ano = acumulador["perda_por_ano"][rel][1:]
            perda_acum = perda_acum + perda_ano

        # Antes de 2000 nao ha informacao de perda: a floresta e a de 2000, e a
        # linha e marcada para que a modelagem saiba que e extrapolacao para tras.
        floresta = floresta_2000 - perda_acum if ano >= ANO_BASE else floresta_2000

        linhas.append(pd.DataFrame({
            "cod_mun": ref["cod_mun"].to_numpy(),
            "ano": ano,
            "floresta_km2": np.maximum(floresta, 0),
            "perda_ano_km2": perda_ano,
            "perda_acumulada_km2": perda_acum.copy(),
            "anterior_ao_hansen": int(ano < ANO_BASE),
            "posterior_ao_hansen": int(ano > ANO_ULTIMO_HANSEN),
        }))

    painel = pd.concat(linhas, ignore_index=True)
    painel = painel.merge(ref[["cod_mun", "area_km2"]], on="cod_mun", how="left")
    painel["cobertura_florestal"] = painel["floresta_km2"] / painel["area_km2"]

    cobertura_tile = pd.DataFrame({
        "cod_mun": ref["cod_mun"].to_numpy(),
        "hansen_cobertura": area_hansen / ref["area_km2"].to_numpy(),
    })
    painel = painel.merge(cobertura_tile, on="cod_mun", how="left")

    sem_cobertura = int((cobertura_tile["hansen_cobertura"] < 0.01).sum())
    parcial = int(((cobertura_tile["hansen_cobertura"] >= 0.01)
                   & (cobertura_tile["hansen_cobertura"] < 0.95)).sum())

    diag = {
        "limiar_cobertura_arborea_pct": LIMIAR_COBERTURA,
        "tiles_treecover2000": len(tiles_cover),
        "tiles_lossyear": len(tiles_loss),
        "tiles_sem_lossyear": sorted(set(tiles_cover) - set(tiles_loss)),
        "municipios_sem_cobertura_hansen": sem_cobertura,
        "municipios_com_cobertura_parcial": parcial,
        "linhas": int(len(painel)),
        "floresta_total_2000_km2": float(floresta_2000.sum()),
        "perda_total_2001_2024_km2": float(perda_acum.sum()),
        "anos_sem_dado_de_perda": {
            "antes": [cfg.ANO_INICIO, ANO_BASE - 1],
            "depois": [ANO_ULTIMO_HANSEN + 1, cfg.ANO_FIM],
        },
    }

    painel.to_parquet(cfg.PROCESSADO / "floresta_muni_ano.parquet", index=False)
    (cfg.METRICAS / "f2_floresta_diagnostico.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    return painel, diag


if __name__ == "__main__":
    p, d = construir()
    print()
    print(json.dumps(d, indent=2, ensure_ascii=False))
