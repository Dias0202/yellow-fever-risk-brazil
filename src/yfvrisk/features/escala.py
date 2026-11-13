"""Painel de modelagem em escala agregada: microrregiao ou regiao imediata.

Por que trocar de escala
------------------------
Ao nivel municipal a prevalencia e de 0,73% por temporada. Com desfecho tao raro,
uma lista de prioridade de cem municipios captura tipicamente 2% dos surtos, o
que nao sustenta decisao.

A escala nao e apenas um detalhe estatistico, e tambem uma questao biologica. O
ciclo enzootico da febre amarela silvestre ocorre em manchas de mata que
atravessam divisas municipais; um foco detectado num municipio e frequentemente o
mesmo foco que apareceria no vizinho se a equipe de vigilancia tivesse ido la. A
divisa municipal e uma linha administrativa, e nao uma barreira epidemiologica.

Agregar troca resolucao espacial por densidade de sinal. E a troca vale se o
alerta agregado for util e o municipal nao for - vale mais dizer corretamente
"esta regiao" que dizer inutilmente "este municipio".

Como a agregacao e feita
------------------------
O desfecho e a UNIAO: a unidade tem evento se qualquer municipio dela teve. Nao a
media nem a contagem, porque a pergunta operacional e "preciso mobilizar equipe
para esta regiao?".

As variaveis de propagacao sao RECALCULADAS sobre os centroides das unidades
agregadas, e nao medias das municipais. A diferenca importa: a forca de infeccao
e nao linear na distancia, e a media das forcas municipais nao e a forca do
centroide da regiao.

As variaveis ambientais sao medias ponderadas por area, exceto populacao, que e
soma, e as de relevo, em que o minimo e o maximo se propagam como minimo e
maximo. Uma media de altitude minima nao seria a altitude minima da regiao.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from yfvrisk import config as cfg
from yfvrisk.features import epidemia
from yfvrisk.features.espacial import PISO_FLORESTA, RAIO_MAXIMO
from yfvrisk.features.espacial import carregar as carregar_espacial
from yfvrisk.ingest import regionalizacao

ESCALAS = {
    "microrregiao": "cod_microrregiao",
    "regiao_imediata": "cod_regiao_imediata",
    "mesorregiao": "cod_mesorregiao",
}

AMOSTRAS_CORREDOR = 12


def _arestas(xy, floresta, raio=RAIO_MAXIMO):
    """Estrutura de vizinhanca para um conjunto qualquer de centroides."""
    arv = cKDTree(xy)
    pares = arv.query_pairs(r=raio, output_type="ndarray")
    i, j = pares[:, 0], pares[:, 1]
    d = np.linalg.norm(xy[i] - xy[j], axis=1)

    soma = np.zeros(len(pares))
    for k in range(AMOSTRAS_CORREDOR):
        t = (k + 0.5) / AMOSTRAS_CORREDOR
        _, quem = arv.query(xy[i] * (1 - t) + xy[j] * t, k=1)
        soma += floresta[quem]
    fc = soma / AMOSTRAS_CORREDOR

    return {"origem": i.astype(np.int32), "destino": j.astype(np.int32),
            "distancia": d.astype(np.float32),
            "distancia_efetiva": (d / np.maximum(fc, PISO_FLORESTA)).astype(
                np.float32)}


def construir(escala: str = "regiao_imediata") -> tuple[pd.DataFrame, dict]:
    if escala not in ESCALAS:
        raise SystemExit(f"escala desconhecida: {escala}. Use {list(ESCALAS)}")
    chave = ESCALAS[escala]
    cfg.garantir_diretorios()

    mapa = regionalizacao.carregar()[["cod_mun", chave]]
    z = carregar_espacial()
    mun = [str(c) for c in z["codigos"]]
    xy_mun = z["xy"]
    flor_mun = z["floresta"]

    pos = pd.DataFrame({"cod_mun": mun, "x": xy_mun[:, 0], "y": xy_mun[:, 1],
                        "floresta": flor_mun}).merge(mapa, on="cod_mun")

    # Centroide da unidade: media dos centroides municipais ponderada por area.
    # Ponderar e importante em regioes com um municipio grande e varios pequenos.
    area = pd.read_parquet(cfg.PROCESSADO / "municipios_referencia.parquet",
                           columns=["cod_mun", "area_km2"])
    pos = pos.merge(area, on="cod_mun")
    pos["w"] = pos["area_km2"]

    g = pos.groupby(chave)
    unidades = pd.DataFrame({
        "x": g.apply(lambda d: np.average(d["x"], weights=d["w"])),
        "y": g.apply(lambda d: np.average(d["y"], weights=d["w"])),
        "floresta": g.apply(lambda d: np.average(d["floresta"], weights=d["w"])),
        "area_km2": g["area_km2"].sum(),
        "n_municipios": g.size(),
    }).sort_index()
    codigos = unidades.index.tolist()
    xy = unidades[["x", "y"]].to_numpy()

    # ------------------------------------------------------------- desfecho
    epi = pd.read_parquet(cfg.PROCESSADO / "epi_temporada.parquet")
    epi = epi.merge(mapa, on="cod_mun")
    temporadas = sorted(epi["temporada"].unique())
    agregado = (epi.groupby([chave, "temporada"])
                .agg(yf_event=("yf_event", "max"),
                     caso_humano=("caso_humano", "max"),
                     epizootia=("epizootia", "max"),
                     n_municipios_com_evento=("yf_event", "sum"),
                     n_casos_humanos=("n_casos_humanos", "sum"),
                     n_epizootias=("n_epizootias", "sum"))
                .reset_index())

    def _desfecho(coluna):
        return (agregado.pivot(index=chave, columns="temporada", values=coluna)
                .reindex(codigos)[temporadas].fillna(0).to_numpy(dtype=np.float32))

    # ---------------------------------------------------------- propagacao
    arestas = _arestas(xy, unidades["floresta"].to_numpy())
    prop, diag_prop = epidemia.derivar(
        codigos, xy, arestas, _desfecho("yf_event"), _desfecho("epizootia"),
        temporadas, chave=chave)

    painel = agregado.merge(prop, on=[chave, "temporada"], how="left")

    # ---------------------------------------------------------- ambiental
    amb = pd.read_parquet(cfg.PROCESSADO / "painel_modelagem_temporada.parquet")
    amb = amb.merge(mapa, on="cod_mun")
    amb = amb.merge(area, on="cod_mun", suffixes=("", "_mun"))

    soma = ["populacao_t1"]
    maximo = ["altitude_max"]
    minimo = ["altitude_min"]
    ignorar = set(soma + maximo + minimo) | {
        "cod_mun", "temporada", chave, "yf_event", "caso_humano", "epizootia",
        "regiao", "uf", "cod_uf", "area_km2", "area_km2_mun",
        "n_casos_humanos", "n_epizootias", "mes_1o_caso", "mes_1a_epizootia",
        "epizootia_observavel", "yf_event_observavel"}
    ignorar |= set(prop.columns)
    media = [c for c in amb.columns
             if c not in ignorar and pd.api.types.is_numeric_dtype(amb[c])]

    # Media ponderada por area feita como soma de produtos dividida por soma de
    # pesos, e nao com `groupby.apply`: alem de ser ordens de grandeza mais
    # rapido, trata ausente corretamente - o peso de um municipio sem valor sai
    # do denominador em vez de contaminar a media.
    w = amb["area_km2_mun"].to_numpy(dtype=float)
    numerador = amb[[chave, "temporada"]].copy()
    denominador = amb[[chave, "temporada"]].copy()
    for c in media:
        v = amb[c].to_numpy(dtype=float)
        ok = np.isfinite(v)
        numerador[c] = np.where(ok, v * w, 0.0)
        denominador[c] = np.where(ok, w, 0.0)
    gnum = numerador.groupby([chave, "temporada"])[media].sum()
    gden = denominador.groupby([chave, "temporada"])[media].sum()
    ambiental = (gnum / gden.replace(0, np.nan))

    gg = amb.groupby([chave, "temporada"])
    for cols, fn in [(soma, "sum"), (maximo, "max"), (minimo, "min")]:
        presentes = [c for c in cols if c in amb.columns]
        if presentes:
            ambiental = ambiental.join(getattr(gg[presentes], fn)())
    ambiental = ambiental.reset_index()

    painel = painel.merge(ambiental, on=[chave, "temporada"], how="left")
    painel = painel.merge(
        unidades[["area_km2", "n_municipios"]].reset_index(), on=chave, how="left")

    # Macrorregiao da unidade: a do primeiro municipio, que para microrregiao e
    # regiao imediata e sempre a mesma - as duas nao atravessam UF.
    reg = (pd.read_parquet(cfg.PROCESSADO / "painel_modelagem_temporada.parquet",
                           columns=["cod_mun", "regiao", "uf"])
           .drop_duplicates("cod_mun").merge(mapa, on="cod_mun")
           .groupby(chave)[["regiao", "uf"]].first().reset_index())
    painel = painel.merge(reg, on=chave, how="left")

    painel = painel.sort_values([chave, "temporada"]).reset_index(drop=True)
    saida = cfg.PROCESSADO / f"painel_{escala}.parquet"
    painel.to_parquet(saida, index=False)

    diag = {
        "escala": escala, "chave": chave,
        "unidades": int(len(codigos)),
        "linhas": int(len(painel)),
        "temporadas": [int(temporadas[0]), int(temporadas[-1])],
        "municipios_por_unidade": {
            "min": int(unidades["n_municipios"].min()),
            "mediana": float(unidades["n_municipios"].median()),
            "max": int(unidades["n_municipios"].max())},
        "prevalencia": {
            "yf_event": float(painel["yf_event"].mean()),
            "caso_humano": float(painel["caso_humano"].mean()),
            "epizootia": float(painel["epizootia"].mean())},
        "eventos_por_temporada": {
            int(t): int(v) for t, v in
            painel.groupby("temporada")["yf_event"].sum().items()},
        "variaveis_propagacao": diag_prop["variaveis"],
        "arquivo": str(saida),
    }
    (cfg.METRICAS / f"f6_escala_{escala}.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    return painel, diag


if __name__ == "__main__":
    import sys
    escala = sys.argv[1] if len(sys.argv) > 1 else "regiao_imediata"
    p, d = construir(escala)
    print(f"\n{escala}: {d['unidades']} unidades, {d['linhas']:,} linhas")
    print(f"municipios por unidade: mediana {d['municipios_por_unidade']['mediana']:.0f}"
          f" (max {d['municipios_por_unidade']['max']})")
    print(f"prevalencia: yf_event {d['prevalencia']['yf_event']:.3%}, "
          f"caso humano {d['prevalencia']['caso_humano']:.3%}")
