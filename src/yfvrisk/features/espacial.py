"""Infraestrutura espacial: quem e vizinho de quem, e a que custo.

A febre amarela silvestre nao e um fenomeno local independente em cada municipio.
E uma onda epizootica que se propaga por floresta contigua, e a probabilidade de
um municipio ser atingido depende sobretudo de quao perto esteve a onda na
temporada anterior. Medido antes de escrever este modulo:

    raio    P(evento em t+1 | vizinho com evento em t)   base     enriquecimento
     50 km              0,0868                          0,0052        16,8x
    100 km              0,0624                          0,0040        15,5x
    200 km              0,0409                          0,0029        14,3x

Variaveis estaticas de adequabilidade de habitat nao capturam isso. Este modulo
constroi a base para capturar.

Distancia euclidiana e distancia efetiva
----------------------------------------
A onda viaja por floresta, nao por linha reta. Dois municipios a 80 km separados
por cerrado e pastagem estao, para o virus, muito mais longe que dois a 80 km
ligados por mata continua.

A distancia efetiva usada aqui pondera a distancia euclidiana pela cobertura
florestal media do CORREDOR entre os dois centroides - os municipios que a linha
reta atravessa. E uma aproximacao de caminho de menor custo, barata o bastante
para rodar sobre milhoes de pares e boa o bastante para o que se pede dela:
distinguir corredor de barreira.

    d_efetiva = d_euclidiana / max(floresta_do_corredor, PISO_FLORESTA)

O piso impede que um corredor sem floresta produza distancia infinita, o que
zeraria a contribuicao de qualquer vizinho em area aberta. Introducoes por
transporte humano existem e nao passam por floresta.

Custo computacional
-------------------
Sao 5.570 municipios, e a matriz completa teria 31 milhoes de pares. Tudo aqui e
esparso, cortado em RAIO_MAXIMO: com nucleo de decaimento de ate 200 km, um
vizinho a 600 km contribui com e^-3 = 5% e pode ser descartado sem consequencia.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from yfvrisk import config as cfg
from yfvrisk.ingest.municipios import carregar_referencia

# Alcance maximo considerado, em quilometros. Tres vezes o maior nucleo de
# decaimento pretendido.
RAIO_MAXIMO = 600.0

# Cobertura florestal minima atribuida a um corredor. Ver a nota do cabecalho.
PISO_FLORESTA = 0.05

# Pontos amostrados ao longo da reta entre dois centroides para estimar a
# floresta do corredor.
AMOSTRAS_CORREDOR = 12

CAMINHO_VIZINHOS = cfg.PROCESSADO / "espacial_vizinhos.npz"
DIAGNOSTICO = cfg.METRICAS / "f6_espacial_diagnostico.json"


def centroides() -> tuple[list[str], np.ndarray]:
    """Codigos e centroides em quilometros, na projecao metrica do IBGE.

    A policonica do SIRGAS 2000 e a projecao oficial para distancia no Brasil, e
    e a mesma ja fixada em `config.CRS_METRICO` para o calculo de distancia a
    estrada. Sobre lat/lon, "distancia em metros" sairia em graus.
    """
    ref = carregar_referencia(com_geometria=True).to_crs(cfg.CRS_METRICO)
    ref = ref.sort_values("cod_mun").reset_index(drop=True)
    c = ref.geometry.centroid
    xy = np.c_[c.x.to_numpy(), c.y.to_numpy()] / 1000.0
    return ref["cod_mun"].tolist(), xy


def _floresta_por_municipio(codigos: list[str]) -> np.ndarray:
    """Cobertura florestal media recente de cada municipio, em [0, 1].

    Usa `cobertura_florestal`, derivada do Hansen por `ingest/floresta.py`, e NAO
    `floresta_frac`, que vem do derivado de terceiros do MapBiomas. A escolha foi
    forcada por verificacao, e o resultado dela merece registro proprio.

    Media por unidade da federacao, 2020:

        UF     floresta_frac    cobertura_florestal (Hansen)
        AM         0,145                 0,894
        AC         0,286                 0,749
        RO         0,348                 0,461
        SP         0,161                 0,210
        RN         0,065                 0,173

    O Hansen ordena o territorio brasileiro como se espera: Amazonia no topo,
    semiarido no fundo. O derivado de uso do solo poe o Amazonas ABAIXO de
    Rondonia, concentra seus valores maximos no arco do desmatamento do leste do
    Para e atribui 0,39 de floresta a Taboao da Serra, que e um dos municipios
    mais urbanizados do pais. A correlacao entre os dois e de apenas 0,29, e as
    fracoes de uso do solo somam 0,805 em vez de 1.

    A escala dessas variaveis ja e tratada em `ingest/uso_solo.py`, mas a escala
    certa nao garante validade: elas nao medem o que dizem medir. Isso e coerente
    com o que a analise de explicabilidade mede por outro caminho -
    `floresta_frac`, `pastagem_frac` e `interface_floresta_antropico` estao entre
    as variaveis cuja importancia nao se distingue de zero em nenhum dos dois
    cortes temporais.

    A media das dez ultimas temporadas, e nao um ano so: o corredor e propriedade
    estrutural da paisagem, e um ano isolado carregaria ruido sem acrescentar
    informacao.
    """
    p = pd.read_parquet(cfg.PROCESSADO / "painel_analitico.parquet",
                        columns=["cod_mun", "ano", "cobertura_florestal"])
    recente = p[p["ano"] >= p["ano"].max() - 9]
    f = recente.groupby("cod_mun")["cobertura_florestal"].mean()
    return f.reindex(codigos).fillna(0.0).to_numpy().clip(0, 1)


def construir(raio: float = RAIO_MAXIMO) -> dict:
    """Constroi e grava a estrutura esparsa de vizinhanca.

    Grava tres vetores paralelos no formato de lista de arestas: origem, destino,
    distancia euclidiana e distancia efetiva por floresta. Lista de arestas e nao
    matriz porque a esparsidade e alta e porque a forma de lista e a que os
    calculos de forca de infeccao consomem diretamente.
    """
    cfg.garantir_diretorios()
    codigos, xy = centroides()
    n = len(codigos)
    floresta = _floresta_por_municipio(codigos)

    arv = cKDTree(xy)
    pares = arv.query_pairs(r=raio, output_type="ndarray")
    print(f"[espacial] {len(pares):,} pares dentro de {raio:.0f} km "
          f"({n:,} municipios)")

    i, j = pares[:, 0], pares[:, 1]
    d = np.linalg.norm(xy[i] - xy[j], axis=1)

    # Floresta do corredor: media da floresta dos municipios que contem os pontos
    # amostrados ao longo da reta. O vizinho mais proximo na arvore devolve o
    # municipio de cada ponto sem precisar de teste de pertinencia poligonal.
    print("[espacial] amostrando corredores")
    soma = np.zeros(len(pares))
    for k in range(AMOSTRAS_CORREDOR):
        t = (k + 0.5) / AMOSTRAS_CORREDOR
        ponto = xy[i] * (1 - t) + xy[j] * t
        _, quem = arv.query(ponto, k=1)
        soma += floresta[quem]
    floresta_corredor = soma / AMOSTRAS_CORREDOR

    d_efetiva = d / np.maximum(floresta_corredor, PISO_FLORESTA)

    np.savez_compressed(
        CAMINHO_VIZINHOS,
        codigos=np.array(codigos), xy=xy, floresta=floresta,
        origem=i.astype(np.int32), destino=j.astype(np.int32),
        distancia=d.astype(np.float32),
        floresta_corredor=floresta_corredor.astype(np.float32),
        distancia_efetiva=d_efetiva.astype(np.float32),
    )

    graus = np.bincount(np.r_[i, j], minlength=n)
    diag = {
        "municipios": n, "raio_maximo_km": raio,
        "pares": int(len(pares)),
        "densidade": float(2 * len(pares) / (n * (n - 1))),
        "piso_floresta": PISO_FLORESTA,
        "amostras_corredor": AMOSTRAS_CORREDOR,
        "vizinhos_por_municipio": {
            "min": int(graus.min()), "mediana": float(np.median(graus)),
            "max": int(graus.max())},
        "distancia_km": {"mediana": float(np.median(d)),
                         "p05": float(np.percentile(d, 5))},
        "floresta_corredor": {
            "media": float(floresta_corredor.mean()),
            "p10": float(np.percentile(floresta_corredor, 10)),
            "p90": float(np.percentile(floresta_corredor, 90))},
        "razao_efetiva_sobre_euclidiana": {
            "mediana": float(np.median(d_efetiva / d)),
            "p90": float(np.percentile(d_efetiva / d, 90))},
    }
    DIAGNOSTICO.write_text(json.dumps(diag, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return diag


def carregar() -> dict:
    """Le a estrutura gravada. Constroi na primeira chamada, se preciso."""
    if not CAMINHO_VIZINHOS.exists():
        construir()
    z = np.load(CAMINHO_VIZINHOS, allow_pickle=False)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    d = construir()
    print(f"\npares: {d['pares']:,}  densidade {d['densidade']:.3%}")
    print(f"vizinhos por municipio: mediana {d['vizinhos_por_municipio']['mediana']:.0f}"
          f", max {d['vizinhos_por_municipio']['max']}")
    print(f"floresta do corredor: media {d['floresta_corredor']['media']:.3f}"
          f"  [p10 {d['floresta_corredor']['p10']:.3f},"
          f" p90 {d['floresta_corredor']['p90']:.3f}]")
    print(f"distancia efetiva / euclidiana: mediana "
          f"{d['razao_efetiva_sobre_euclidiana']['mediana']:.1f}x")
