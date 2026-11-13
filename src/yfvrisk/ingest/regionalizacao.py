"""Hierarquia territorial do IBGE: microrregiao, regiao imediata, mesorregiao.

Por que uma escala alem do municipio
------------------------------------
O municipio pode ser fino demais para o que se quer prever. A febre amarela
silvestre nao respeita divisa municipal - o ciclo enzootico ocorre em manchas de
mata que atravessam varios municipios, e um surto detectado em A e frequentemente
o mesmo foco que apareceria em B se a equipe de vigilancia tivesse ido a B.

Com prevalencia de 0,7% ao nivel municipal, o desfecho e raro demais para
sustentar uma lista de prioridade util. Agregar em unidades maiores troca
resolucao espacial por densidade de sinal, e essa e uma troca que pode valer:
um alerta regional correto e mais util que um alerta municipal inutil.

Tres escalas entram, todas oficiais:

    microrregiao       ~558 unidades, agrupamento tradicional do IBGE
    regiao imediata    ~510 unidades, a divisao vigente desde 2017, construida
                       em torno de centros urbanos e da rede de deslocamento -
                       que e como a populacao humana de fato circula
    mesorregiao        ~137 unidades, escala mais grossa

A regiao imediata e a preferida a priori: ela e definida pela area de influencia
de um centro urbano, e portanto pelo padrao de deslocamento das pessoas, que e o
que determina onde um caso humano infectado na mata sera notificado.

Fonte
-----
API de localidades do IBGE. E consulta publica, sem chave, e devolve a hierarquia
completa dos 5.571 municipios. Fica gravada em disco para que a construcao do
painel nao dependa de rede.

Uso:
    python -m yfvrisk.ingest.regionalizacao
"""

from __future__ import annotations

import gzip
import json
import urllib.request

import pandas as pd

from yfvrisk import config as cfg
from yfvrisk.ingest.municipios import carregar_referencia

URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios?orderBy=id"
BRUTO = cfg.BRUTO / "ibge" / "municipios_hierarquia.json"
SAIDA = cfg.PROCESSADO / "regionalizacao.parquet"
DIAGNOSTICO = cfg.METRICAS / "f6_regionalizacao_diagnostico.json"


def baixar(forcar: bool = False) -> list:
    BRUTO.parent.mkdir(parents=True, exist_ok=True)
    if BRUTO.exists() and not forcar:
        return json.loads(BRUTO.read_text(encoding="utf-8"))
    print(f"[baixar] {URL}")
    # O servico responde comprimido mesmo sem Accept-Encoding, de modo que o
    # corpo precisa ser descomprimido antes de decodificar. Sem isso o erro sai
    # como UnicodeDecodeError, que despista.
    pedido = urllib.request.Request(URL, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(pedido, timeout=90) as r:
        corpo = r.read()
        magica = bytes([0x1F, 0x8B])
        if r.headers.get("Content-Encoding") == "gzip" or corpo[:2] == magica:
            corpo = gzip.decompress(corpo)
    dados = json.loads(corpo.decode("utf-8"))
    if len(dados) < 5500:
        raise RuntimeError(f"resposta com apenas {len(dados)} municipios")
    BRUTO.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    return dados


def construir() -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()
    dados = baixar()

    linhas = []
    for m in dados:
        micro = m.get("microrregiao") or {}
        meso = micro.get("mesorregiao") or {}
        imed = m.get("regiao-imediata") or {}
        inter = imed.get("regiao-intermediaria") or {}
        linhas.append({
            "cod_mun": str(m["id"]),
            "cod_microrregiao": str(micro.get("id", "")) or None,
            "nome_microrregiao": micro.get("nome"),
            "cod_mesorregiao": str(meso.get("id", "")) or None,
            "nome_mesorregiao": meso.get("nome"),
            "cod_regiao_imediata": str(imed.get("id", "")) or None,
            "nome_regiao_imediata": imed.get("nome"),
            "cod_regiao_intermediaria": str(inter.get("id", "")) or None,
            "nome_regiao_intermediaria": inter.get("nome"),
        })
    quadro = pd.DataFrame(linhas)

    # Alinha com a referencia municipal do projeto, que tem 5.570 - a API traz
    # 5.571 porque inclui Brasilia com tratamento proprio ou um codigo extinto.
    ref = carregar_referencia(com_geometria=False)
    faltando = sorted(set(ref["cod_mun"]) - set(quadro["cod_mun"]))
    sobrando = sorted(set(quadro["cod_mun"]) - set(ref["cod_mun"]))
    quadro = quadro[quadro["cod_mun"].isin(set(ref["cod_mun"]))].copy()

    if faltando:
        raise SystemExit(
            f"{len(faltando)} municipios da referencia sem hierarquia do IBGE: "
            f"{faltando[:10]}. Sem isso a agregacao perderia municipios em "
            f"silencio.")

    quadro.to_parquet(SAIDA, index=False)

    diag = {
        "fonte": URL,
        "municipios_na_api": len(dados),
        "municipios_apos_alinhamento": int(len(quadro)),
        "municipios_da_api_fora_da_referencia": sobrando,
        "unidades": {
            "microrregiao": int(quadro["cod_microrregiao"].nunique()),
            "mesorregiao": int(quadro["cod_mesorregiao"].nunique()),
            "regiao_imediata": int(quadro["cod_regiao_imediata"].nunique()),
            "regiao_intermediaria": int(quadro["cod_regiao_intermediaria"].nunique()),
        },
        "municipios_por_unidade": {
            escala: {
                "min": int(quadro.groupby(col).size().min()),
                "mediana": float(quadro.groupby(col).size().median()),
                "max": int(quadro.groupby(col).size().max()),
            }
            for escala, col in [("microrregiao", "cod_microrregiao"),
                                ("regiao_imediata", "cod_regiao_imediata"),
                                ("mesorregiao", "cod_mesorregiao")]
        },
    }
    DIAGNOSTICO.write_text(json.dumps(diag, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return quadro, diag


def carregar() -> pd.DataFrame:
    if not SAIDA.exists():
        construir()
    return pd.read_parquet(SAIDA)


if __name__ == "__main__":
    q, d = construir()
    print(f"regionalizacao: {len(q):,} municipios")
    for escala, n in d["unidades"].items():
        tam = d["municipios_por_unidade"].get(escala)
        extra = (f"  ({tam['min']}-{tam['max']} municipios, mediana "
                 f"{tam['mediana']:.0f})" if tam else "")
        print(f"  {escala:24s} {n:4d} unidades{extra}")
    if d["municipios_da_api_fora_da_referencia"]:
        print("fora da referencia:", d["municipios_da_api_fora_da_referencia"])
