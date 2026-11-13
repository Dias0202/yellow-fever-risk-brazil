"""Download dos tiles do Hansen Global Forest Change que faltam.

Contexto
--------
Os tiles do Hansen sao organizados em faixas de 10 graus de latitude. Enumerar
apenas 00N, 10S, 20S e 30S - o recorte "obvio" para o Brasil - deixa de fora a
faixa 10N, e o Brasil vai ate 5,27 graus NORTE, no monte Caburai, em Roraima.

Sem a faixa 10N:

  24 municipios sem NENHUMA cobertura de Hansen    230.464 km2
  20 municipios cobertos apenas em parte           822.141 km2
  ------------------------------------------------------------
  total afetado                                  1.052.605 km2  = 12,4% do pais

Sao os 13 municipios de Roraima e 11 do Amapa inteiramente ao norte do equador,
mais gigantes amazonicos parcialmente cortados, como Barcelos (122.461 km2),
Sao Gabriel da Cachoeira (109.191 km2) e Oriximina (107.614 km2).

Qualquer imputacao pela media da coluna atribuiria a Roraima e ao Amapa a
cobertura florestal MEDIA DO BRASIL - em plena Amazonia - e contaminaria as
variaveis de cobertura arborea, perda e ganho.

Este modulo baixa a faixa 10N. As camadas `first` e `last`, que somam
121 dos 127 GB, nao sao baixadas: sao composicoes Landsat de reflectancia e nao
entram em nenhuma metrica de floresta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request

from yfvrisk import config as cfg

BASE = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12"
VERSAO = "GFC-2024-v1.12"

# Camadas necessarias para metricas de floresta. `first` e `last` ficam de fora.
CAMADAS = ["treecover2000", "lossyear", "gain", "datamask"]

# Faixa 10N: cobre o territorio brasileiro ao norte do equador.
# Roraima fica entre 60W e 58W; o Amapa entre 54W e 50W.
TILES_FALTANTES = ["10N_050W", "10N_060W", "10N_070W", "10N_080W"]

DESTINO = cfg.BRUTO / "hansen"
BYTES_MINIMOS = 100_000


def url_de(camada: str, tile: str) -> str:
    return f"{BASE}/Hansen_{VERSAO}_{camada}_{tile}.tif"


def sha256(caminho, blocos: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as fh:
        for b in iter(lambda: fh.read(blocos), b""):
            h.update(b)
    return h.hexdigest()


def baixar(camada: str, tile: str, forcar: bool = False):
    DESTINO.mkdir(parents=True, exist_ok=True)
    nome = f"Hansen_{VERSAO}_{camada}_{tile}.tif"
    destino = DESTINO / nome

    if destino.exists() and destino.stat().st_size >= BYTES_MINIMOS and not forcar:
        print(f"[pular]  {nome}")
        return destino

    url = url_de(camada, tile)
    parcial = destino.with_suffix(".tif.parcial")
    print(f"[baixar] {nome}")

    def progresso(bloco, tam, total):
        if total > 0 and bloco % 400 == 0:
            print(f"\r  {min(100 * bloco * tam / total, 100):5.1f}%  "
                  f"{bloco * tam / 1e6:.0f}/{total / 1e6:.0f} MB", end="")

    try:
        urllib.request.urlretrieve(url, parcial, reporthook=progresso)
    except urllib.error.HTTPError as e:
        print(f"\n[erro]   {nome}: HTTP {e.code} - tile pode nao existir na colecao")
        if parcial.exists():
            parcial.unlink()
        return None
    print()

    tam = parcial.stat().st_size
    if tam < BYTES_MINIMOS:
        parcial.unlink()
        print(f"[erro]   {nome}: download truncado ({tam} bytes)")
        return None

    parcial.replace(destino)
    with (DESTINO / "MANIFESTO.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"arquivo": nome, "url": url, "bytes": tam,
                             "sha256": sha256(destino)}, ensure_ascii=False) + "\n")
    print(f"[ok]     {nome}  {tam / 1e6:.0f} MB")
    return destino


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camadas", nargs="*", default=CAMADAS)
    ap.add_argument("--tiles", nargs="*", default=TILES_FALTANTES)
    args = ap.parse_args()

    total, falhas = 0, 0
    for camada in args.camadas:
        for tile in args.tiles:
            total += 1
            if baixar(camada, tile) is None:
                falhas += 1
    print(f"\n[resumo] {total - falhas}/{total} arquivos disponiveis em {DESTINO}")
    return 0 if falhas == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
