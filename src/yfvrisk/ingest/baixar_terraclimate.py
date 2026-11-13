"""Download reproduzivel do TerraClimate.

Cada arquivo e baixado da URL oficial, conferido por tamanho minimo e registrado
com sha256. Sem essa verificacao, um download truncado (ex.: um tmax_1994.nc de
0 bytes) passa despercebido e vira um ano inteiro de clima ausente.

Este modulo torna a fonte declarada e verificavel.

Uso:
    python -m yfvrisk.ingest.baixar_terraclimate --verificar
    python -m yfvrisk.ingest.baixar_terraclimate --baixar tmax 1994
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from yfvrisk import config as cfg

BASE = "https://climate.northwestknowledge.net/TERRACLIMATE-DATA"
VARIAVEIS = ["ppt", "tmax", "tmin", "vpd", "ws"]

# Onde os arquivos rebaixados sao gravados, separados da copia local recebida.
DESTINO = cfg.BRUTO / "terraclimate"

# Tamanho minimo plausivel. Um arquivo anual global do TerraClimate nunca e menor
# que isso; abaixo disso e download truncado.
BYTES_MINIMOS = 10_000_000


def url_de(variavel: str, ano: int) -> str:
    return f"{BASE}/TerraClimate_{variavel}_{ano}.nc"


def sha256(caminho: Path, blocos: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as fh:
        for b in iter(lambda: fh.read(blocos), b""):
            h.update(b)
    return h.hexdigest()


def verificar_integridade(diretorio: Path = None) -> dict:
    """Varre os arquivos presentes e sinaliza os truncados ou ausentes."""
    diretorio = diretorio or (cfg.FONTE_MANUAL / "TerraClimate_raw")
    relatorio = {"diretorio": str(diretorio), "problemas": [], "ok": 0}

    for var in VARIAVEIS:
        for arq in sorted(diretorio.glob(f"{var}_*.nc")):
            tam = arq.stat().st_size
            if tam < BYTES_MINIMOS:
                relatorio["problemas"].append({
                    "arquivo": arq.name, "bytes": tam,
                    "motivo": "truncado ou vazio",
                    "url_para_recuperar": url_de(var, int(arq.stem.split("_")[1])),
                })
            else:
                relatorio["ok"] += 1
    return relatorio


def baixar(variavel: str, ano: int, forcar: bool = False) -> Path:
    DESTINO.mkdir(parents=True, exist_ok=True)
    destino = DESTINO / f"{variavel}_{ano}.nc"

    if destino.exists() and destino.stat().st_size >= BYTES_MINIMOS and not forcar:
        print(f"[pular] {destino.name} ja existe e tem tamanho plausivel")
        return destino

    url = url_de(variavel, ano)
    print(f"[baixar] {url}")

    parcial = destino.with_suffix(".nc.parcial")

    def progresso(bloco, tam_bloco, total):
        if total > 0 and bloco % 200 == 0:
            pct = 100 * bloco * tam_bloco / total
            print(f"\r  {min(pct, 100):5.1f}%  "
                  f"{bloco * tam_bloco / 1e6:.0f}/{total / 1e6:.0f} MB", end="")

    urllib.request.urlretrieve(url, parcial, reporthook=progresso)
    print()

    tam = parcial.stat().st_size
    if tam < BYTES_MINIMOS:
        parcial.unlink()
        raise RuntimeError(f"download truncado: {tam} bytes")

    # So promove a definitivo depois de conferir: sem isso, um arquivo de 0 byte
    # seria aceito em silencio.
    parcial.replace(destino)

    registro = {
        "arquivo": destino.name, "url": url, "bytes": tam,
        "sha256": sha256(destino),
    }
    manifesto = DESTINO / "MANIFESTO.jsonl"
    with manifesto.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"[ok] {destino.name}  {tam / 1e6:.0f} MB  sha256 {registro['sha256'][:16]}...")
    return destino


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verificar", action="store_true",
                    help="varre os arquivos existentes e lista os truncados")
    ap.add_argument("--baixar", nargs=2, metavar=("VARIAVEL", "ANO"),
                    help="baixa um arquivo especifico")
    ap.add_argument("--recuperar-truncados", action="store_true",
                    help="baixa todos os arquivos sinalizados como truncados")
    args = ap.parse_args()

    if args.verificar or args.recuperar_truncados:
        rel = verificar_integridade()
        print(f"[verificacao] {rel['ok']} arquivos integros, "
              f"{len(rel['problemas'])} com problema")
        for p in rel["problemas"]:
            print(f"  {p['arquivo']:20s} {p['bytes']:>12,} bytes  {p['motivo']}")
        if args.recuperar_truncados:
            for p in rel["problemas"]:
                var, ano = p["arquivo"].replace(".nc", "").split("_")
                baixar(var, int(ano))
        return 0

    if args.baixar:
        baixar(args.baixar[0], int(args.baixar[1]))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
