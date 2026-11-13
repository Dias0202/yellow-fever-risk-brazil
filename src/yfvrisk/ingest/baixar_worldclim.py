"""Download reproduzivel do WorldClim 2.1 - linha de base e projecoes CMIP6.

Tres decisoes de desenho; este modulo trata das duas primeiras e a terceira e
do modulo de cenarios.

1. Varios GCMs. Um unico GCM nao quantifica a incerteza estrutural entre
   modelos, que no horizonte de 2021-2040 e a maior fonte de incerteza de
   todas. Aqui sao cinco GCMs,
   escolhidos por cobrirem a faixa de sensibilidade climatica de equilibrio do
   CMIP6, de ~1,9 a ~5,4 K - ver GCMS abaixo.

2. Mesma resolucao nos dois lados da anomalia. Com a linha de base a 2,5
   arc-min e o futuro a 10 arc-min, a diferenca entre as duas misturaria
   mudanca climatica com mudanca de resolucao. Aqui as duas pontas do
   WorldClim ficam a 10 arc-min, e a anomalia resultante e que e aplicada sobre a
   linha de base observada do TerraClimate, que e a do painel. E o metodo delta,
   e ele tem a propriedade de cancelar o vies sistematico do modelo: o que se
   transporta para o futuro e a MUDANCA projetada, nao o nivel projetado.

3. As variaveis do futuro passam pela mesma funcao que as deriva do observado,
   para que o modelo leia exatamente as colunas em que foi treinado.
   Tratado em `yfvrisk/scenarios/`.

O WorldClim futuro traz apenas tmin, tmax e prec. O painel usa tambem vpd e ws,
que nao tem projecao nesta fonte; o que se faz com elas esta declarado no modulo
de cenarios, e nao aqui.

Uso:
    python -m yfvrisk.ingest.baixar_worldclim --listar
    python -m yfvrisk.ingest.baixar_worldclim --linha-de-base
    python -m yfvrisk.ingest.baixar_worldclim --futuro
    python -m yfvrisk.ingest.baixar_worldclim --verificar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from yfvrisk import config as cfg

BASE_FUTURO = "https://geodata.ucdavis.edu/cmip6"
BASE_HISTORICO = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"

# 10 arc-min nas duas pontas. A escolha e deliberada e nao e por economia de
# banda: o que se extrai daqui e uma ANOMALIA, e anomalia climatica e um campo
# suave - varia em escala de centenas de quilometros, nao de quilometros. O
# detalhe espacial fino vem da linha de base observada do TerraClimate, que ja
# esta no painel a 1/24 de grau. Baixar o futuro a 30 arc-sec acrescentaria 400x
# o volume para interpolar a mesma informacao.
RESOLUCAO = "10m"

# Cinco GCMs cobrindo a faixa de sensibilidade climatica de equilibrio (ECS) do
# CMIP6. O criterio e esse, e nao disponibilidade ou popularidade: um ensemble
# que so contivesse modelos de sensibilidade media reportaria uma incerteza
# falsamente estreita. Os valores de ECS sao os do IPCC AR6, tabela 7.SM.5.
#
#   INM-CM5-0        ECS ~1,9 K   extremo inferior do CMIP6
#   MIROC6           ECS ~2,6 K   sensibilidade intermediaria
#   MPI-ESM1-2-HR    ECS ~3,0 K   proximo da melhor estimativa do AR6
#   EC-Earth3-Veg    ECS ~4,3 K   metade superior
#   UKESM1-0-LL      ECS ~5,3 K   extremo superior do CMIP6
GCMS = ["INM-CM5-0", "MIROC6", "MPI-ESM1-2-HR", "EC-Earth3-Veg", "UKESM1-0-LL"]

SSPS = ["ssp126", "ssp245", "ssp370", "ssp585"]

# 2021-2040 e o horizonte de projecao do projeto. Os demais ficam declarados para que a
# extensao seja um argumento, e nao uma reescrita.
PERIODOS = ["2021-2040"]

# tmin, tmax e prec sao mensais, 12 bandas por arquivo. bioc nao entra: as
# bioclimaticas do WorldClim seriam derivadas com definicao diferente da que o
# painel usa, e foi exatamente essa mistura de convencoes que produziu as colunas
# `Bio1_raw...Bio15_raw` que o modelo final nao le.
VARIAVEIS = ["tmin", "tmax", "prec"]

DESTINO = cfg.BRUTO / "worldclim"
MANIFESTO = DESTINO / "MANIFESTO.jsonl"

# Abaixo disso e download truncado. O menor arquivo legitimo do conjunto e o
# prec futuro, na casa dos 6 MB.
BYTES_MINIMOS = 3_000_000


def url_futuro(gcm: str, ssp: str, variavel: str, periodo: str) -> str:
    return (f"{BASE_FUTURO}/{RESOLUCAO}/{gcm}/{ssp}/"
            f"wc2.1_{RESOLUCAO}_{variavel}_{gcm}_{ssp}_{periodo}.tif")


def url_historico(variavel: str) -> str:
    return f"{BASE_HISTORICO}/wc2.1_{RESOLUCAO}_{variavel}.zip"


def sha256(caminho: Path, blocos: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as fh:
        for b in iter(lambda: fh.read(blocos), b""):
            h.update(b)
    return h.hexdigest()


def _registrar(destino: Path, url: str) -> dict:
    registro = {"arquivo": destino.name, "url": url,
                "bytes": destino.stat().st_size, "sha256": sha256(destino)}
    MANIFESTO.parent.mkdir(parents=True, exist_ok=True)
    with MANIFESTO.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return registro


def _baixar(url: str, destino: Path, forcar: bool = False) -> Path:
    """Baixa para arquivo parcial e so promove depois de conferir o tamanho.

    E a licao do TerraClimate: um arquivo de 0 byte foi aceito e ficou um ano no
    disco sem ninguem notar, porque nada conferia o que tinha chegado.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size >= BYTES_MINIMOS and not forcar:
        print(f"[pular] {destino.name} ja existe e tem tamanho plausivel")
        return destino

    print(f"[baixar] {url}")
    parcial = destino.with_suffix(destino.suffix + ".parcial")

    def progresso(bloco, tam_bloco, total):
        if total > 0 and bloco % 400 == 0:
            pct = 100 * bloco * tam_bloco / total
            print(f"\r  {min(pct, 100):5.1f}%  "
                  f"{bloco * tam_bloco / 1e6:.0f}/{total / 1e6:.0f} MB", end="")

    urllib.request.urlretrieve(url, parcial, reporthook=progresso)
    print()

    tam = parcial.stat().st_size
    if tam < BYTES_MINIMOS:
        parcial.unlink()
        raise RuntimeError(f"download truncado: {tam} bytes em {url}")

    parcial.replace(destino)
    reg = _registrar(destino, url)
    print(f"[ok] {destino.name}  {tam / 1e6:.0f} MB  sha256 {reg['sha256'][:16]}...")
    return destino


def baixar_linha_de_base(forcar: bool = False) -> list[Path]:
    """Normais de 1970-2000 do WorldClim, o denominador da anomalia.

    Vem em zip com 12 GeoTIFF mensais. E extraido aqui para que o modulo de
    cenarios leia rasters, e nao arquivos compactados.
    """
    saidas = []
    for var in VARIAVEIS:
        zipado = DESTINO / "historico" / f"wc2.1_{RESOLUCAO}_{var}.zip"
        _baixar(url_historico(var), zipado, forcar)
        pasta = DESTINO / "historico" / var
        pasta.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zipado) as z:
            z.extractall(pasta)
        tifs = sorted(pasta.glob("*.tif"))
        if len(tifs) != 12:
            raise RuntimeError(
                f"{var}: esperados 12 rasters mensais, encontrados {len(tifs)}")
        saidas.extend(tifs)
        print(f"[ok] {var}: 12 rasters mensais em {pasta}")
    return saidas


def baixar_futuro(gcms=None, ssps=None, periodos=None,
                  forcar: bool = False) -> list[Path]:
    gcms = gcms or GCMS
    ssps = ssps or SSPS
    periodos = periodos or PERIODOS

    saidas, falhas = [], []
    for gcm in gcms:
        for ssp in ssps:
            for periodo in periodos:
                for var in VARIAVEIS:
                    url = url_futuro(gcm, ssp, var, periodo)
                    destino = (DESTINO / "futuro" / gcm / ssp /
                               f"wc2.1_{RESOLUCAO}_{var}_{gcm}_{ssp}_{periodo}.tif")
                    try:
                        saidas.append(_baixar(url, destino, forcar))
                    except Exception as erro:
                        # Uma combinacao ausente na fonte nao pode derrubar o
                        # lote inteiro, mas tambem nao pode passar em silencio:
                        # um ensemble com um GCM a menos e um ensemble diferente.
                        falhas.append({"url": url, "erro": str(erro)})
                        print(f"[FALHA] {url}: {erro}")

    if falhas:
        print(f"\n[atencao] {len(falhas)} arquivos nao obtidos. "
              f"O ensemble esta incompleto e isso precisa constar do resultado.")
        (DESTINO / "FALHAS.json").write_text(
            json.dumps(falhas, indent=2, ensure_ascii=False), encoding="utf-8")
    return saidas


def verificar() -> dict:
    """Confere presenca e tamanho de tudo que o ensemble declara precisar."""
    esperados, ausentes, truncados = 0, [], []

    for var in VARIAVEIS:
        pasta = DESTINO / "historico" / var
        esperados += 12
        tifs = sorted(pasta.glob("*.tif")) if pasta.exists() else []
        if len(tifs) != 12:
            ausentes.append(f"historico/{var} ({len(tifs)} de 12 rasters)")

    for gcm in GCMS:
        for ssp in SSPS:
            for periodo in PERIODOS:
                for var in VARIAVEIS:
                    esperados += 1
                    p = (DESTINO / "futuro" / gcm / ssp /
                         f"wc2.1_{RESOLUCAO}_{var}_{gcm}_{ssp}_{periodo}.tif")
                    if not p.exists():
                        ausentes.append(str(p.relative_to(DESTINO)))
                    elif p.stat().st_size < BYTES_MINIMOS:
                        truncados.append(str(p.relative_to(DESTINO)))

    rel = {"resolucao": RESOLUCAO, "gcms": GCMS, "ssps": SSPS,
           "periodos": PERIODOS, "variaveis": VARIAVEIS,
           "arquivos_esperados": esperados,
           "ausentes": ausentes, "truncados": truncados,
           "completo": not ausentes and not truncados}
    return rel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listar", action="store_true",
                    help="mostra o ensemble declarado e as URLs que ele implica")
    ap.add_argument("--linha-de-base", action="store_true",
                    help="baixa as normais de 1970-2000")
    ap.add_argument("--futuro", action="store_true",
                    help="baixa o ensemble CMIP6 completo")
    ap.add_argument("--verificar", action="store_true",
                    help="confere presenca e tamanho de tudo")
    ap.add_argument("--forcar", action="store_true",
                    help="rebaixa mesmo o que ja existe")
    args = ap.parse_args()

    if args.listar:
        print(f"resolucao: {RESOLUCAO}")
        print(f"GCMs ({len(GCMS)}): {', '.join(GCMS)}")
        print(f"SSPs ({len(SSPS)}): {', '.join(SSPS)}")
        print(f"periodos: {', '.join(PERIODOS)}")
        print(f"variaveis: {', '.join(VARIAVEIS)}")
        n = len(GCMS) * len(SSPS) * len(PERIODOS) * len(VARIAVEIS)
        print(f"\n{n} rasters futuros + {12 * len(VARIAVEIS)} mensais historicos")
        print(f"exemplo: {url_futuro(GCMS[0], SSPS[0], VARIAVEIS[0], PERIODOS[0])}")
        return 0

    if args.linha_de_base:
        baixar_linha_de_base(args.forcar)
    if args.futuro:
        baixar_futuro(forcar=args.forcar)
    if args.verificar or args.linha_de_base or args.futuro:
        rel = verificar()
        print(f"\n[verificacao] {rel['arquivos_esperados']} arquivos esperados, "
              f"{len(rel['ausentes'])} ausentes, {len(rel['truncados'])} truncados")
        for a in rel["ausentes"][:20]:
            print(f"  ausente: {a}")
        for t in rel["truncados"]:
            print(f"  truncado: {t}")
        (cfg.METRICAS / "f5_worldclim_verificacao.json").write_text(
            json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0 if rel["completo"] else 1

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
