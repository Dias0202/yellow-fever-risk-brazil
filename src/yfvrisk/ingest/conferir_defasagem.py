"""Alerta quando uma fonte manual fica defasada em relacao ao periodo do painel.

As fontes que exigem navegador ou login - TabNet, planilhas do IBGE, MapBiomas,
WorldClim - nao podem ser baixadas em integracao continua. Elas ficam versionadas
por checksum, e este script existe para que a defasagem seja VISIVEL em vez de
silenciosa.

E a diferenca entre "o painel vai ate 2025" e "o painel vai ate 2025 mas a
vacinacao para em 2022 e o clima em 2024". A segunda frase e a verdadeira, e
precisa aparecer em toda execucao.

Uso:  python -m yfvrisk.ingest.conferir_defasagem
Codigo de saida 1 se houver defasagem, para o fluxo poder falhar se quiser.
"""

from __future__ import annotations

import sys

import pandas as pd

from yfvrisk import config as cfg

# Cada entrada: nome, arquivo processado, coluna de ano, tolerancia em anos.
# A tolerancia reconhece que algumas fontes tem atraso normal de publicacao.
FONTES = [
    ("epidemiologia", "painel_epi_muni_ano.parquet", "ano", 0,
     "Ministerio da Saude, CGARB - exportacao manual"),
    ("vacinacao", "vacinacao_muni_ano.parquet", None, 1,
     "PNI/DATASUS TabNet - exportacao manual, http://tabnet.datasus.gov.br"),
    ("populacao", "populacao_muni_ano.parquet", None, 0,
     "IBGE estimativas - planilha anual"),
    ("clima", "clima_muni_mes.parquet", "ano", 1,
     "TerraClimate - automatizado, ver baixar_terraclimate.py"),
    ("floresta", "floresta_muni_ano.parquet", None, 1,
     "Hansen GFC - automatizado, ver baixar_hansen.py"),
]


def _ultimo_ano_observado(df: pd.DataFrame, nome: str):
    """Ultimo ano com dado REAL, nao apenas linha no painel.

    Distinguir isso importa: o painel tem linha para todo municipio em todo ano,
    inclusive nos anos em que a fonte nao existe e o valor foi interpolado ou
    carregado para frente.
    """
    if nome == "vacinacao":
        real = df[df["cobertura_disponivel"] == 1]
    elif nome == "populacao":
        real = df[df["populacao_observada"] == 1]
    elif nome == "floresta":
        real = df[(df["anterior_ao_hansen"] == 0) & (df["posterior_ao_hansen"] == 0)]
    else:
        real = df
    return int(real["ano"].max()) if len(real) else None


def main() -> int:
    print("## Defasagem das fontes\n")
    print(f"Periodo do painel: {cfg.ANO_INICIO}-{cfg.ANO_FIM}\n")
    print("| Fonte | Ultimo ano com dado | Defasagem | Situacao |")
    print("|---|---|---|---|")

    defasadas = []
    for nome, arquivo, _, tolerancia, origem in FONTES:
        caminho = cfg.PROCESSADO / arquivo
        if not caminho.exists():
            print(f"| {nome} | - | - | nao construido |")
            continue

        df = pd.read_parquet(caminho)
        ultimo = _ultimo_ano_observado(df, nome)
        if ultimo is None:
            print(f"| {nome} | - | - | sem dado observado |")
            continue

        atraso = cfg.ANO_FIM - ultimo
        if atraso > tolerancia:
            situacao = "**DEFASADA**"
            defasadas.append((nome, ultimo, atraso, origem))
        else:
            situacao = "em dia"
        print(f"| {nome} | {ultimo} | {atraso} ano(s) | {situacao} |")

    if defasadas:
        print("\n### Fontes que precisam de reexportacao manual\n")
        for nome, ultimo, atraso, origem in defasadas:
            print(f"- **{nome}**: ultimo dado em {ultimo}, {atraso} ano(s) atras. "
                  f"Origem: {origem}")
        print("\nEnquanto nao forem atualizadas, os anos finais do painel usam "
              "valores interpolados ou carregados para frente, e isso precisa "
              "constar de qualquer resultado divulgado.")
        return 1

    print("\nTodas as fontes dentro da tolerancia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
