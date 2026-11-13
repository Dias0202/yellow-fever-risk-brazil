"""Caminhos e constantes do projeto.

Tudo relativo a raiz do repositorio: nenhum caminho absoluto, para que o projeto
rode em qualquer maquina.
"""

from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

DADOS = RAIZ / "data"
BRUTO = DADOS / "raw"

# Entradas que nao tem download automatizado (exportacoes do Ministerio da Saude,
# PNI, IBGE, MapBiomas, malha municipal) e as camadas pesadas de clima e
# floresta. Ver data/SOURCES.yml e a secao "Reproducibility" do README.
FONTE_MANUAL = BRUTO

INTERMEDIARIO = DADOS / "interim"
PROCESSADO = DADOS / "processed"

RESULTADOS = RAIZ / "results"
METRICAS = RESULTADOS / "metrics"
FIGURAS = RESULTADOS / "figures"
TABELAS = RESULTADOS / "tables"
MAPAS = RESULTADOS / "maps"

# --- Malha municipal ------------------------------------------------------
# IBGE 2022, EPSG:4674 (SIRGAS 2000). Escolhida e declarada como a malha unica
# do projeto, para nao misturar vintages de malha.
MALHA_MUNICIPIOS = FONTE_MANUAL / "BR_Municipios_2022"
MALHA_UF = FONTE_MANUAL / "BR_UF_2024"
CRS_MALHA = "EPSG:4674"

# CRS metrico para DISTANCIA. Policonica do SIRGAS 2000, projecao oficial do IBGE
# para o Brasil: distancias calculadas sobre lat/lon nao projetada sairiam em graus.
CRS_METRICO = "EPSG:5880"

# CRS para AREA. A policonica NAO preserva area: somada sobre os 5.570 municipios
# ela superestima o territorio brasileiro em 1,13%. A Albers equal-area para a
# America do Sul reproduz o atributo oficial do IBGE com desvio de 0,00%.
CRS_AREA = (
    "+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 "
    "+x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
)

# Feicoes da malha do IBGE que NAO sao municipios: dois corpos d'agua no RS.
# Sao a origem da divergencia 5.570 / 5.571 / 5.572. Truncados a 6 digitos, ambos
# colapsam no mesmo codigo 430000, o que explica a contagem intermediaria de 5.571.
NAO_MUNICIPIOS = {"4300001", "4300002"}  # Lagoa Mirim, Lagoa dos Patos
N_MUNICIPIOS = 5570

# --- Recorte temporal -----------------------------------------------------
ANO_INICIO = 1994
ANO_FIM = 2025

# Vigilancia de epizootias em primatas nao humanos so existe a partir de 2000.
# Antes disso, ausencia de registro nao e evidencia de ausencia de circulacao.
ANO_INICIO_EPIZOOTIAS = 2000

# Anos sem nenhum registro nacional de epizootia. Sao lacunas de vigilancia, nao
# zeros verdadeiros, e nao devem ser tratados como ausencia de circulacao.
ANOS_SEM_VIGILANCIA_EPIZOOTIAS = {2004, 2005, 2006, 2010, 2011, 2012, 2013}

def garantir_diretorios() -> None:
    for d in (BRUTO, INTERMEDIARIO, PROCESSADO, METRICAS, FIGURAS, TABELAS, MAPAS):
        d.mkdir(parents=True, exist_ok=True)
