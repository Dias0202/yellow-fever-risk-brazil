"""Testes do painel de dados.

Cada teste aqui trava uma invariante do painel, para que ela nao possa regredir
em silencio. Sao os modos de falha silenciosa mais provaveis num painel montado
a partir de dez fontes heterogeneas.

Os testes que dependem do painel construido (`data/processed/`) sao pulados,
com o motivo declarado, quando ele nao existe.

Rodar:  python -m pytest src/tests -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from yfvrisk import config as cfg

ANOS_ESPERADOS = cfg.ANO_FIM - cfg.ANO_INICIO + 1
LINHAS_ESPERADAS = cfg.N_MUNICIPIOS * ANOS_ESPERADOS


def _carregar(nome):
    caminho = cfg.PROCESSADO / nome
    if not caminho.exists():
        pytest.skip(f"{nome} ainda nao foi construido")
    return pd.read_parquet(caminho)


# --------------------------------------------------------------- referencia
class TestReferenciaMunicipal:
    def test_conta_5570_municipios(self):
        """5.572 na malha do IBGE menos os dois corpos d'agua do RS."""
        ref = _carregar("municipios_referencia.parquet")
        assert len(ref) == cfg.N_MUNICIPIOS

    def test_nao_contem_corpos_dagua(self):
        """Lagoa Mirim e Lagoa dos Patos nao sao municipios."""
        ref = _carregar("municipios_referencia.parquet")
        assert not set(ref["cod_mun"]) & cfg.NAO_MUNICIPIOS

    def test_codigos_tem_sete_digitos(self):
        """Truncar para seis digitos foi o que fundiu os dois corpos d'agua."""
        ref = _carregar("municipios_referencia.parquet")
        assert ref["cod_mun"].str.fullmatch(r"\d{7}").all()

    def test_codigos_sao_unicos(self):
        ref = _carregar("municipios_referencia.parquet")
        assert not ref["cod_mun"].duplicated().any()

    def test_27_unidades_federativas(self):
        ref = _carregar("municipios_referencia.parquet")
        assert ref["uf"].nunique() == 27

    def test_area_bate_com_o_ibge(self):
        """Detecta uso de projecao nao equivalente em area.

        A policonica EPSG:5880, que e a projecao oficial do IBGE para distancia,
        superestima o territorio em 1,13% quando usada para area.
        """
        ref = _carregar("municipios_referencia.parquet")
        desvio = abs(ref["area_km2"].sum() - ref["area_km2_ibge"].sum())
        assert desvio / ref["area_km2_ibge"].sum() < 0.01

    def test_area_total_plausivel(self):
        ref = _carregar("municipios_referencia.parquet")
        assert 8.4e6 < ref["area_km2"].sum() < 8.6e6


# --------------------------------------------------------------- estrutura
@pytest.mark.parametrize("arquivo", [
    "painel_epi_muni_ano.parquet",
    "vacinacao_muni_ano.parquet",
    "populacao_muni_ano.parquet",
])
class TestEstruturaDoPainel:
    def test_uma_linha_por_municipio_ano(self, arquivo):
        df = _carregar(arquivo)
        assert len(df) == LINHAS_ESPERADAS

    def test_sem_duplicatas(self, arquivo):
        df = _carregar(arquivo)
        assert not df.duplicated(subset=["cod_mun", "ano"]).any()

    def test_cobre_todo_o_periodo(self, arquivo):
        df = _carregar(arquivo)
        assert df["ano"].min() == cfg.ANO_INICIO
        assert df["ano"].max() == cfg.ANO_FIM

    def test_todos_os_municipios_presentes(self, arquivo):
        df = _carregar(arquivo)
        assert df["cod_mun"].nunique() == cfg.N_MUNICIPIOS


# --------------------------------------------------------------- epidemiologia
class TestEpidemiologia:
    def test_desfecho_binario(self):
        p = _carregar("painel_epi_muni_ano.parquet")
        assert set(p["yf_event"].unique()) <= {0, 1}

    def test_desfecho_e_uniao_dos_componentes(self):
        p = _carregar("painel_epi_muni_ano.parquet")
        esperado = ((p["caso_humano"] == 1) | (p["epizootia"] == 1)).astype(int)
        assert (p["yf_event"] == esperado).all()

    def test_componentes_ficam_separados(self):
        """Agregar humanos e epizootias e o que faz a vacinacao parecer inocua."""
        p = _carregar("painel_epi_muni_ano.parquet")
        assert {"caso_humano", "epizootia"} <= set(p.columns)

    def test_epizootias_nao_existem_antes_de_2000(self):
        p = _carregar("painel_epi_muni_ano.parquet")
        antes = p[p["ano"] < cfg.ANO_INICIO_EPIZOOTIAS]
        assert antes["epizootia"].sum() == 0

    def test_lacunas_de_vigilancia_marcadas(self):
        """Zero observado e ausencia de observacao nao podem ser a mesma coisa."""
        p = _carregar("painel_epi_muni_ano.parquet")
        assert "epizootia_observavel" in p.columns
        for ano in cfg.ANOS_SEM_VIGILANCIA_EPIZOOTIAS:
            linhas = p[p["ano"] == ano]
            if len(linhas):
                assert linhas["epizootia_observavel"].sum() == 0, \
                    f"{ano} nao tem registro nacional e deveria estar marcado"

    def test_prevalencia_anual_e_rara(self):
        """A prevalencia anual e muito menor que os 4,12% do agregado por periodo."""
        p = _carregar("painel_epi_muni_ano.parquet")
        assert 0.001 < p["yf_event"].mean() < 0.02

    def test_pico_de_2017_2018_presente(self):
        p = _carregar("painel_epi_muni_ano.parquet")
        por_ano = p.groupby("ano")["yf_event"].sum()
        assert por_ano.loc[2017] > por_ano.loc[2015] * 5


# --------------------------------------------------------------- vacinacao
class TestVacinacao:
    def test_imunidade_no_intervalo_unitario(self):
        v = _carregar("vacinacao_muni_ano.parquet")
        assert v["imunidade_acumulada"].between(0, 1).all()

    def test_ausencia_nao_vira_suscetibilidade_total(self):
        """Cobertura ausente nao pode virar populacao 100% suscetivel.

        A ausencia e maior em municipios pequenos e remotos; aqui ela so deixa
        de incrementar a imunidade acumulada.
        """
        v = _carregar("vacinacao_muni_ano.parquet")
        sem_dado = v[(v["cobertura_disponivel"] == 0) & (v["ano"] >= 2010)]
        if len(sem_dado):
            assert sem_dado["imunidade_acumulada"].max() > 0, \
                "linhas sem dado recente estao com imunidade zerada"

    def test_imunidade_nao_cai_sem_motivo(self):
        """So a renovacao populacional pode reduzir a imunidade acumulada."""
        v = _carregar("vacinacao_muni_ano.parquet").sort_values(["cod_mun", "ano"])
        d = v.groupby("cod_mun")["imunidade_acumulada"].diff()
        assert d.min() > -0.05

    def test_marcador_de_valores_implausiveis(self):
        v = _carregar("vacinacao_muni_ano.parquet")
        assert "cobertura_implausivel" in v.columns

    def test_imunidade_de_modelagem_e_defasada(self):
        """A imunidade que entra no modelo nao pode ser a do proprio ano.

        A cobertura do PNI e registrada no ano do surto, em resposta a ele: o
        estudo de evento mostra salto de 4,5x em k = 0. Usar a
        contemporanea coloca uma variavel a jusante do desfecho no preditor.
        """
        from yfvrisk.ingest.vacinacao import DEFASAGEM_IMUNIDADE

        v = _carregar("vacinacao_muni_ano.parquet").sort_values(["cod_mun", "ano"])
        esperado = v.groupby("cod_mun")["imunidade_acumulada"].shift(
            DEFASAGEM_IMUNIDADE)
        assert v["imunidade_defasada2"].equals(esperado)

    def test_imunidade_contemporanea_fora_do_modelo(self):
        """`imunidade_acumulada` e `suscetivel_pct` ficam fora do grupo IMUNIDADE.

        A primeira por estar a jusante do desfecho; a segunda por ser exatamente
        o complemento da primeira, o que dividia a importancia de Gini entre duas
        colunas com a mesma informacao.
        """
        from yfvrisk.features.painel import GRUPOS

        assert "imunidade_acumulada" not in GRUPOS["IMUNIDADE"]
        assert "suscetivel_pct" not in GRUPOS["IMUNIDADE"]

    def test_imunidade_relativa_centrada_no_ano(self):
        """O contraste transversal soma zero dentro de cada ano, por construcao."""
        v = _carregar("vacinacao_muni_ano.parquet")
        medias = v.groupby("ano")["imunidade_relativa2"].mean().abs()
        assert medias.max() < 1e-9

    def test_registro_no_pni_nao_retrocede(self):
        """A contagem de anos de presenca na serie do PNI e cumulativa."""
        v = _carregar("vacinacao_muni_ano.parquet").sort_values(["cod_mun", "ano"])
        d = v.groupby("cod_mun")["anos_com_registro_pni"].diff()
        assert d.min(skipna=True) >= 0


# --------------------------------------------------------------- populacao
class TestPopulacao:
    def test_sem_populacao_ausente(self):
        p = _carregar("populacao_muni_ano.parquet")
        assert p["populacao"].isna().sum() == 0

    def test_populacao_positiva(self):
        p = _carregar("populacao_muni_ano.parquet")
        assert (p["populacao"] > 0).all()

    def test_total_nacional_plausivel(self):
        """Conferencia contra os valores publicados pelo IBGE."""
        p = _carregar("populacao_muni_ano.parquet")
        total_2020 = p.loc[p["ano"] == 2020, "populacao"].sum()
        assert 210e6 < total_2020 < 214e6

    def test_interpolacao_marcada(self):
        p = _carregar("populacao_muni_ano.parquet")
        assert "populacao_observada" in p.columns
        assert p.loc[p["ano"] == 2010, "populacao_observada"].sum() == 0

    def test_densidade_em_habitantes_por_km2(self):
        """Detecta area em grau quadrado.

        Com area em graus, a densidade do Brasil sairia na casa dos milhares.
        """
        p = _carregar("populacao_muni_ano.parquet")
        mediana = p.loc[p["ano"] == 2020, "densidade_hab_km2"].median()
        assert 1 < mediana < 200


# --------------------------------------------------------------- clima
class TestClima:
    def test_uma_linha_por_municipio_ano_mes(self):
        c = _carregar("clima_muni_mes.parquet")
        assert not c.duplicated(subset=["cod_mun", "ano", "mes"]).any()

    def test_sem_valores_ausentes(self):
        """Fernando de Noronha ficava fora do recorte de longitude."""
        c = _carregar("clima_muni_mes.parquet")
        for v in ["ppt", "tmax", "tmin", "vpd", "ws"]:
            assert c[v].isna().sum() == 0, f"{v} tem valores ausentes"

    def test_todos_os_municipios_cobertos(self):
        c = _carregar("clima_muni_mes.parquet")
        assert c["cod_mun"].nunique() == cfg.N_MUNICIPIOS

    def test_temperaturas_em_celsius_plausiveis(self):
        c = _carregar("clima_muni_mes.parquet")
        assert -5 < c["tmin"].min() and c["tmax"].max() < 50
        assert (c["tmax"] >= c["tmin"]).all()

    def test_precipitacao_mensal_nao_anual(self):
        """O clima mensal guarda precipitacao mensal, nao acumulado anual."""
        c = _carregar("clima_muni_mes.parquet")
        assert c["ppt"].max() < 2000
        assert 50 < c["ppt"].mean() < 250

    def test_fernando_de_noronha_tem_dado(self):
        c = _carregar("clima_muni_mes.parquet")
        fn = c[c["cod_mun"] == "2605459"]
        assert len(fn) > 0 and fn["tmax"].notna().all()


# --------------------------------------------------------------- floresta
class TestFloresta:
    def test_norte_do_equador_coberto(self):
        """A faixa 10N do Hansen precisa estar baixada.

        Sem ela, Roraima e Amapa ficam sem dado e recebem a media nacional.
        """
        f = _carregar("floresta_muni_ano.parquet")
        ref = _carregar("municipios_referencia.parquet")
        norte = ref[ref["uf"].isin(["RR", "AP"])]["cod_mun"]
        cobertura = (f[f["cod_mun"].isin(norte)]
                     .groupby("cod_mun")["hansen_cobertura"].first())
        assert cobertura.min() > 0.5, \
            f"{(cobertura <= 0.5).sum()} municipios de RR/AP mal cobertos"

    def test_perda_acumulada_nao_decresce(self):
        f = _carregar("floresta_muni_ano.parquet").sort_values(["cod_mun", "ano"])
        d = f.groupby("cod_mun")["perda_acumulada_km2"].diff()
        assert d.min() >= -1e-6

    def test_floresta_nao_negativa(self):
        f = _carregar("floresta_muni_ano.parquet")
        assert (f["floresta_km2"] >= 0).all()

    def test_cobertura_florestal_e_fracao(self):
        f = _carregar("floresta_muni_ano.parquet")
        assert f["cobertura_florestal"].max() <= 1.05

    def test_sem_perda_antes_de_2001(self):
        """Serie causal: nenhuma linha enxerga perda posterior ao proprio ano."""
        f = _carregar("floresta_muni_ano.parquet")
        assert f.loc[f["ano"] < 2001, "perda_acumulada_km2"].sum() == 0


class TestGruposDeVariaveis:
    """GRUPOS precisa estar completo no import, nao so depois de construir().

    Se CLIMA_DEFASADO fosse preenchido dentro de construir(), quem importasse
    GRUPOS sem construir o painel receberia a lista vazia e treinaria com 25
    variaveis em vez de 33, perdendo as oito defasagens climaticas sem nenhum
    aviso.
    """

    def test_clima_defasado_populado_no_import(self):
        from yfvrisk.features.painel import GRUPOS
        assert GRUPOS["CLIMA_DEFASADO"], "CLIMA_DEFASADO vazio no import"

    def test_nenhum_grupo_vazio(self):
        from yfvrisk.features.painel import GRUPOS
        vazios = [nome for nome, cols in GRUPOS.items() if not cols]
        assert not vazios, f"grupos vazios no import: {vazios}"

    def test_grupos_existem_no_painel(self):
        from yfvrisk.features.painel import GRUPOS
        painel = _carregar("painel_analitico.parquet")
        faltando = {nome: [c for c in cols if c not in painel.columns]
                    for nome, cols in GRUPOS.items()}
        faltando = {k: v for k, v in faltando.items() if v}
        assert not faltando, f"declaradas em GRUPOS e ausentes do painel: {faltando}"

    def test_sem_variavel_repetida_entre_grupos(self):
        from yfvrisk.features.painel import GRUPOS
        todas = [c for cols in GRUPOS.values() for c in cols]
        assert len(todas) == len(set(todas)), "variavel declarada em dois grupos"
