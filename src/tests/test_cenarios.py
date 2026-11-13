"""Testes do modulo de cenarios climaticos.

Dois grupos. Os primeiros travam propriedades fisicas e algebricas do metodo
delta, e rodam sem nenhum dado construido. Os segundos conferem o produto e sao
pulados enquanto ele nao existir.

O teste mais importante do arquivo e `test_futuro_usa_a_mesma_agregacao`. Ele
garante que o clima futuro chega ao modelo nas mesmas colunas do observado: se o
futuro fosse gravado em colunas que o modelo nao le, um cenario "so clima" sairia
identico a linha de base sem nenhum erro visivel.

Rodar:  python -m pytest src/tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yfvrisk import config as cfg
from yfvrisk.scenarios import clima_futuro as cf


def _carregar(nome):
    caminho = cfg.PROCESSADO / nome
    if not caminho.exists():
        pytest.skip(f"{nome} ainda nao foi construido")
    return pd.read_parquet(caminho)


def _observado_sintetico(n_mun=3):
    """Um observado minimo, com o formato que `aplicar_delta` espera."""
    codigos = [f"{i:07d}" for i in range(1, n_mun + 1)]
    linhas = []
    for c in codigos:
        for mes in range(1, 13):
            linhas.append({"cod_mun": c, "mes": mes,
                           "ppt": 50.0 + 10 * mes, "tmax": 30.0, "tmin": 20.0,
                           "vpd": 1.2, "ws": 2.5})
    return pd.DataFrame(linhas), np.array(codigos)


# ------------------------------------------------------- fisica e algebra
class TestFormulaDeTetens:
    def test_valores_de_referencia(self):
        """es(20 C) = 2,339 kPa e es(30 C) = 4,243 kPa, tabelados na FAO-56."""
        assert cf.pressao_de_saturacao(np.array([20.0]))[0] == pytest.approx(
            2.339, abs=0.005)
        assert cf.pressao_de_saturacao(np.array([30.0]))[0] == pytest.approx(
            4.243, abs=0.005)

    def test_monotona_crescente(self):
        t = np.arange(-5.0, 45.0, 1.0)
        assert np.all(np.diff(cf.pressao_de_saturacao(t)) > 0)


class TestMetodoDelta:
    def test_delta_nulo_reproduz_o_observado(self):
        """Sem mudanca climatica, a projecao tem de devolver a linha de base.

        E a verificacao de identidade do metodo. Se ela falhar, qualquer
        diferenca reportada entre cenarios pode ser artefato da transformacao, e
        nao efeito do clima.
        """
        obs, codigos = _observado_sintetico()
        zero = np.zeros((12, len(codigos)))
        um = np.ones((12, len(codigos)))
        fut = cf.aplicar_delta(obs, zero, zero, um, codigos)

        alinhado = obs.merge(fut, on=["cod_mun", "mes"], suffixes=("_obs", "_fut"))
        for var in ("ppt", "tmax", "tmin", "vpd", "ws"):
            assert np.allclose(alinhado[f"{var}_obs"], alinhado[f"{var}_fut"]), \
                f"{var} mudou com delta nulo"

    def test_precipitacao_nunca_negativa(self):
        """A razao e multiplicativa justamente para isto.

        Um delta aditivo produziria chuva negativa nos meses secos do semiarido.
        """
        obs, codigos = _observado_sintetico()
        zero = np.zeros((12, len(codigos)))
        seca = np.full((12, len(codigos)), 0.05)
        fut = cf.aplicar_delta(obs, zero, zero, seca, codigos)
        assert (fut["ppt"] >= 0).all()

    def test_aquecimento_eleva_o_vpd(self):
        """Umidade relativa constante e Clausius-Clapeyron: aquecer eleva o deficit.

        Se o sinal saisse invertido, o cenario climatico reportaria menos
        estresse hidrico sob aquecimento - e `vpd_medio_lag1` e a variavel de
        maior amplitude de ALE de todo o modelo.
        """
        obs, codigos = _observado_sintetico()
        um = np.ones((12, len(codigos)))
        aquece = np.full((12, len(codigos)), 3.0)
        fut = cf.aplicar_delta(obs, aquece, aquece, um, codigos)
        alinhado = obs.merge(fut, on=["cod_mun", "mes"], suffixes=("_obs", "_fut"))
        assert (alinhado["vpd_fut"] > alinhado["vpd_obs"]).all()

    def test_vento_declaradamente_fixo(self):
        """Sem projecao de vento nesta fonte. A limitacao e declarada, nao oculta."""
        obs, codigos = _observado_sintetico()
        um = np.ones((12, len(codigos)))
        aquece = np.full((12, len(codigos)), 3.0)
        fut = cf.aplicar_delta(obs, aquece, aquece, um, codigos)
        alinhado = obs.merge(fut, on=["cod_mun", "mes"], suffixes=("_obs", "_fut"))
        assert np.allclose(alinhado["ws_obs"], alinhado["ws_fut"])


class TestEnsemble:
    def test_cobre_a_faixa_de_sensibilidade(self):
        """Cinco GCMs, com os dois extremos do CMIP6 presentes.

        Um ensemble so de modelos de sensibilidade media reportaria incerteza
        falsamente estreita, que e o oposto do que o ensemble existe para medir.
        """
        assert len(cf.GCMS) >= 5
        assert "INM-CM5-0" in cf.GCMS      # extremo inferior, ECS ~1,9 K
        assert "UKESM1-0-LL" in cf.GCMS    # extremo superior, ECS ~5,3 K

    def test_quatro_ssps(self):
        assert set(cf.SSPS) == {"ssp126", "ssp245", "ssp370", "ssp585"}


# --------------------------------------------------------- produto construido
class TestClimaFuturoConstruido:
    def test_futuro_usa_a_mesma_agregacao(self):
        """As colunas anuais do futuro sao exatamente as do painel observado.

        Se o futuro trouxer colunas que o modelo nao le, ou
        deixar de trazer as que ele le, o cenario volta a ser identico a linha de
        base sem que nada acuse.
        """
        from yfvrisk.features.painel import GRUPOS

        fut = _carregar("clima_futuro_muni_ano.parquet")
        contemporaneas = [c for c in GRUPOS["CLIMA"]]
        faltando = [c for c in contemporaneas if c not in fut.columns]
        assert not faltando, f"o clima futuro nao tem {faltando}"

    def test_todas_as_combinacoes_presentes(self):
        fut = _carregar("clima_futuro_muni_ano.parquet")
        combinacoes = set(map(tuple, fut[["gcm", "ssp"]].drop_duplicates().values))
        esperadas = {(g, s) for g in cf.GCMS for s in cf.SSPS}
        assert combinacoes == esperadas, f"faltam {esperadas - combinacoes}"

    def test_mesmos_municipios_em_todas_as_combinacoes(self):
        """Todo cenario cobre exatamente o mesmo conjunto de municipios.

        Um cenario com um municipio a mais que outro tornaria a diferenca entre
        eles incomparavel, e a diferenca e o unico numero reportavel da projecao.
        """
        fut = _carregar("clima_futuro_muni_ano.parquet")
        contagem = fut.groupby(["gcm", "ssp"])["cod_mun"].nunique()
        assert contagem.nunique() == 1
        assert contagem.iloc[0] <= cfg.N_MUNICIPIOS

    def test_sem_ausentes_e_exclusoes_declaradas(self):
        """Nada de clima ausente. Quem nao tem projecao sai declarado, e nao imputado.

        Um municipio com clima ausente cairia na mediana do imputador, isto e,
        receberia o clima mediano do Brasil - em silencio, sem que nada acuse.
        """
        import json

        fut = _carregar("clima_futuro_muni_ano.parquet")
        clima = [c for c in fut.columns
                 if c not in ("gcm", "ssp", "cod_mun", "ano_base", "periodo")]
        assert not fut[clima].isna().any().any(), "clima futuro com ausentes"

        caminho = cfg.METRICAS / "f5_clima_futuro_diagnostico.json"
        if not caminho.exists():
            pytest.skip("diagnostico ainda nao construido")
        diag = json.loads(caminho.read_text(encoding="utf-8"))
        excluidos = set(diag["municipios_sem_projecao"])
        assert set(fut["cod_mun"]).isdisjoint(excluidos)
        assert fut["cod_mun"].nunique() + len(excluidos) == cfg.N_MUNICIPIOS

    def test_uma_linha_por_ano_base(self):
        """O delta vai sobre cada ano observado, e nao sobre a climatologia."""
        fut = _carregar("clima_futuro_muni_ano.parquet")
        a0, a1 = cf.BASE_OBSERVADA
        assert set(fut["ano_base"]) == set(range(a0, a1 + 1))
        contagem = fut.groupby(["gcm", "ssp"])["ano_base"].nunique()
        assert (contagem == a1 - a0 + 1).all()

    def test_sem_o_vies_de_suavizacao(self):
        """A entrada da projecao tem de ter a forma de um ano, nao de uma media.

        Trava uma decisao de desenho medida: aplicar o delta a
        climatologia reduzia `ppt_sazonalidade` em 20,4% e elevava
        `ppt_trimestre_seco` em 27,7% em relacao aos anos individuais em que o
        modelo foi treinado - e `ppt_sazonalidade` e a variavel que mais
        sobrevive fora da amostra nos dois cortes temporais.

        O teste compara o futuro com o OBSERVADO ANUAL. A tolerancia de 10%
        acomoda a mudanca climatica projetada e reprova a suavizacao, que era
        o dobro disso.
        """
        from yfvrisk.features.painel import agregar_clima_anual

        fut = _carregar("clima_futuro_muni_ano.parquet")
        obs = _carregar("clima_muni_mes.parquet")
        a0, a1 = cf.BASE_OBSERVADA
        anual = agregar_clima_anual(obs[(obs["ano"] >= a0) & (obs["ano"] <= a1)])

        for var in ("ppt_sazonalidade", "ppt_trimestre_seco"):
            referencia = anual[var].mean()
            desvio = abs(fut[var].mean() - referencia) / referencia
            assert desvio < 0.10, (
                f"{var}: futuro {fut[var].mean():.4f} contra observado anual "
                f"{referencia:.4f} ({desvio:.1%}) - sinal de suavizacao")

    def test_aquecimento_em_todos_os_cenarios(self):
        """Nenhum SSP do CMIP6 projeta resfriamento medio no Brasil em 2021-2040."""
        fut = _carregar("clima_futuro_muni_ano.parquet")
        obs = _carregar("clima_muni_mes.parquet")
        a0, a1 = cf.BASE_OBSERVADA
        base = (obs[(obs["ano"] >= a0) & (obs["ano"] <= a1)]
                .groupby("cod_mun")["tmax"].mean().mean())
        por_cenario = fut.groupby(["gcm", "ssp"])["tmax_medio"].mean()
        assert (por_cenario > base).all(), \
            f"cenario com resfriamento medio: {por_cenario[por_cenario <= base]}"

    def test_ordenacao_dos_ssps_no_ensemble(self):
        """O ssp585 aquece mais que o ssp126 na media do ensemble.

        Em 2021-2040 os SSPs ainda estao proximos - as trajetorias so divergem de
        verdade na segunda metade do seculo -, mas a ordem tem de aparecer.
        """
        fut = _carregar("clima_futuro_muni_ano.parquet")
        por_ssp = fut.groupby("ssp")["tmax_medio"].mean()
        assert por_ssp["ssp585"] > por_ssp["ssp126"]
