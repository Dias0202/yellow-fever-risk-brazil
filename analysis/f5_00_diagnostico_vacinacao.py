"""Diagnostico da vacinacao - Por que a imunidade nao carrega sinal generalizavel.

A analise de explicabilidade (f4_01) mede que `imunidade_acumulada` e
`suscetivel_pct` sao #1 e #2 na importancia de Gini e #29 e #31 na permutacao
fora da amostra, ambas com importancia media NEGATIVA, e que a familia IMUNIDADE
inteira contribui -1,2%: embaralha-la melhora a PR-AUC.

Um cenario de vacinacao construido sobre uma variavel sem sinal generalizavel
produz numeros arbitrarios - inclusive o absurdo de aumentar a cobertura ELEVAR o
risco previsto. Antes de projetar qualquer coisa e preciso saber por que o sinal
nao existe.

O ponto de partida ja trata duas causas conhecidas: a ausencia de registro nao
vira 0% de cobertura, e a variavel nao e a cobertura pontual de um ano, e sim a
imunidade acumulada com renovacao populacional (ver `yfvrisk/ingest/vacinacao.py`).
Isso nao basta. Este script testa seis hipoteses para o que resta. As quatro primeiras
foram escritas antes de rodar; a H5 e a H6 sao consequencia do que as anteriores
apuraram, e a ordem em que aparecem e a ordem em que foram formuladas.

  H1  A variavel e um relogio. Se a imunidade acumulada e majoritariamente uma
      funcao do ano, ela e um indice temporal disfarcado. Sob particao temporal o
      teste cai inteiramente fora da faixa vista no treino, e uma arvore nao
      extrapola: todo ponto acima do maximo do treino recebe a predicao da folha
      daquele maximo.

  H2  A acumulacao comeca quando o REGISTRO comeca, nao quando a vacinacao
      comeca. A disponibilidade do PNI vai de 5,3% das linhas em 1994 a 99,9% em
      2022. Municipio que so aparece na fonte em 2005 chega a 2014 com menos
      imunidade acumulada que um vizinho identico que aparece desde 1994.

  H3  O sinal inverte entre regimes. Ate 2016 a vacina era recomendada apenas na
      area com recomendacao de vacinacao (ACRV), que e onde a febre amarela
      circulava. Alta imunidade e alto risco andavam juntos por confundimento por
      indicacao. A onda de 2016-2019 atingiu a Mata Atlantica, fora da ACRV, com
      imunidade baixa. Se o sinal dessa associacao inverte entre os dois regimes,
      uma importancia media proxima de zero e o resultado esperado de um efeito
      nao estacionario, e nao a ausencia de efeito.

  H4  A variavel nao varia no horizonte de projecao. Se em 2020-2025 a imunidade
      esta saturada e quase constante entre municipios, nao ha contraste do qual
      extrair discriminacao, nem contraste que um cenario possa mover.

  H5  Sobra algum sinal DENTRO do municipio? E a hipotese decisiva para os
      cenarios, porque e a unica que corresponde ao que um cenario faz. Um cenario nao troca
      um municipio por outro: ele move a cobertura do proprio municipio. Toda
      associacao que venha de comparar municipios diferentes - o confundimento por
      indicacao da H3, a ACRV, a vigilancia da H2 - e irrelevante para essa
      pergunta e so atrapalha. Centrando a imunidade pela media do municipio E
      pela media do ano, o que resta e o desvio da trajetoria propria do municipio
      em relacao a sua propria tendencia. Se a associacao desse residuo com o
      evento for nula, nenhuma construcao alternativa da variavel produz alavanca: os dados
      observacionais nao identificam o efeito, e o cenario tem de ser
      reformulado em vez de recalculado.

  H6  Causalidade reversa. A H5 nao deu associacao nula: deu associacao POSITIVA
      dentro do municipio, isto e, os anos em que o municipio esteve mais vacinado
      que o proprio normal foram os anos COM surto. Vacina nao causa febre
      amarela, entao ou a seta do tempo esta invertida ou a variavel mede outra
      coisa. Um estudo de evento sobre a cobertura anual separa as duas leituras.

Nada aqui altera o painel. E medicao sobre o painel existente, e o que ela apura
define a construcao da variavel de modelagem (`imunidade_defasada2`).

Rodar:
    PYTHONPATH=src python \\
        analysis/f5_00_diagnostico_vacinacao.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import spearmanr                            # noqa: E402

from yfvrisk import config as cfg                            # noqa: E402

warnings.filterwarnings("ignore")

ALVO = "yf_event"
SAIDA = cfg.METRICAS / "f5_00_diagnostico_vacinacao.json"
LOG = cfg.METRICAS / "f5_00_diagnostico_vacinacao.log"

# Os mesmos cortes de f4_01, para que o diagnostico fale da mesma particao que
# produziu a importancia negativa que ele investiga.
CORTES = [2014, 2019]

# Primeiro ano da onda da Mata Atlantica. E o divisor de regime da H3.
ANO_INICIO_ONDA = 2016

_linhas_log: list[str] = []


def diga(texto: str = "") -> None:
    print(texto)
    _linhas_log.append(texto)


def carregar() -> pd.DataFrame:
    p = pd.read_parquet(cfg.PROCESSADO / "painel_analitico.parquet")
    v = pd.read_parquet(cfg.PROCESSADO / "vacinacao_muni_ano.parquet")
    p = p.merge(v[["cod_mun", "ano", "cobertura_pct"]], on=["cod_mun", "ano"],
                how="left")
    return p.sort_values(["cod_mun", "ano"]).reset_index(drop=True)


def _parcial_posto(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """Correlacao parcial de Spearman de x com y, controlando z."""
    rxy = spearmanr(x, y).statistic
    rxz = spearmanr(x, z).statistic
    ryz = spearmanr(y, z).statistic
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------
# H1 - a variavel e um relogio
# --------------------------------------------------------------------------
def h1_relogio(p: pd.DataFrame) -> dict:
    """Decompoe a variancia da imunidade em entre-anos e dentro-do-ano.

    Se a fracao entre-anos domina, a variavel carrega sobretudo a passagem do
    tempo, e nao o contraste entre municipios que um cenario precisa mover.
    """
    diga("")
    diga("=" * 74)
    diga("H1 - a imunidade acumulada e um relogio?")
    diga("=" * 74)

    x = p["imunidade_acumulada"]
    total = float(x.var())
    media_ano = p.groupby("ano")["imunidade_acumulada"].transform("mean")
    entre = float(media_ano.var())
    dentro = float((x - media_ano).var())

    rho_ano, p_ano = spearmanr(p["ano"], x)

    # O ponto operacional: sob particao temporal, quanto do teste cai fora da
    # faixa de valores vista no treino.
    deslocamento = {}
    for corte in CORTES:
        tr = p.loc[p["ano"] <= corte, "imunidade_acumulada"]
        te = p.loc[p["ano"] > corte, "imunidade_acumulada"]
        deslocamento[corte] = {
            "treino_max": float(tr.max()),
            "teste_mediana": float(te.median()),
            "fracao_teste_acima_do_max_do_treino": float((te > tr.max()).mean()),
            # Percentil do treino em que a MEDIANA do teste cai: mede o
            # deslocamento de distribuicao mesmo sem extrapolacao estrita.
            "percentil_do_treino_na_mediana_do_teste": float(
                (tr < te.median()).mean()),
        }

    diga(f"variancia total ................... {total:.5f}")
    diga(f"  entre anos (o relogio) .......... {entre:.5f}  ({entre / total:.1%})")
    diga(f"  dentro do ano (o contraste) ..... {dentro:.5f}  ({dentro / total:.1%})")
    diga(f"correlacao de posto com o ano ..... rho = {rho_ano:+.3f} (p = {p_ano:.2g})")
    diga("")
    diga("deslocamento de distribuicao sob particao temporal:")
    diga("corte  max treino  mediana teste  percentil do treino na mediana do teste")
    for corte, d in deslocamento.items():
        diga(f"{corte}     {d['treino_max']:.4f}      {d['teste_mediana']:.4f}"
             f"              {d['percentil_do_treino_na_mediana_do_teste']:.1%}")

    return {
        "variancia_total": total,
        "variancia_entre_anos": entre,
        "variancia_dentro_do_ano": dentro,
        "fracao_entre_anos": entre / total,
        "rho_com_ano": float(rho_ano),
        "p_rho_com_ano": float(p_ano),
        "deslocamento_por_corte": deslocamento,
    }


# --------------------------------------------------------------------------
# H2 - a acumulacao mede o inicio do registro
# --------------------------------------------------------------------------
def h2_registro(p: pd.DataFrame) -> dict:
    """A imunidade acumulada depende de quando o municipio entrou na fonte?

    A correlacao parcial responde a pergunta que importa: descontada a cobertura
    media efetivamente observada, quanto do valor acumulado ainda e explicado
    pela data de entrada do municipio no registro. Se sobra muito, a variavel
    esta medindo cobertura de vigilancia, nao cobertura vacinal.
    """
    diga("")
    diga("=" * 74)
    diga("H2 - a acumulacao comeca no registro, nao na vacinacao?")
    diga("=" * 74)

    com_dado = p.loc[p["cobertura_pct"].notna()]
    primeiro = (com_dado.groupby("cod_mun")["ano"].min()
                .rename("ano_primeiro_registro"))
    media_cob = (com_dado.groupby("cod_mun")["cobertura_pct"].mean()
                 .rename("cobertura_media"))
    n_anos = com_dado.groupby("cod_mun").size().rename("anos_com_registro")

    resultados = {}
    for ano_ref in CORTES:
        imun = p.loc[p["ano"] == ano_ref].set_index("cod_mun")["imunidade_acumulada"]
        m = pd.concat([primeiro, media_cob, n_anos, imun], axis=1).dropna()

        rho_prim, p_prim = spearmanr(m["ano_primeiro_registro"],
                                     m["imunidade_acumulada"])
        rho_cob, p_cob = spearmanr(m["cobertura_media"], m["imunidade_acumulada"])
        rho_parcial = _parcial_posto(m["ano_primeiro_registro"],
                                     m["imunidade_acumulada"],
                                     m["cobertura_media"])

        resultados[ano_ref] = {
            "municipios": int(len(m)),
            "rho_ano_primeiro_registro": float(rho_prim),
            "p_ano_primeiro_registro": float(p_prim),
            "rho_cobertura_media": float(rho_cob),
            "p_cobertura_media": float(p_cob),
            "rho_parcial_registro_dado_cobertura": rho_parcial,
            "distribuicao_ano_primeiro_registro": {
                str(k): int(v) for k, v in
                m["ano_primeiro_registro"].value_counts().sort_index().items()},
        }
        diga("")
        diga(f"imunidade acumulada em {ano_ref} ({len(m):,} municipios)")
        diga(f"  vs ano do primeiro registro ..... rho = {rho_prim:+.3f} "
             f"(p = {p_prim:.2g})")
        diga(f"  vs cobertura media observada .... rho = {rho_cob:+.3f} "
             f"(p = {p_cob:.2g})")
        diga(f"  parcial: registro | cobertura ... rho = {rho_parcial:+.3f}")

    return resultados


# --------------------------------------------------------------------------
# H3 - o sinal inverte entre regimes
# --------------------------------------------------------------------------
def h3_inversao(p: pd.DataFrame) -> dict:
    """Associacao entre imunidade e evento, medida DENTRO de cada ano.

    Medir dentro do ano remove o relogio da H1: o que sobra e o contraste entre
    municipios, que e o que um cenario de vacinacao pretende mover.
    """
    diga("")
    diga("=" * 74)
    diga("H3 - o sinal da associacao inverte entre regimes?")
    diga("=" * 74)

    por_ano = []
    for ano, g in p.groupby("ano"):
        if g[ALVO].sum() < 5:
            continue
        rho, pv = spearmanr(g["imunidade_acumulada"], g[ALVO])
        com = float(g.loc[g[ALVO] == 1, "imunidade_acumulada"].mean())
        sem = float(g.loc[g[ALVO] == 0, "imunidade_acumulada"].mean())
        por_ano.append({"ano": int(ano), "eventos": int(g[ALVO].sum()),
                        "rho": float(rho), "p": float(pv),
                        "imunidade_media_com_evento": com,
                        "imunidade_media_sem_evento": sem,
                        "diferenca": com - sem})

    tab = pd.DataFrame(por_ano)
    diga("")
    diga("ano  eventos     rho        p    imun|evento  imun|sem  diferenca")
    for r in por_ano:
        diga(f"{r['ano']}   {r['eventos']:5d}  {r['rho']:+.3f}  {r['p']:8.2g}"
             f"      {r['imunidade_media_com_evento']:.3f}     "
             f"{r['imunidade_media_sem_evento']:.3f}     {r['diferenca']:+.3f}")

    regimes = {"ate_2015": tab["ano"] < ANO_INICIO_ONDA,
               "de_2016_em_diante": tab["ano"] >= ANO_INICIO_ONDA}
    resumo = {}
    for nome, mascara in regimes.items():
        sub = tab.loc[mascara]
        # Media ponderada pelo numero de eventos: anos com 5 eventos nao pesam o
        # mesmo que anos com 700.
        peso = sub["eventos"]
        resumo[nome] = {
            "anos": int(len(sub)),
            "eventos": int(peso.sum()),
            "rho_medio_ponderado": float(np.average(sub["rho"], weights=peso)),
            "diferenca_media_ponderada": float(np.average(sub["diferenca"],
                                                         weights=peso)),
            "anos_com_rho_positivo": int((sub["rho"] > 0).sum()),
            "anos_com_rho_negativo": int((sub["rho"] < 0).sum()),
        }
        diga("")
        diga(f"{nome}: {len(sub)} anos, {int(peso.sum())} eventos")
        diga(f"  rho medio ponderado ......... "
             f"{resumo[nome]['rho_medio_ponderado']:+.3f}")
        diga(f"  diferenca media ponderada ... "
             f"{resumo[nome]['diferenca_media_ponderada']:+.3f}")
        diga(f"  anos com rho > 0 ............ "
             f"{resumo[nome]['anos_com_rho_positivo']} de {len(sub)}")

    return {"por_ano": por_ano, "por_regime": resumo}


# --------------------------------------------------------------------------
# H4 - a variavel nao varia no horizonte de projecao
# --------------------------------------------------------------------------
def h4_saturacao(p: pd.DataFrame) -> dict:
    """Dispersao entre municipios, ano a ano.

    Um cenario move a variavel. Se em 2025 ela ja esta saturada e sem dispersao,
    nao ha contraste a mover, nem a partir do qual discriminar.
    """
    diga("")
    diga("=" * 74)
    diga("H4 - ainda ha contraste entre municipios no horizonte de projecao?")
    diga("=" * 74)

    linhas = []
    for ano, g in p.groupby("ano"):
        x = g["imunidade_acumulada"]
        linhas.append({"ano": int(ano), "media": float(x.mean()),
                       "dp": float(x.std()),
                       "p10": float(x.quantile(0.10)),
                       "p90": float(x.quantile(0.90)),
                       "amplitude_interdecil": float(x.quantile(0.90)
                                                     - x.quantile(0.10)),
                       "fracao_acima_de_090": float((x > 0.90).mean())})

    diga("")
    diga("ano   media     dp     p10     p90   amplitude  % acima de 0,90")
    for r in linhas:
        if r["ano"] % 3 == 0 or r["ano"] >= 2019:
            diga(f"{r['ano']}  {r['media']:.3f}  {r['dp']:.3f}  {r['p10']:.3f}"
                 f"  {r['p90']:.3f}     {r['amplitude_interdecil']:.3f}"
                 f"        {r['fracao_acima_de_090']:.1%}")

    return {"por_ano": linhas}


# --------------------------------------------------------------------------
# H5 - sobra algum sinal dentro do municipio
# --------------------------------------------------------------------------
def h5_dentro_do_municipio(p: pd.DataFrame) -> dict:
    """Associacao do residuo de dupla centragem com o evento.

    Tres medicoes encaixadas, cada uma removendo uma fonte de confundimento:

      bruta            todos contra todos. E a associacao que as arvores veem.
      dentro do ano    remove o relogio da H1. E a H3.
      dupla centragem  remove tambem tudo que e fixo no municipio - a ACRV, o
                       bioma, a altitude, o esforco de vigilancia, a data de
                       entrada no registro da H2. O que sobra e: quando ESTE
                       municipio esteve mais vacinado que o seu proprio normal,
                       teve menos eventos?

    A terceira e a unica que responde a pergunta do cenario. As duas primeiras
    entram para que a diferenca entre elas seja legivel: e a diferenca que mede
    o quanto do "efeito" aparente e confundimento entre municipios.

    Restringe-se aos municipios com pelo menos um evento e com variacao propria
    na imunidade. Municipio sem evento nenhum nao contribui informacao para uma
    comparacao dentro dele mesmo, e municipio com imunidade constante tem residuo
    identicamente zero - incluir os dois so dilui a estimativa com linhas que nao
    carregam contraste.
    """
    diga("")
    diga("=" * 74)
    diga("H5 - sobra algum sinal DENTRO do municipio?")
    diga("=" * 74)

    d = p.copy()
    d["imun_menos_ano"] = (d["imunidade_acumulada"]
                           - d.groupby("ano")["imunidade_acumulada"]
                           .transform("mean"))
    d["residuo"] = (d["imun_menos_ano"]
                    - d.groupby("cod_mun")["imun_menos_ano"].transform("mean"))

    rho_bruta, p_bruta = spearmanr(d["imunidade_acumulada"], d[ALVO])
    rho_ano, p_ano = spearmanr(d["imun_menos_ano"], d[ALVO])

    # Municipios que informam a comparacao dentro deles mesmos.
    com_evento = d.groupby("cod_mun")[ALVO].transform("sum") > 0
    com_variacao = d.groupby("cod_mun")["imunidade_acumulada"].transform("std") > 1e-9
    sub = d.loc[com_evento & com_variacao]

    rho_res, p_res = spearmanr(sub["residuo"], sub[ALVO])
    com = float(sub.loc[sub[ALVO] == 1, "residuo"].mean())
    sem = float(sub.loc[sub[ALVO] == 0, "residuo"].mean())

    # Intervalo por reamostragem agrupada: reamostra MUNICIPIOS, nao linhas. As
    # 32 linhas de um municipio nao sao independentes entre si, e reamostrar
    # linhas produziria um intervalo falsamente estreito.
    rng = np.random.default_rng(42)
    codigos = sub["cod_mun"].unique()
    por_cod = {c: g for c, g in sub.groupby("cod_mun")}
    amostras = []
    for _ in range(500):
        escolhidos = rng.choice(codigos, size=len(codigos), replace=True)
        b = pd.concat([por_cod[c] for c in escolhidos], ignore_index=True)
        if b[ALVO].nunique() > 1:
            amostras.append(spearmanr(b["residuo"], b[ALVO]).statistic)
    ic = [float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))]

    diga("")
    diga(f"bruta (todos contra todos) ......... rho = {rho_bruta:+.4f} "
         f"(p = {p_bruta:.2g})")
    diga(f"centrada no ano (remove o relogio) . rho = {rho_ano:+.4f} "
         f"(p = {p_ano:.2g})")
    diga(f"dupla centragem (dentro do muni) ... rho = {rho_res:+.4f} "
         f"(p = {p_res:.2g})")
    diga(f"  IC 95% por reamostragem de municipio: [{ic[0]:+.4f}, {ic[1]:+.4f}]")
    diga(f"  {len(codigos):,} municipios, {len(sub):,} linhas, "
         f"{int(sub[ALVO].sum()):,} eventos")
    diga(f"  residuo medio em ano com evento .... {com:+.4f}")
    diga(f"  residuo medio em ano sem evento .... {sem:+.4f}")
    diga(f"  diferenca .......................... {com - sem:+.4f}")

    return {
        "rho_bruta": float(rho_bruta), "p_bruta": float(p_bruta),
        "rho_centrada_no_ano": float(rho_ano), "p_centrada_no_ano": float(p_ano),
        "rho_dupla_centragem": float(rho_res),
        "p_dupla_centragem": float(p_res),
        "ic95_dupla_centragem": ic,
        "municipios_informativos": int(len(codigos)),
        "linhas_informativas": int(len(sub)),
        "eventos_informativos": int(sub[ALVO].sum()),
        "residuo_medio_com_evento": com,
        "residuo_medio_sem_evento": sem,
        "diferenca": com - sem,
    }


# --------------------------------------------------------------------------
# H6 - causalidade reversa
# --------------------------------------------------------------------------
def h6_causalidade_reversa(p: pd.DataFrame) -> dict:
    """Estudo de evento: a cobertura sobe ANTES ou DEPOIS do surto?

    A H5 encontrou que, dentro do municipio, os anos de imunidade acima da
    propria tendencia sao os anos COM evento, e nao os anos sem. Vacina nao causa
    febre amarela, entao ou a variavel esta medindo outra coisa ou a seta do tempo
    esta invertida. A explicacao que o proprio sistema de vigilancia sugere e a
    segunda: um surto dispara bloqueio vacinal e campanha de massa no mesmo ano, e
    o PNI registra essas doses no ano do surto.

    Se for isso, a imunidade do painel e uma variavel a JUSANTE do desfecho, e
    nenhum reajuste de escala, imputacao ou formula de acumulo conserta - o modelo
    esta lendo a resposta ao surto como se fosse a condicao anterior a ele.

    O teste e um estudo de evento sobre a COBERTURA anual, nao sobre o acumulado:
    o acumulado, por ser cumulativo, borra qualquer pico sobre todos os anos
    seguintes e nao distingue antes de depois. A cobertura e duplamente centrada
    (pelo ano e pelo municipio) para que o perfil descreva o desvio em relacao ao
    normal do proprio municipio naquele ano do pais.

    O contrafactual esta no mesmo grafico: se a vacinacao PREVINE, a cobertura
    elevada aparece em k negativo e o risco cai depois; se ela RESPONDE, o pico
    esta em k = 0 ou k positivo.
    """
    diga("")
    diga("=" * 74)
    diga("H6 - causalidade reversa: a campanha vem depois do surto?")
    diga("=" * 74)

    d = p.copy()
    cob_ano = d.groupby("ano")["cobertura_pct"].transform("mean")
    d["cob_menos_ano"] = d["cobertura_pct"] - cob_ano
    d["cob_residuo"] = (d["cob_menos_ano"]
                        - d.groupby("cod_mun")["cob_menos_ano"].transform("mean"))

    # Indice (municipio, ano) -> residuo, para deslocar no tempo sem reordenar.
    residuo = d.set_index(["cod_mun", "ano"])["cob_residuo"]
    eventos = d.loc[d[ALVO] == 1, ["cod_mun", "ano"]]
    # Municipio com varios surtos tem janelas sobrepostas: o k = -1 de um surto
    # pode ser o k = 0 do surto anterior, o que contaminaria justamente o lado
    # que precisa estar limpo para a leitura causal. O perfil restrito ao
    # PRIMEIRO surto de cada municipio nao tem essa sobreposicao a esquerda.
    primeiros = (eventos.groupby("cod_mun")["ano"].min().reset_index())

    def _perfil(alvos: pd.DataFrame) -> list[dict]:
        linhas = []
        for k in range(-3, 4):
            chaves = pd.MultiIndex.from_arrays([alvos["cod_mun"], alvos["ano"] + k])
            valores = residuo.reindex(chaves)
            linhas.append({"k": k, "n": int(valores.notna().sum()),
                           "residuo_medio": float(valores.mean()),
                           "residuo_mediano": float(valores.median())})
        return linhas

    perfil = _perfil(eventos)
    perfil_primeiro = _perfil(primeiros)

    diga("")
    diga("cobertura anual duplamente centrada, em torno do ano do evento (k = 0)")
    diga("               todos os surtos          so o primeiro de cada municipio")
    diga("  k       n   medio   mediana            n   medio   mediana")
    for r, rp in zip(perfil, perfil_primeiro):
        marca = "  <-- ano do evento" if r["k"] == 0 else ""
        diga(f"{r['k']:+d}   {r['n']:5d}  {r['residuo_medio']:+7.2f}  "
             f"{r['residuo_mediano']:+7.2f}        {rp['n']:5d}  "
             f"{rp['residuo_medio']:+7.2f}  {rp['residuo_mediano']:+7.2f}{marca}")

    # A consequencia pratica: a mesma associacao da H5, com a imunidade defasada.
    # Defasar remove o registro da campanha disparada pelo proprio surto, e e a
    # unica versao que corresponde ao uso - so se pode agir sobre a imunidade que
    # ja existe quando o ano comeca.
    d = d.sort_values(["cod_mun", "ano"])
    com_evento = d.groupby("cod_mun")[ALVO].transform("sum") > 0
    com_variacao = d.groupby("cod_mun")["imunidade_acumulada"].transform("std") > 1e-9

    defasadas = {}
    for k in (0, 1, 2):
        col = d.groupby("cod_mun")["imunidade_acumulada"].shift(k)
        centrada = col - col.groupby(d["ano"]).transform("mean")
        res = centrada - centrada.groupby(d["cod_mun"]).transform("mean")
        sel = com_evento & com_variacao & res.notna()
        rho, pv = spearmanr(res[sel], d.loc[sel, ALVO])
        defasadas[k] = {"rho": float(rho), "p": float(pv),
                        "linhas": int(sel.sum())}
        diga(f"imunidade defasada em {k} ano(s), dupla centragem: "
             f"rho = {rho:+.4f} (p = {pv:.2g}, n = {int(sel.sum()):,})")

    return {"estudo_de_evento_cobertura": perfil,
            "estudo_de_evento_primeiro_surto": perfil_primeiro,
            "associacao_por_defasagem": defasadas}


def main() -> None:
    cfg.garantir_diretorios()
    p = carregar()
    diga(f"painel: {len(p):,} linhas, {p['cod_mun'].nunique():,} municipios, "
         f"{p['ano'].min()}-{p['ano'].max()}, {int(p[ALVO].sum()):,} eventos")

    resultado = {
        "h1_relogio": h1_relogio(p),
        "h2_registro": h2_registro(p),
        "h3_inversao": h3_inversao(p),
        "h4_saturacao": h4_saturacao(p),
        "h5_dentro_do_municipio": h5_dentro_do_municipio(p),
        "h6_causalidade_reversa": h6_causalidade_reversa(p),
    }

    SAIDA.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    LOG.write_text("\n".join(_linhas_log), encoding="utf-8")
    print(f"\nmetricas em {SAIDA}")
    print(f"log em {LOG}")


if __name__ == "__main__":
    main()
