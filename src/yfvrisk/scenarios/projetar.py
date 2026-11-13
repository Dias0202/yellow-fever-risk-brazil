"""Projecao do risco sob o ensemble CMIP6, com o piso de ruido do proprio modelo.

Projeta o risco para 2021-2040 sob quatro SSPs e cinco GCMs e responde, antes
de reportar diferencas entre cenarios, a pergunta que decide se elas podem ser
reportadas:

    a diferenca entre os cenarios e maior que a incerteza do proprio modelo?

Sem essa comparacao, uma tabela de quatro SSPs com valores distintos parece
resultado mesmo quando os quatro valores estao dentro do erro. E o risco e
concreto aqui: o modelo de projecao climatica tem ROC-AUC de 0,5238 sob bloco
espaco-temporal (analysis/f3_02), que e o esquema correspondente ao uso que a
projecao faz. Um modelo nesse nivel produz
diferencas entre cenarios que podem ser inteiramente ruido de ajuste.

O piso de ruido
---------------
Reajusta-se a floresta sobre reamostragens do treino e mede-se a dispersao das
predicoes do MESMO cenario. A reamostragem e por MUNICIPIO, nao por linha: as 32
linhas de um municipio nao sao independentes, e reamostrar linhas produziria um
piso falsamente baixo e, portanto, um veredito falsamente favoravel ao cenario.

Duas dispersoes, entao, para cada municipio:

    entre cenarios   desvio das predicoes ao longo dos GCM x SSP, modelo fixo
    entre ajustes    desvio das predicoes ao longo das reamostragens, cenario fixo

A razao entre elas e o resultado. Abaixo de 1, a escolha do SSP importa menos que
a semente da reamostragem, e uma tabela de cenarios por SSP nao e reportavel.

O que se mantem fixo, e por que
-------------------------------
Este e um cenario CLIMATICO. Paisagem, demografia e imunidade ficam no ultimo
valor observado de cada municipio, e a diferenca entre a projecao e a linha de
base e portanto atribuivel so ao clima.

Isso e uma limitacao declarada, nao uma hipotese de que essas coisas nao mudam -
mudam, e o desmatamento em particular e o candidato mais provavel a dominar o
efeito climatico no horizonte de 2021-2040. Projeta-las exigiria um modelo de uso
do solo, que este trabalho nao tem. O que nao se pode fazer e manter tudo fixo
por acidente (ex.: de nomenclatura de coluna) e chamar o resultado de efeito do
clima.

Uma linha por municipio e ano-base
----------------------------------
O modelo foi treinado em anos individuais: cada linha e o clima de um ano. A
projecao respeita isso. O delta climatico e aplicado a cada ano observado do
periodo de referencia, e o cenario tem portanto 28 linhas por municipio - anos
reais deslocados pelo clima projetado, cada um ainda com a forma de um ano.

A alternativa, que era o desenho inicial deste modulo, seria prever sobre a
climatologia. Foi descartada por medicao: a media entre anos reduz
`ppt_sazonalidade` em 20,4% e eleva `ppt_trimestre_seco` em 27,7%, e
`ppt_sazonalidade` e a variavel que mais sobrevive fora da amostra nos dois
cortes temporais da analise de explicabilidade. Alem disso, a floresta e nao
linear, e a predicao da media nao e a media das predicoes. O que se agrega aqui
e a PREDICAO, nao a entrada.

A linha de base sao os proprios anos observados, sem delta, passados pelo mesmo
caminho. A diferenca entre os dois lados e portanto atribuivel so ao clima.

A imunidade merece nota propria. O estudo de evento em
`analysis/f5_00_diagnostico_vacinacao.py` mediu que a cobertura e consequencia
do surto e nao causa, e que no horizonte de projecao ja esta saturada - amplitude
interdecil de 0,116 em 2025. Nao ha cenario de vacinacao aqui, e nao por omissao:
o dado observacional nao identifica o efeito protetor, e uma alavanca de
vacinacao teria sinal arbitrario. Ver `ingest/vacinacao.py`.

Rodar:
    python -m yfvrisk.scenarios.projetar
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from yfvrisk import config as cfg
from yfvrisk.features.painel import GRUPOS

ALVO = "yf_event"

# Os mesmos grupos das analises de validacao e de explicabilidade. HISTORICO fica
# de fora por nao existir no horizonte de projecao: depende do desfecho que se
# quer projetar.
GRUPOS_USADOS = ["CLIMA", "CLIMA_DEFASADO", "PAISAGEM", "IMUNIDADE", "DEMOGRAFIA"]

# Reamostragens para o piso de ruido. Cada uma e um reajuste completo da floresta.
N_REAMOSTRAGENS = 30

MACRORREGIOES = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul",
                 5: "Centro-Oeste"}

SAIDA = cfg.METRICAS / "f5_01_projecao.json"
TABELA_MUNICIPIO = cfg.TABELAS / "f5_01_projecao_por_municipio.csv"
TABELA_CENARIO = cfg.TABELAS / "f5_01_projecao_por_cenario.csv"


def floresta_aleatoria(semente: int = 42):
    """Identica a das Fases 3 e 4, para que a projecao descreva aquele modelo."""
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=semente)),
    ])


def montar_variaveis(painel) -> list[str]:
    colunas = [c for g in GRUPOS_USADOS for c in GRUPOS[g] if c in painel.columns]
    faltando = [c for g in GRUPOS_USADOS for c in GRUPOS[g]
                if c not in painel.columns]
    if faltando:
        raise SystemExit(f"variaveis de GRUPOS ausentes do painel: {faltando}")
    return colunas


def _ultimo_observado(painel: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Ultimo valor nao ausente de cada variavel, por municipio.

    `groupby.last()` do pandas ja ignora ausentes, de modo que um municipio sem
    dado de floresta em 2025 herda o ultimo ano em que teve, em vez de virar
    ausente e cair na mediana do imputador.
    """
    return (painel.sort_values("ano").groupby("cod_mun")[colunas].last())


def montar_cenarios(painel: pd.DataFrame, colunas: list[str]):
    """Matrizes de projecao: uma linha de base e uma por combinacao GCM x SSP.

    Cada cenario tem uma linha por (municipio, ano-base): o delta climatico e
    aplicado a cada ano observado do periodo de referencia, e nao a uma
    climatologia. O motivo esta em `clima_futuro`: a media entre anos reduz
    `ppt_sazonalidade` em 20% e a floresta e nao linear, de modo que prever sobre
    um ano medio nao e o mesmo que a media das previsoes. O que se agrega e a
    PREDICAO, feita em `construir`, e nao a entrada.

    A linha de base sao os proprios anos observados, sem delta - tirados do
    painel, que ja os tem agregados e defasados. Os dois lados passam portanto
    pelo mesmo caminho, e a diferenca entre eles e atribuivel so ao clima.

    Paisagem, demografia e imunidade vem do ultimo valor observado de cada
    municipio, iguais nos dois lados.
    """
    from yfvrisk.scenarios.clima_futuro import BASE_OBSERVADA, SAIDA_ANUAL

    fixas = _ultimo_observado(painel, colunas)
    clima_cols = [c for c in GRUPOS["CLIMA"] if c in colunas]
    defasadas = [c for c in GRUPOS["CLIMA_DEFASADO"] if c in colunas]

    futuro = pd.read_parquet(SAIDA_ANUAL)

    # Defasagens do futuro: o ano-base anterior do MESMO cenario e municipio.
    # Os dois primeiros anos-base ficam sem defasagem e sao descartados, de modo
    # que nenhuma linha da projecao dependa de valor imputado nessas colunas.
    futuro = futuro.sort_values(["gcm", "ssp", "cod_mun", "ano_base"])
    g = futuro.groupby(["gcm", "ssp", "cod_mun"], sort=False)
    for c in defasadas:
        base_var, k = c.rsplit("_lag", 1)
        if base_var in futuro.columns:
            futuro[c] = g[base_var].shift(int(k))
    futuro = futuro.dropna(subset=defasadas)

    anos_uteis = sorted(futuro["ano_base"].unique())
    print(f"[cenarios] anos-base com defasagem completa: "
          f"{anos_uteis[0]}-{anos_uteis[-1]} ({len(anos_uteis)})")

    # Linha de base: os anos observados, direto do painel, nos mesmos anos-base e
    # nos mesmos municipios. A intersecao e necessaria porque `clima_futuro`
    # remove - declaradamente - os municipios sem celula terrestre no CMIP6, e um
    # lado com mais municipios que o outro tornaria a diferenca incomparavel.
    projetaveis = set(futuro["cod_mun"].unique())
    fora = sorted(set(painel["cod_mun"].unique()) - projetaveis)
    if fora:
        print(f"[cenarios] {len(fora)} municipios fora da projecao por falta de "
              f"clima futuro: {fora}")
    base = painel[painel["ano"].isin(anos_uteis)
                  & painel["cod_mun"].isin(projetaveis)][
        ["cod_mun", "ano"] + clima_cols + defasadas].copy()
    base = base.rename(columns={"ano": "ano_base"})

    def _montar(clima: pd.DataFrame) -> pd.DataFrame:
        x = clima[["cod_mun", "ano_base"]].copy()
        for c in colunas:
            x[c] = (clima[c].to_numpy() if c in clima_cols or c in defasadas
                    else fixas[c].reindex(clima["cod_mun"]).to_numpy())
        return x

    cenarios = {"linha_de_base": _montar(base)}
    for (gcm, ssp), grupo in futuro.groupby(["gcm", "ssp"]):
        cenarios[f"{gcm}|{ssp}"] = _montar(grupo)
    return cenarios, BASE_OBSERVADA, fora


def piso_de_ruido(X, y, grupos, cenarios: dict, colunas, n=N_REAMOSTRAGENS,
                  semente=42) -> dict:
    """Predicoes de VARIOS cenarios sob os mesmos reajustes do modelo.

    Reamostra municipios com reposicao, reajusta a floresta e prediz todos os
    cenarios pedidos com AQUELE ajuste. Devolve {nome: matriz (n, n_municipios)},
    todas na mesma ordem de reamostragem.

    Predizer varios cenarios por reajuste, e nao um so, e o que torna a
    comparacao justa. A versao inicial deste modulo media a dispersao das
    predicoes de um unico cenario entre reajustes e comparava esse numero com a
    dispersao entre cenarios - o que sobrestima o ruido, porque ao reajustar o
    modelo as predicoes de todos os cenarios se movem JUNTAS. O que degrada uma
    comparacao entre cenarios e so a parte do ruido que os separa, e essa parte
    so aparece se os cenarios forem preditos pelo mesmo ajuste e subtraidos.

    A reamostragem e por MUNICIPIO, nao por linha: as 28 linhas de um municipio
    nao sao independentes, e reamostrar linhas produziria um piso falsamente
    baixo e, portanto, um veredito falsamente favoravel ao cenario.
    """
    rng = np.random.default_rng(semente)
    codigos = np.unique(grupos)
    indices = {c: np.where(grupos == c)[0] for c in codigos}
    entradas = {nome: (m[colunas].to_numpy(float), m["cod_mun"].to_numpy())
                for nome, m in cenarios.items()}

    saida = {nome: [] for nome in cenarios}
    for i in range(n):
        escolhidos = rng.choice(codigos, size=len(codigos), replace=True)
        linhas = np.concatenate([indices[c] for c in escolhidos])
        modelo = floresta_aleatoria(semente + i)
        modelo.fit(X[linhas], y[linhas])
        for nome, (matriz, chave) in entradas.items():
            saida[nome].append(_media_por_municipio(
                modelo.predict_proba(matriz)[:, 1], chave).to_numpy())
        print(f"\r  reamostragem {i + 1}/{n}", end="")
    print()
    return {nome: np.vstack(v) for nome, v in saida.items()}


def _media_por_municipio(proba: np.ndarray, cod_mun: np.ndarray) -> pd.Series:
    """Media da predicao sobre os anos-base, por municipio.

    A agregacao e da PREDICAO e nao da entrada: e o ponto do desenho de uma
    linha por ano-base. Ordenada por codigo, para que todos os cenarios saiam
    na mesma ordem.
    """
    return pd.Series(proba).groupby(pd.Series(cod_mun)).mean().sort_index()


def construir() -> dict:
    cfg.garantir_diretorios()
    inicio = time.time()

    painel = pd.read_parquet(cfg.PROCESSADO / "painel_analitico.parquet")
    painel["regiao"] = (painel["cod_uf"].astype(int) // 10).map(MACRORREGIOES)
    colunas = montar_variaveis(painel)

    X = painel[colunas].to_numpy(dtype=float)
    y = painel[ALVO].to_numpy(dtype=int)
    grupos = painel["cod_mun"].to_numpy()

    print(f"[treino] {len(painel):,} linhas, {len(colunas)} variaveis, "
          f"{int(y.sum()):,} eventos")

    cenarios, base_observada, fora_da_projecao = montar_cenarios(painel, colunas)
    nomes = [k for k in cenarios if k != "linha_de_base"]
    print(f"[cenarios] linha de base + {len(nomes)} combinacoes GCM x SSP, "
          f"{len(cenarios['linha_de_base']):,} linhas cada")

    # Modelo de referencia: um so ajuste, todos os cenarios.
    modelo = floresta_aleatoria(42)
    modelo.fit(X, y)

    predicao = {}
    for i, (nome, m) in enumerate(cenarios.items(), 1):
        entrada = m[colunas].to_numpy(dtype=float)
        predicao[nome] = _media_por_municipio(
            modelo.predict_proba(entrada)[:, 1], m["cod_mun"].to_numpy())
        print(f"\r  cenario {i}/{len(cenarios)}", end="")
    print()

    ordem_mun = predicao["linha_de_base"].index.to_numpy()
    base = predicao["linha_de_base"].to_numpy()
    matriz = np.vstack([predicao[n].to_numpy() for n in nomes])

    # Dispersao entre cenarios, com o modelo fixo.
    dp_entre_cenarios = matriz.std(axis=0, ddof=1)

    # Piso de ruido: dispersao entre ajustes, com o cenario fixo. Usa o cenario
    # central do ensemble para nao depender de qual GCM se escolheu.
    # Piso de ruido, pareado. Um subconjunto de cenarios e predito sob CADA
    # reajuste, para que os contrastes possam ser formados dentro do ajuste.
    #
    # O GCM e fixado no de sensibilidade mais proxima da melhor estimativa do
    # AR6, porque a dispersao entre GCMs ja foi medida com o modelo fixo e nao e
    # o que este passo investiga: aqui a pergunta e se o SSP se distingue depois
    # de descontada a incerteza do AJUSTE.
    gcm_central = "MPI-ESM1-2-HR"
    alvos = {"linha_de_base": cenarios["linha_de_base"]}
    for ssp in ("ssp126", "ssp585"):
        chave = f"{gcm_central}|{ssp}"
        if chave in cenarios:
            alvos[chave] = cenarios[chave]
    print(f"[piso de ruido] {len(alvos)} cenarios x {N_REAMOSTRAGENS} reajustes")
    reamostras = piso_de_ruido(X, y, grupos, alvos, colunas)

    dp_entre_ajustes = reamostras[f"{gcm_central}|ssp585"].std(axis=0, ddof=1)

    # Razao nao pareada: a dispersao entre os 20 cenarios contra a dispersao das
    # predicoes de um cenario entre ajustes. Responde "o nivel de risco previsto
    # para um municipio e mais incerto pelo cenario ou pelo ajuste?".
    with np.errstate(invalid="ignore", divide="ignore"):
        razao = np.where(dp_entre_ajustes > 0,
                         dp_entre_cenarios / dp_entre_ajustes, np.nan)

    # Contrastes PAREADOS, formados dentro de cada ajuste. Sao os que respondem
    # as duas perguntas que a projecao de fato faz.
    def _contraste(a, b):
        """Media nacional de (a - b) por reajuste, e o intervalo entre eles."""
        v = (reamostras[a] - reamostras[b]).mean(axis=1)
        return {"media": float(v.mean()), "dp_entre_ajustes": float(v.std(ddof=1)),
                "ic95": [float(np.percentile(v, 2.5)),
                         float(np.percentile(v, 97.5))],
                "reajustes_com_sinal_positivo": int((v > 0).sum()),
                "n_reajustes": int(len(v))}

    pareados = {
        # O clima futuro eleva o risco em relacao ao observado?
        "futuro_menos_base": _contraste(f"{gcm_central}|ssp585", "linha_de_base"),
        # O SSP mais severo se distingue do mais brando?
        "ssp585_menos_ssp126": _contraste(f"{gcm_central}|ssp585",
                                          f"{gcm_central}|ssp126"),
    }
    print("\ncontrastes pareados (GCM fixo em " + gcm_central + "):")
    for nome, c in pareados.items():
        print(f"  {nome:22s} {c['media']:+.5f}  IC 95% "
              f"[{c['ic95'][0]:+.5f}, {c['ic95'][1]:+.5f}]  "
              f"{c['reajustes_com_sinal_positivo']}/{c['n_reajustes']} positivos")

    por_municipio = pd.DataFrame({
        "cod_mun": ordem_mun,
        "risco_linha_de_base": base,
        "risco_medio_futuro": matriz.mean(axis=0),
        "delta_medio": matriz.mean(axis=0) - base,
        "dp_entre_cenarios": dp_entre_cenarios,
        "dp_entre_ajustes": dp_entre_ajustes,
        "razao_sinal_ruido": razao,
    })
    ref = (painel[["cod_mun", "uf", "regiao", "nome_mun"]]
           .drop_duplicates("cod_mun"))
    por_municipio = por_municipio.merge(ref, on="cod_mun", how="left")

    por_cenario = pd.DataFrame([
        {"cenario": n, "gcm": n.split("|")[0], "ssp": n.split("|")[1],
         "risco_medio": float(predicao[n].mean()),
         "delta_medio": float((predicao[n].to_numpy() - base).mean()),
         "delta_mediano": float(np.median(predicao[n].to_numpy() - base))}
        for n in nomes]).sort_values(["ssp", "gcm"])

    por_regiao = []
    for reg, g in por_municipio.groupby("regiao"):
        por_regiao.append({
            "regiao": str(reg), "municipios": int(len(g)),
            "delta_medio": float(g["delta_medio"].mean()),
            "dp_entre_cenarios_media": float(g["dp_entre_cenarios"].mean()),
            "dp_entre_ajustes_media": float(g["dp_entre_ajustes"].mean()),
            "razao_sinal_ruido_mediana": float(g["razao_sinal_ruido"].median()),
        })

    # Separacao entre SSPs, agregando os GCMs.
    por_ssp = (por_cenario.groupby("ssp")["risco_medio"]
               .agg(["mean", "std", "min", "max"]).reset_index()
               .rename(columns={"mean": "risco_medio", "std": "dp_entre_gcms"}))

    resultado = {
        "n_treino": int(len(painel)), "n_variaveis": len(colunas),
        "base_observada": list(base_observada),
        "municipios_projetados": int(len(ordem_mun)),
        "municipios_fora_da_projecao": fora_da_projecao,
        "linhas_por_cenario": int(len(cenarios["linha_de_base"])),
        "variaveis": colunas, "eventos": int(y.sum()),
        "n_cenarios": len(nomes), "n_reamostragens": N_REAMOSTRAGENS,
        "risco_linha_de_base_medio": float(base.mean()),
        "delta_medio_nacional": float((matriz.mean(axis=0) - base).mean()),
        "dp_entre_cenarios_media": float(dp_entre_cenarios.mean()),
        "dp_entre_ajustes_media": float(dp_entre_ajustes.mean()),
        "razao_sinal_ruido_mediana": float(np.nanmedian(razao)),
        "municipios_com_sinal_acima_do_ruido": int(np.nansum(razao > 1)),
        "gcm_central_do_piso": gcm_central,
        "contrastes_pareados": pareados,
        "por_cenario": por_cenario.to_dict("records"),
        "por_ssp": por_ssp.to_dict("records"),
        "por_regiao": por_regiao,
        "segundos_total": round(time.time() - inicio, 1),
    }

    por_municipio.to_csv(TABELA_MUNICIPIO, index=False, encoding="utf-8")
    por_cenario.to_csv(TABELA_CENARIO, index=False, encoding="utf-8")
    SAIDA.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    return resultado


if __name__ == "__main__":
    r = construir()
    print(f"\nrisco medio na linha de base ....... {r['risco_linha_de_base_medio']:.5f}")
    print(f"delta medio nacional ............... {r['delta_medio_nacional']:+.5f}")
    print(f"dp entre cenarios (media) .......... {r['dp_entre_cenarios_media']:.5f}")
    print(f"dp entre ajustes  (media) .......... {r['dp_entre_ajustes_media']:.5f}")
    print(f"razao sinal/ruido (mediana) ........ {r['razao_sinal_ruido_mediana']:.3f}")
    print(f"municipios com sinal acima do ruido  "
          f"{r['municipios_com_sinal_acima_do_ruido']:,} de {cfg.N_MUNICIPIOS:,}")
    print("\ncontrastes pareados, dentro de cada reajuste:")
    for nome, c in r["contrastes_pareados"].items():
        print(f"  {nome:22s} {c['media']:+.5f}  IC 95% "
              f"[{c['ic95'][0]:+.5f}, {c['ic95'][1]:+.5f}]  "
              f"{c['reajustes_com_sinal_positivo']}/{c['n_reajustes']} positivos")
    print("\npor SSP:")
    for s in r["por_ssp"]:
        print(f"  {s['ssp']}  risco medio {s['risco_medio']:.5f}  "
              f"dp entre GCMs {s['dp_entre_gcms']:.5f}")
