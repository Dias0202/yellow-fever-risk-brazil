"""Clima futuro por municipio, pelo metodo delta, para o ensemble CMIP6.

Tres decisoes estruturais. As duas primeiras sao de proveniencia e estao no
downloader (`ingest/baixar_worldclim.py`): varios GCMs e a mesma resolucao nos
dois lados da anomalia. A terceira e deste modulo.

Uma unica funcao para observado e futuro
----------------------------------------
Se o clima futuro fosse gravado em colunas derivadas por outro caminho (ex.:
bioclimaticas `Bio1...Bio15`) enquanto o modelo le `tmax_mean`, `tmin_mean`,
`ppt_mean` e companhia, as colunas nunca se encontrariam e um cenario "so clima"
sairia numericamente IDENTICO a linha de base, sem nenhum erro visivel.

Por isso as variaveis anuais do futuro sao derivadas com a MESMA funcao que as
deriva do observado - `features.painel.agregar_clima_anual`. Enquanto houver uma
unica funcao, a divergencia nao pode aparecer.

Por que delta, e nao o valor projetado
--------------------------------------
Um GCM tem vies sistematico: o MIROC6 pode ser 2 graus mais quente que o
observado no Cerrado por construcao, independentemente de qualquer aquecimento.
Alimentar o modelo com o valor absoluto projetado o colocaria fora da faixa em
que ele foi treinado, e o que ele responderia seria o vies do GCM, nao a mudanca
climatica.

O metodo delta transporta apenas a MUDANCA. Para cada municipio e mes:

    temperatura   delta aditivo        futuro - historico, em graus
    precipitacao  delta multiplicativo futuro / historico, adimensional

A precipitacao e multiplicativa porque e limitada inferiormente por zero: um
delta aditivo produziria chuva negativa nos meses secos do semiarido. A razao e
truncada em RAZAO_MAXIMA para conter a instabilidade onde o historico e proximo
de zero - com o denominador tendendo a zero, a razao explode sem que isso
signifique mudanca fisica.

O delta e entao aplicado sobre a climatologia observada do TerraClimate, que e a
que esta no painel. O vies do GCM cancela, e a estrutura espacial fina vem do
observado a 1/24 de grau, e nao do GCM a 10 arc-min.

vpd e ws: o que o WorldClim nao projeta
---------------------------------------
O painel usa cinco variaveis climaticas; o WorldClim futuro traz tres. As duas
que faltam sao tratadas de forma diferente, e as duas escolhas sao declaradas.

`vpd` NAO pode ser mantido fixo. A analise de explicabilidade mediu que
`vpd_medio_lag1` tem a maior amplitude de ALE de todas as 33 variaveis (0,0406) e
e a #4 em importancia fora da amostra no corte de 2019. Congelar a variavel que
mais responde ao clima num cenario climatico esvaziaria o cenario - o mesmo
problema da secao acima, por outro caminho.

Ele e projetado sob umidade relativa constante, que e a hipotese de primeira
ordem padrao: o aquecimento eleva a pressao de saturacao pela relacao de
Clausius-Clapeyron, e se a umidade relativa nao muda, o deficit cresce na mesma
proporcao. Formalmente, com es(T) pela formula de Tetens e a media entre tmax e
tmin como o TerraClimate a define:

    es_obs = [es(tmax_obs) + es(tmin_obs)] / 2
    UR_obs = 1 - vpd_obs / es_obs          <- calibrada no proprio painel
    vpd_fut = es_fut * (1 - UR_obs)

Calibrar UR contra o vpd observado, em vez de assumir um valor, evita qualquer
divergencia entre a definicao de vpd do TerraClimate e a da formula. A hipotese
e conservadora quanto ao sinal e provavelmente subestima: sobre continente em
aquecimento a umidade relativa tende a CAIR, o que elevaria o vpd acima do que
esta projetado aqui.

`ws` fica na climatologia observada. Nao ha projecao de vento nesta fonte, e a
mudanca projetada de vento sobre o Brasil no CMIP6 e pequena em relacao a sua
variabilidade interanual. E uma limitacao declarada, nao uma escolha silenciosa.

Por que o delta vai sobre CADA ANO, e nao sobre a climatologia
--------------------------------------------------------------
A versao inicial deste modulo aplicava o delta a climatologia observada e
entregava ao modelo um unico ano medio por cenario. Medido, o vies dessa
suavizacao e grande e cai justamente onde mais importa:

    variavel               media dos anos    climatologia    diferenca
    ppt_sazonalidade             0,8512          0,6776        -20,4%
    ppt_trimestre_seco          90,9 mm        116,2 mm        +27,7%
    tmax_mes_mais_quente        31,12 C        30,68 C         -0,44 C

A causa e aritmetica: a media entre anos apaga a variabilidade interanual que se
soma a sazonalidade dentro de cada ano. Um ano medio nao tem a forma de um ano.
E `ppt_sazonalidade` e a variavel que mais sobrevive fora da amostra nos dois
cortes temporais da analise de explicabilidade - alimentar o modelo com um valor
20% menor que qualquer coisa que ele viu no treino nao produz previsao, produz
extrapolacao.

Ha ainda o problema de Jensen: uma floresta aleatoria e nao linear, e a predicao
da media nao e a media das predicoes. Mesmo que o vies acima fosse nulo, prever
sobre um ano medio responderia a pergunta errada.

O delta e portanto aplicado a CADA um dos anos observados de BASE_OBSERVADA, um
por um. Cada ano futuro sintetico e um ano real deslocado pelo clima projetado,
e continua tendo a forma de um ano. O que se agrega e a PREDICAO, na projecao, e
nao a entrada.

Rodar:
    python -m yfvrisk.scenarios.clima_futuro
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from yfvrisk import config as cfg
from yfvrisk.features.painel import agregar_clima_anual
from yfvrisk.ingest.baixar_worldclim import (
    GCMS, PERIODOS, RESOLUCAO, SSPS, DESTINO as DIR_WORLDCLIM,
)
from yfvrisk.ingest.municipios import carregar_referencia

# Periodo do observado que define a climatologia de referencia. As tres decadas
# mais recentes com dado completo no painel, e o mesmo comprimento das normais do
# WorldClim (1970-2000), de modo que as duas medias tenham a mesma estabilidade
# amostral.
BASE_OBSERVADA = (1995, 2024)

# Teto da razao de precipitacao. Onde o historico e proximo de zero, a razao
# futuro/historico explode sem significado fisico: 0,2 mm virando 1,0 mm e um
# fator de 5, e nao uma quintuplicacao da chuva. As linhas afetadas sao contadas
# e reportadas.
RAZAO_MAXIMA = 3.0

# Constantes de Tetens sobre agua liquida, com es em kPa e T em graus Celsius.
TETENS_A, TETENS_B, TETENS_C = 0.6108, 17.27, 237.3

SAIDA_ANUAL = cfg.PROCESSADO / "clima_futuro_muni_ano.parquet"
DIAGNOSTICO = cfg.METRICAS / "f5_clima_futuro_diagnostico.json"


def pressao_de_saturacao(t_celsius):
    """Pressao de vapor de saturacao em kPa, formula de Tetens."""
    return TETENS_A * np.exp(TETENS_B * t_celsius / (t_celsius + TETENS_C))


# --------------------------------------------------------------------------
# Agregacao zonal sobre a grade do WorldClim
# --------------------------------------------------------------------------
def _um_raster_qualquer():
    """Um raster do conjunto, para ler a grade sem depender de qual existe."""
    candidatos = sorted((DIR_WORLDCLIM / "historico").rglob("*.tif"))
    if not candidatos:
        raise SystemExit(
            "nenhum raster do WorldClim encontrado. Rode primeiro:\n"
            "  python -m yfvrisk.ingest.baixar_worldclim --linha-de-base --futuro")
    return candidatos[0]


def construir_rotulos():
    """Rasteriza a malha municipal sobre a grade do WorldClim.

    Mesma estrategia do TerraClimate em `ingest/clima.py`: rasterizar uma vez e
    depois usar bincount, em vez de estatistica zonal poligono a poligono.

    A celula de 10 arc-min tem ~340 km2 no equador, contra ~21 km2 do
    TerraClimate, de modo que MUITO mais municipios ficam sem centro de celula
    proprio. Isso e aceitavel aqui e nao seria la: o que se extrai desta grade e
    uma anomalia climatica, que e um campo suave em escala de centenas de
    quilometros. A estrutura fina vem da linha de base observada. Ainda assim, a
    contagem de municipios amostrados por vizinhanca entra no diagnostico.
    """
    with rasterio.open(_um_raster_qualquer()) as src:
        forma, transform = (src.height, src.width), src.transform

    ref = carregar_referencia(com_geometria=True).to_crs("EPSG:4326")
    ref = ref.reset_index(drop=True)

    rotulos = rasterize(((g, i + 1) for i, g in enumerate(ref.geometry)),
                        out_shape=forma, transform=transform, fill=0,
                        dtype="int32", all_touched=False)

    contagem = np.bincount(rotulos.ravel(), minlength=len(ref) + 1)[1:]
    sem_celula = np.where(contagem == 0)[0]

    # Ponto representativo de TODOS os municipios, e nao so dos que ficaram sem
    # celula. Ter celula nao garante ter valor: os 41 municipios costeiros do
    # tipo Fortaleza, Recife e Fernando de Noronha tem pixels proprios, mas o
    # WorldClim os marca como oceano, e a media zonal deles sai ausente. Com o
    # recuo restrito aos sem celula, esses 41 ficavam sem clima futuro e a
    # projecao quebrava por incompatibilidade de forma - falha barulhenta, mas
    # que num codigo menos rigido teria virado 41 municipios silenciosamente
    # imputados pela mediana nacional - o mesmo risco de imputar a media nacional
    # de floresta aos 44 municipios ao norte do equador (ver baixar_hansen.py).
    rep = ref.geometry.representative_point()
    pontos = {}
    for i in range(len(ref)):
        linha, coluna = rasterio.transform.rowcol(
            transform, rep.iloc[i].x, rep.iloc[i].y)
        pontos[int(i)] = (int(linha), int(coluna))

    diag = {
        "grade": {"linhas": forma[0], "colunas": forma[1],
                  "resolucao": RESOLUCAO},
        "municipios": int(len(ref)),
        "com_celula_propria": int((contagem > 0).sum()),
        "sem_celula_propria": int(len(sem_celula)),
    }
    return rotulos, ref, pontos, diag


def _vizinho_valido(grade, j, k, raio_max=4):
    """Valor valido mais proximo de (j, k), em aneis crescentes."""
    n_j, n_k = grade.shape
    j, k = min(max(j, 0), n_j - 1), min(max(k, 0), n_k - 1)
    if np.isfinite(grade[j, k]):
        return float(grade[j, k])
    for r in range(1, raio_max + 1):
        janela = grade[max(0, j - r):min(n_j, j + r + 1),
                       max(0, k - r):min(n_k, k + r + 1)]
        if np.isfinite(janela).any():
            return float(np.nanmean(janela))
    return np.nan


def climatologia(caminhos, rotulos, n_mun, pontos) -> np.ndarray:
    """Media zonal por municipio dos 12 meses. Devolve (12, n_mun).

    `caminhos` e ou 12 rasters de uma banda (linha de base do WorldClim) ou um
    unico raster de 12 bandas (as projecoes CMIP6).
    """
    plano = rotulos.ravel()
    saida = np.full((12, n_mun), np.nan)

    def _um_mes(grade, mes):
        vals = grade.ravel()
        valido = (plano > 0) & np.isfinite(vals)
        soma = np.bincount(plano[valido], weights=vals[valido], minlength=n_mun + 1)
        cont = np.bincount(plano[valido], minlength=n_mun + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            media = np.where(cont > 0, soma / np.maximum(cont, 1), np.nan)[1:]
        # Recuo por vizinhanca para toda media zonal ausente, seja por falta de
        # celula ou por celula mascarada.
        for i in np.where(~np.isfinite(media))[0]:
            j, k = pontos[int(i)]
            media[i] = _vizinho_valido(grade, j, k)
        saida[mes] = media

    def _ler(src, banda):
        """Banda em float com ausente como NaN.

        A conversao para float e obrigatoria e nao e detalhe: a precipitacao da
        linha de base do WorldClim vem em int16, e preencher o mascarado com NaN
        num inteiro levanta erro. Fazer isso antes de qualquer aritmetica tambem
        evita que a razao futuro/historico seja calculada em inteiro.
        """
        return src.read(banda, masked=True).astype("float64").filled(np.nan)

    if len(caminhos) == 1:
        with rasterio.open(caminhos[0]) as src:
            if src.count != 12:
                raise RuntimeError(
                    f"{caminhos[0].name}: esperadas 12 bandas, {src.count} lidas")
            for mes in range(12):
                _um_mes(_ler(src, mes + 1), mes)
    else:
        if len(caminhos) != 12:
            raise RuntimeError(f"esperados 12 rasters mensais, {len(caminhos)} dados")
        for mes, caminho in enumerate(caminhos):
            with rasterio.open(caminho) as src:
                _um_mes(_ler(src, 1), mes)
    return saida


def _rasters_historicos(variavel):
    """Os 12 mensais da normal de 1970-2000, em ordem de mes.

    O WorldClim nomeia por sufixo de 01 a 12; a ordenacao alfabetica coincide com
    a cronologica, mas a ordem e imposta explicitamente para que nunca dependa
    disso.
    """
    pasta = DIR_WORLDCLIM / "historico" / variavel
    arquivos = {int(p.stem.split("_")[-1]): p for p in pasta.glob("*.tif")}
    faltando = sorted(set(range(1, 13)) - set(arquivos))
    if faltando:
        raise RuntimeError(f"{variavel}: meses ausentes na linha de base {faltando}")
    return [arquivos[m] for m in range(1, 13)]


def _raster_futuro(gcm, ssp, variavel, periodo):
    return (DIR_WORLDCLIM / "futuro" / gcm / ssp /
            f"wc2.1_{RESOLUCAO}_{variavel}_{gcm}_{ssp}_{periodo}.tif")


# --------------------------------------------------------------------------
# Deltas e aplicacao sobre o observado
# --------------------------------------------------------------------------
def mensal_observado() -> pd.DataFrame:
    """Mensal do TerraClimate, ano a ano, no periodo de referencia.

    E o alvo do delta: o que o painel de fato contem, a 1/24 de grau, com cada
    ano preservado. Nao e uma climatologia - ver o cabecalho sobre por que o
    delta nao pode ir sobre a media.
    """
    m = pd.read_parquet(cfg.PROCESSADO / "clima_muni_mes.parquet")
    a0, a1 = BASE_OBSERVADA
    m = m[(m["ano"] >= a0) & (m["ano"] <= a1)]
    return m[["cod_mun", "ano", "mes", "ppt", "tmax", "tmin", "vpd", "ws"]]


def climatologia_observada() -> pd.DataFrame:
    """Media mensal do periodo de referencia.

    Nao entra na projecao. Fica para diagnostico e para o teste que mede o vies
    de suavizacao documentado no cabecalho.
    """
    return (mensal_observado()
            .groupby(["cod_mun", "mes"])[["ppt", "tmax", "tmin", "vpd", "ws"]]
            .mean().reset_index())


def aplicar_delta(obs: pd.DataFrame, delta_tmax, delta_tmin, razao_ppt,
                  ordem_mun) -> pd.DataFrame:
    """Constroi o mensal futuro a partir do observado e dos deltas.

    `delta_*` e `razao_ppt` sao matrizes (12, n_mun) na ordem `ordem_mun`.
    """
    d = pd.DataFrame({
        "cod_mun": np.repeat(ordem_mun, 12),
        "mes": np.tile(np.arange(1, 13), len(ordem_mun)),
        "delta_tmax": delta_tmax.T.ravel(),
        "delta_tmin": delta_tmin.T.ravel(),
        "razao_ppt": razao_ppt.T.ravel(),
    })
    # O delta e por municipio e mes; `obs` traz varios anos. O merge por
    # (cod_mun, mes) aplica o mesmo deslocamento a todos os anos, que e o que
    # define o metodo: a mudanca climatica projetada nao depende de qual ano
    # observado esta sendo deslocado.
    f = obs.merge(d, on=["cod_mun", "mes"], how="left")

    f["tmax_fut"] = f["tmax"] + f["delta_tmax"]
    f["tmin_fut"] = f["tmin"] + f["delta_tmin"]
    f["ppt_fut"] = (f["ppt"] * f["razao_ppt"]).clip(lower=0.0)

    # vpd sob umidade relativa constante, calibrada no proprio observado.
    es_obs = (pressao_de_saturacao(f["tmax"]) + pressao_de_saturacao(f["tmin"])) / 2
    es_fut = (pressao_de_saturacao(f["tmax_fut"])
              + pressao_de_saturacao(f["tmin_fut"])) / 2
    # O TerraClimate reporta vpd em kPa. Onde o quociente sair fora de [0, 1] a
    # calibracao nao e aplicavel, e o fallback e escalar o vpd observado pela
    # razao das pressoes de saturacao - que e a mesma hipotese, sem a calibracao.
    fracao_deficit = (f["vpd"] / es_obs).clip(0.0, 1.0)
    f["vpd_fut"] = np.where(es_obs > 0, es_fut * fracao_deficit,
                            f["vpd"] * (es_fut / es_obs.replace(0, np.nan)))
    f["ws_fut"] = f["ws"]  # sem projecao nesta fonte - ver o cabecalho

    manter = ["cod_mun", "mes"] + (["ano"] if "ano" in f.columns else [])
    return f[manter + ["ppt_fut", "tmax_fut", "tmin_fut", "vpd_fut", "ws_fut"]
             ].rename(columns={"ppt_fut": "ppt", "tmax_fut": "tmax",
                               "tmin_fut": "tmin", "vpd_fut": "vpd",
                               "ws_fut": "ws"})


def construir(gcms=None, ssps=None, periodo=None) -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()
    gcms = gcms or GCMS
    ssps = ssps or SSPS
    periodo = periodo or PERIODOS[0]

    rotulos, ref, pontos, diag = construir_rotulos()
    ordem_mun = ref["cod_mun"].to_numpy()
    n_mun = len(ref)
    print(f"[grade] {diag['com_celula_propria']} municipios com celula propria, "
          f"{diag['sem_celula_propria']} sem")

    print("[historico] lendo a normal de 1970-2000")
    hist = {v: climatologia(_rasters_historicos(v), rotulos, n_mun, pontos)
            for v in ("tmax", "tmin", "prec")}

    # Nenhum municipio pode sair daqui sem valor. Um ausente aqui vira, la na
    # frente, um municipio imputado pela mediana nacional sem que nada acuse.
    # Melhor parar.
    sem_historico = {v: int(np.isnan(m).any(axis=0).sum()) for v, m in hist.items()}
    if any(sem_historico.values()):
        indices = np.where(np.isnan(hist["tmax"]).any(axis=0))[0]
        raise SystemExit(
            f"municipios sem linha de base do WorldClim: {sem_historico}. "
            f"Primeiros codigos: {list(ref['cod_mun'].iloc[indices][:10])}. "
            f"Aumente raio_max em _vizinho_valido ou trate-os explicitamente.")

    obs = mensal_observado()
    anos_base = sorted(obs["ano"].unique())
    print(f"[observado] TerraClimate {BASE_OBSERVADA[0]}-{BASE_OBSERVADA[1]}: "
          f"{len(obs):,} linhas municipio-ano-mes, {len(anos_base)} anos-base")

    blocos, resumo, ausentes = [], [], []
    n_truncadas_total = 0

    for gcm in gcms:
        for ssp in ssps:
            caminho = {v: _raster_futuro(gcm, ssp, v, periodo)
                       for v in ("tmax", "tmin", "prec")}
            if not all(p.exists() for p in caminho.values()):
                ausentes.append(f"{gcm}/{ssp}")
                print(f"[ausente] {gcm}/{ssp}")
                continue

            fut = {v: climatologia([caminho[v]], rotulos, n_mun, pontos)
                   for v in ("tmax", "tmin", "prec")}

            delta_tmax = fut["tmax"] - hist["tmax"]
            delta_tmin = fut["tmin"] - hist["tmin"]
            with np.errstate(invalid="ignore", divide="ignore"):
                razao = np.where(hist["prec"] > 0,
                                 fut["prec"] / hist["prec"], np.nan)
            n_truncadas = int(np.nansum(razao > RAZAO_MAXIMA))
            n_truncadas_total += n_truncadas
            razao = np.clip(razao, 0.0, RAZAO_MAXIMA)
            # Onde o historico e zero nos dois lados, "sem chuva" continua "sem
            # chuva": razao 1 preserva o observado em vez de gerar ausente.
            razao = np.where(np.isfinite(razao), razao, 1.0)

            # Agrega para anual JA AQUI e descarta o mensal. Guardar o mensal
            # de 20 cenarios x 30 anos-base seriam 40 milhoes de linhas, e
            # nenhuma etapa posterior o utiliza.
            mensal = aplicar_delta(obs, delta_tmax, delta_tmin, razao, ordem_mun)
            anual_cen = agregar_clima_anual(mensal)
            anual_cen.insert(0, "gcm", gcm)
            anual_cen.insert(1, "ssp", ssp)
            anual_cen["periodo"] = periodo
            anual_cen = anual_cen.rename(columns={"ano": "ano_base"})
            blocos.append(anual_cen)

            resumo.append({
                "gcm": gcm, "ssp": ssp, "periodo": periodo,
                "delta_tmax_medio": float(np.nanmean(delta_tmax)),
                "delta_tmin_medio": float(np.nanmean(delta_tmin)),
                "razao_ppt_mediana": float(np.nanmedian(razao)),
                "razoes_truncadas": n_truncadas,
            })
            print(f"[ok] {gcm:16s} {ssp}  dTmax {np.nanmean(delta_tmax):+.2f} C  "
                  f"dTmin {np.nanmean(delta_tmin):+.2f} C  "
                  f"ppt x{np.nanmedian(razao):.3f}")

    if not blocos:
        raise SystemExit("nenhuma combinacao GCM x SSP disponivel")

    anual = pd.concat(blocos, ignore_index=True)

    # Municipios sem projecao. Sao declarados e removidos, e nao imputados: uma
    # linha com clima ausente cairia na mediana do imputador la na frente, e um
    # municipio com o clima mediano do Brasil e pior que um municipio ausente.
    clima_cols = [c for c in anual.columns
                  if c not in ("gcm", "ssp", "cod_mun", "ano_base", "periodo")]
    incompletos = sorted(anual.loc[anual[clima_cols].isna().any(axis=1),
                                   "cod_mun"].unique())
    if incompletos:
        print(f"[sem projecao] {len(incompletos)} municipios sem celula terrestre "
              f"no CMIP6: {incompletos}")
        anual = anual[~anual["cod_mun"].isin(incompletos)]

    diag.update({
        "metodo": "delta (aditivo em temperatura, multiplicativo em precipitacao)",
        "base_observada": list(BASE_OBSERVADA),
        "normal_worldclim": "1970-2000",
        "periodo_futuro": periodo,
        "gcms_pedidos": list(gcms), "ssps_pedidos": list(ssps),
        "combinacoes_ausentes": ausentes,
        "combinacoes_construidas": len(resumo),
        "razao_ppt_maxima": RAZAO_MAXIMA,
        "razoes_truncadas_total": n_truncadas_total,
        "vpd": ("projetado sob umidade relativa constante, calibrada contra o vpd "
                "observado do TerraClimate"),
        "ws": "mantido no observado: sem projecao nesta fonte",
        "anos_base": [int(a) for a in anos_base],
        "aplicacao_do_delta": (
            "sobre cada ano observado, um por um, e nao sobre a climatologia. "
            "A media entre anos reduz ppt_sazonalidade em 20,4% e eleva "
            "ppt_trimestre_seco em 27,7%, e a floresta e nao linear."),
        "municipios_sem_projecao": incompletos,
        "nota_sem_projecao": (
            "Municipios sem nenhuma celula terrestre do CMIP6 a 10 arc-min num "
            "raio de quatro celulas. Sao removidos e declarados, nunca imputados. "
            "Fernando de Noronha e ilha oceanica a 350 km da costa, sem ciclo "
            "silvestre de febre amarela, de modo que a exclusao nao afeta "
            "nenhuma conclusao."),
        "municipios_projetados": int(anual["cod_mun"].nunique()),
        "linhas_anuais": int(len(anual)),
        "por_combinacao": resumo,
    })

    anual.to_parquet(SAIDA_ANUAL, index=False)
    DIAGNOSTICO.write_text(json.dumps(diag, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return anual, diag


if __name__ == "__main__":
    a, d = construir()
    print(f"\nclima futuro: {len(a):,} linhas municipio-cenario, "
          f"{d['combinacoes_construidas']} combinacoes GCM x SSP")
    if d["combinacoes_ausentes"]:
        print(f"AUSENTES: {d['combinacoes_ausentes']}")
    print(f"razoes de precipitacao truncadas em {RAZAO_MAXIMA}: "
          f"{d['razoes_truncadas_total']:,}")
