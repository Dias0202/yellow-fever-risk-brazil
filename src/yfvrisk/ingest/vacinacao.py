"""Cobertura vacinal e imunidade acumulada contra febre amarela.

E a variavel mais facil de construir errado no painel: tratada de forma ingenua,
ela faz AUMENTAR a cobertura vacinal AUMENTAR o risco previsto. Quatro cuidados
sao necessarios, e todos sao tratados aqui.

Cuidado 1 - ausencia nao e suscetibilidade total
------------------------------------------------
Preencher cobertura ausente com 0 (isto e, "100% suscetivel") e perigoso porque a
ausencia e sistematicamente maior nos primeiros anos da serie e em municipios
pequenos e remotos - justamente os de menor notificacao. O modelo aprenderia
"sem dado de vacina, logo sem evento".

Aqui a ausencia nunca vira zero de protecao. Ela vira ausencia de INCREMENTO na
imunidade acumulada, que e coisa diferente, e fica marcada em coluna propria.

Cuidado 2 - imunidade, nao cobertura pontual
--------------------------------------------
A vacina de febre amarela confere protecao duradoura. A cobertura de um unico ano
nao representa a fracao protegida da populacao: um municipio que vacinou 90% em
2018 e nada em 2019 nao volta a ser suscetivel em 2019.

A variavel passa a ser a imunidade ACUMULADA, pela formula de composicao

    I_t = I_{t-1} * (1 - d) + c_t * (1 - I_{t-1} * (1 - d))

em que c_t e a cobertura do ano e d e a taxa anual de renovacao populacional
(nascimentos e migracao trazendo pessoas nao vacinadas). Cada ano vacina uma
fracao c_t dos ainda suscetiveis, e a imunidade satura em 1.

Sobre valores acima de 100%
---------------------------
O PNI reporta doses aplicadas sobre populacao-alvo estimada, e o arquivo traz
valores de ate 214%. Isso indica campanha de massa, nao erro. Truncar em 100%
descartaria essa informacao; aqui o excedente e aproveitado no acumulo, que
satura naturalmente.

Cuidado 3 - a cobertura e registrada no ano do surto, em resposta a ele
----------------------------------------------------------------------
Mesmo acumulada, a imunidade contemporanea tem importancia por permutacao
NEGATIVA fora da amostra. O diagnostico esta em
`analysis/f5_00_diagnostico_vacinacao.py`.

Um estudo de evento sobre a cobertura anual, centrada pelo ano e pelo municipio,
da o perfil abaixo em torno do ano do surto:

    k = -3   -0,33      k = 0    +14,62   <- ano do surto
    k = -2   +0,82      k = +1   +14,68
    k = -1   +2,36      k = +2   +14,10

A cobertura fica no normal do municipio ate o ano anterior e salta 4,5x no ano do
surto. E o bloqueio vacinal e a campanha de massa que o proprio surto dispara, e
o PNI os registra no ano em que ocorrem. A variavel esta a JUSANTE do desfecho: o
modelo nao aprende que a vacinacao protege, aprende que municipio que vacinou
muito neste ano e municipio que teve surto neste ano.

Nenhum reajuste de escala, imputacao ou formula de acumulo conserta isso. O que
conserta e defasar. Medida por dupla centragem, a associacao entre imunidade e
evento dentro do municipio e:

    defasagem 0 anos   rho = +0,0817   (p = 7,5e-49)   <- imunidade contemporanea
    defasagem 1 ano    rho = +0,0167   (p = 0,0031)
    defasagem 2 anos   rho = -0,0054   (p = 0,35)

Dois anos zeram a contaminacao; um ano nao basta, porque em k = +1 a cobertura
ainda esta +14,7 acima do normal. Dai `imunidade_defasada2` ser a variavel de
modelagem, e nao `imunidade_acumulada`. A defasagem tambem e a escolha correta
do ponto de vista operacional: so se pode agir sobre a imunidade que ja existe
quando o ano comeca.

Cuidado 4 - a acumulacao comeca quando o REGISTRO comeca
---------------------------------------------------------
A disponibilidade do PNI vai de 5,3% das linhas em 1994 a 99,9% em 2022.
Municipio que so aparece na fonte em 2005 chega a 2014 com menos imunidade
acumulada que um vizinho identico presente desde 1994. Descontada a cobertura
media efetivamente observada, a imunidade acumulada em 2014 ainda correlaciona
rho = -0,304 com o ano do primeiro registro: e cobertura de VIGILANCIA entrando
disfarcada de cobertura vacinal.

`anos_com_registro_pni` torna isso explicito, para que o modelo possa separar as
duas coisas em vez de confundi-las. A variavel serve tambem como proxy de
capacidade administrativa municipal.

Sobre `suscetivel_pct`
----------------------
Era exatamente `1 - imunidade_acumulada`, isto e, a mesma variavel com o sinal
trocado. Duas colunas perfeitamente colineares dividem entre si a importancia de
Gini - e por isso aparecem no topo de um ranking de Gini - e ocupam duas vagas no
sorteio de `max_features="sqrt"` para carregar uma informacao so. Continua no
arquivo, mas fora do grupo de modelagem.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yfvrisk import config as cfg
from yfvrisk.ingest.municipios import carregar_referencia

ARQUIVO = cfg.FONTE_MANUAL / "vacin.csv"

# Fracao da populacao substituida por ano por pessoas nao vacinadas (nascimentos e
# migracao de entrada). Valor de referencia proximo da taxa bruta de natalidade
# brasileira nas ultimas decadas. Entra como parametro explicito, e nao como
# constante escondida, para que a sensibilidade a ele possa ser testada.
RENOVACAO_ANUAL_PADRAO = 0.015

# Defasagem, em anos, entre a imunidade e o ano que ela ajuda a prever. Fixada em
# 2 pelo estudo de evento descrito no cabecalho: em k = +1 a cobertura ainda esta 14,7 pontos
# acima do normal do municipio, e so em k = +2 a associacao dentro do municipio
# deixa de ser distinguivel de zero. Parametro explicito para que a sensibilidade
# a ele possa ser medida.
DEFASAGEM_IMUNIDADE = 2


def ler_bruto() -> pd.DataFrame:
    """Le o arquivo do PNI no formato largo, tratando BOM e virgula decimal."""
    df = pd.read_csv(ARQUIVO, sep=";", dtype=str, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "cod6"})
    df["cod6"] = df["cod6"].str.strip().str.zfill(6)

    anos = [c for c in df.columns if c.strip().isdigit()]
    longo = df.melt(id_vars="cod6", value_vars=anos,
                    var_name="ano", value_name="bruto")
    longo["ano"] = longo["ano"].astype(int)
    longo["cobertura_pct"] = (longo["bruto"].str.strip()
                              .str.replace(".", "", regex=False)
                              .str.replace(",", ".", regex=False)
                              .replace("", np.nan).astype(float))
    return longo.drop(columns="bruto")


def acumular_imunidade(cobertura: np.ndarray, renovacao: float) -> np.ndarray:
    """Imunidade acumulada ano a ano, saturando em 1.

    Anos sem dado nao incrementam a imunidade, mas tambem nao a zeram - so sofrem
    a renovacao populacional - zerar a protecao sempre que falta o registro
    confundiria lacuna de registro com suscetibilidade.
    """
    imunidade = np.empty(len(cobertura), dtype=float)
    atual = 0.0
    for i, c in enumerate(cobertura):
        atual *= (1.0 - renovacao)
        if np.isfinite(c):
            fracao = min(max(c / 100.0, 0.0), 1.0)
            atual = atual + fracao * (1.0 - atual)
        imunidade[i] = atual
    return imunidade


def construir(renovacao: float = RENOVACAO_ANUAL_PADRAO) -> tuple[pd.DataFrame, dict]:
    cfg.garantir_diretorios()
    ref = carregar_referencia(com_geometria=False)
    mapa = {c[:6]: c for c in ref["cod_mun"]}

    bruto = ler_bruto()
    bruto["cod_mun"] = bruto["cod6"].map(mapa)

    orfaos = sorted(bruto.loc[bruto["cod_mun"].isna(), "cod6"].unique())
    bruto = bruto.dropna(subset=["cod_mun"])

    ano_min_fonte = int(bruto["ano"].min())
    ano_max_fonte = int(bruto["ano"].max())

    # Painel completo: todo municipio, todo ano do recorte do estudo.
    anos = range(cfg.ANO_INICIO, cfg.ANO_FIM + 1)
    painel = pd.MultiIndex.from_product(
        [ref["cod_mun"], anos], names=["cod_mun", "ano"]).to_frame(index=False)
    painel = painel.merge(bruto[["cod_mun", "ano", "cobertura_pct"]],
                          on=["cod_mun", "ano"], how="left")

    painel["cobertura_disponivel"] = painel["cobertura_pct"].notna().astype(int)

    # Outliers reais da fonte, nao de leitura: o arquivo traz valores como
    # "10.100,00" para o municipio 150475 em 2017. Sao tipicos de municipio
    # pequeno em campanha de massa, quando a populacao-alvo estimada no
    # denominador e muito menor que o numero de doses aplicadas, inclusive a
    # pessoas de municipios vizinhos. Ficam marcados para poderem ser reportados
    # e para permitir analise de sensibilidade sem eles.
    painel["cobertura_implausivel"] = (painel["cobertura_pct"] > 200).astype(int)
    painel["fora_do_periodo_da_fonte"] = (
        (painel["ano"] < ano_min_fonte) | (painel["ano"] > ano_max_fonte)
    ).astype(int)

    painel = painel.sort_values(["cod_mun", "ano"]).reset_index(drop=True)
    painel["imunidade_acumulada"] = (
        painel.groupby("cod_mun", sort=False)["cobertura_pct"]
        .transform(lambda s: acumular_imunidade(s.to_numpy(), renovacao))
    )
    # Colinear com a anterior por construcao. Mantida no arquivo, fora do grupo
    # de modelagem - ver a nota no cabecalho.
    painel["suscetivel_pct"] = 1.0 - painel["imunidade_acumulada"]

    # --- Variaveis de modelagem, admissiveis quanto a seta do tempo ---------
    # A imunidade que pode explicar o risco de um ano e a que ja existia antes da
    # campanha que o proprio surto dispara. Ver causa 3 no cabecalho.
    g = painel.groupby("cod_mun", sort=False)
    painel["imunidade_defasada2"] = g["imunidade_acumulada"].shift(DEFASAGEM_IMUNIDADE)

    # Contraste transversal: quanto este municipio esta acima ou abaixo do pais
    # naquele momento. E o que um cenario de vacinacao move - um cenario nao troca
    # um municipio por outro, ele desloca a posicao do municipio em relacao aos
    # demais. Separar o contraste do nivel evita que o modelo leia como imunidade
    # a tendencia nacional de saturacao, que e comum a todos.
    painel["imunidade_relativa2"] = (
        painel["imunidade_defasada2"]
        - painel.groupby("ano")["imunidade_defasada2"].transform("mean"))

    # Anos de presenca do municipio na serie do PNI, tambem defasados. Torna
    # explicita a contaminacao da causa 4, em vez de deixa-la disfarcada de
    # imunidade. Ver causa 4 no cabecalho.
    painel["anos_com_registro_pni"] = (
        g["cobertura_disponivel"]
        .transform(lambda s: s.shift(DEFASAGEM_IMUNIDADE).cumsum()))

    diag = {
        "arquivo": ARQUIVO.name,
        "renovacao_anual": renovacao,
        "defasagem_imunidade": DEFASAGEM_IMUNIDADE,
        "variaveis_de_modelagem": ["imunidade_defasada2", "imunidade_relativa2",
                                   "anos_com_registro_pni"],
        "nota_defasagem": (
            "A cobertura salta 4,5x no ano do surto e permanece elevada por tres "
            "anos: e o bloqueio vacinal que o surto dispara, registrado pelo PNI "
            "no ano em que ocorre. A imunidade contemporanea esta a jusante do "
            "desfecho. Medida por dupla centragem, a associacao dentro do "
            "municipio cai de rho = +0,082 sem defasagem para -0,005 (p = 0,35) "
            "com dois anos. Ver analysis/f5_00_diagnostico_vacinacao.py."
        ),
        "anos_na_fonte": [ano_min_fonte, ano_max_fonte],
        "anos_no_painel": [cfg.ANO_INICIO, cfg.ANO_FIM],
        "anos_sem_fonte": sorted(set(anos) - set(range(ano_min_fonte, ano_max_fonte + 1))),
        "codigos_nao_encontrados_na_malha": orfaos,
        "linhas": int(len(painel)),
        "cobertura_disponivel_pct": float(painel["cobertura_disponivel"].mean()),
        "cobertura_acima_de_100": int((painel["cobertura_pct"] > 100).sum()),
        "cobertura_acima_de_200_implausivel": int(painel["cobertura_implausivel"].sum()),
        "cobertura_acima_de_500": int((painel["cobertura_pct"] > 500).sum()),
        "cobertura_max": float(painel["cobertura_pct"].max()),
        "nota_valores_extremos": (
            "Valores acima de 100% sao esperados: o PNI reporta doses aplicadas "
            "sobre populacao-alvo estimada, e campanhas de massa ultrapassam o "
            "denominador. Os acima de 200% sao marcados como implausiveis; o "
            "maximo, 10.100% no municipio 150475 em 2017, esta assim na fonte. "
            "No acumulo, o incremento anual satura em 100% dos ainda suscetiveis."
        ),
        "disponibilidade_por_ano": {
            int(a): float(g["cobertura_disponivel"].mean())
            for a, g in painel.groupby("ano")
        },
        "imunidade_media_por_ano": {
            int(a): round(float(g["imunidade_acumulada"].mean()), 4)
            for a, g in painel.groupby("ano")
        },
    }

    painel.to_parquet(cfg.PROCESSADO / "vacinacao_muni_ano.parquet", index=False)
    (cfg.METRICAS / "f2_vacinacao_diagnostico.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    return painel, diag


if __name__ == "__main__":
    p, d = construir()
    print(f"vacinacao: {len(p):,} linhas")
    print(f"cobertura disponivel em {d['cobertura_disponivel_pct']:.1%} das linhas")
    print(f"valores acima de 100%: {d['cobertura_acima_de_100']:,} "
          f"(maximo {d['cobertura_max']:.1f}%)")
    print(f"anos sem fonte: {d['anos_sem_fonte']}")
    print(f"codigos orfaos: {d['codigos_nao_encontrados_na_malha']}")
    print("\nano  disponibilidade  imunidade media")
    for a in sorted(d["disponibilidade_por_ano"]):
        print(f"{a}       {d['disponibilidade_por_ano'][a]:6.1%}"
              f"         {d['imunidade_media_por_ano'][a]:6.3f}")
