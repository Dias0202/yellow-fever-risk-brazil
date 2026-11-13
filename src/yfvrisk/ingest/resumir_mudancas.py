"""Resumo do painel para o corpo do Pull Request automatico.

Imprime os numeros que um revisor precisa olhar antes de aprovar uma atualizacao
de dados. A escolha do que mostrar cobre os modos de falha silenciosa mais comuns
em dados como estes: contagem de municipios que muda, prevalencia que salta por
mudanca de criterio, fonte que fica defasada, cobertura de raster que desaparece
numa regiao inteira.

Uso:  python -m yfvrisk.ingest.resumir_mudancas
"""

from __future__ import annotations

import sys

import pandas as pd

from yfvrisk import config as cfg


def _ler(nome):
    caminho = cfg.PROCESSADO / nome
    return pd.read_parquet(caminho) if caminho.exists() else None


def main() -> int:
    print("## Resumo do painel\n")

    ref = _ler("municipios_referencia.parquet")
    if ref is not None:
        print(f"- Municipios: **{len(ref)}** (esperado {cfg.N_MUNICIPIOS})")
        print(f"- Unidades federativas: {ref['uf'].nunique()}")
        print(f"- Area total: {ref['area_km2'].sum():,.0f} km2\n")

    epi = _ler("painel_epi_muni_ano.parquet")
    if epi is not None:
        print("### Desfecho\n")
        print(f"- Linhas: {len(epi):,}")
        print(f"- Prevalencia anual de yf_event: **{epi['yf_event'].mean():.4%}**")
        print(f"- Casos humanos: {epi['caso_humano'].sum():,} municipio-ano")
        print(f"- Epizootias: {epi['epizootia'].sum():,} municipio-ano\n")

        print("| Ano | Municipios com evento | Casos humanos | Epizootias |")
        print("|---|---|---|---|")
        por_ano = epi.groupby("ano")[["yf_event", "caso_humano", "epizootia"]].sum()
        for ano in sorted(por_ano.index)[-8:]:
            r = por_ano.loc[ano]
            print(f"| {ano} | {int(r['yf_event'])} | {int(r['caso_humano'])} "
                  f"| {int(r['epizootia'])} |")
        print()

    vac = _ler("vacinacao_muni_ano.parquet")
    if vac is not None:
        ultimo = vac[vac["cobertura_disponivel"] == 1]["ano"].max()
        media = vac.loc[vac["ano"] == ultimo, "imunidade_acumulada"].mean()
        print("### Vacinacao\n")
        print(f"- Ultimo ano com dado do PNI: **{ultimo}**")
        print(f"- Imunidade acumulada media em {ultimo}: {media:.1%}")
        print(f"- Registros marcados como implausiveis (>200%): "
              f"{int(vac['cobertura_implausivel'].sum()):,}\n")

    pop = _ler("populacao_muni_ano.parquet")
    if pop is not None:
        ultimo = int(pop["ano"].max())
        print("### Populacao\n")
        print(f"- Total nacional em {ultimo}: "
              f"{int(pop.loc[pop['ano'] == ultimo, 'populacao'].sum()):,}")
        print(f"- Linhas interpoladas: "
              f"{int((pop['populacao_observada'] == 0).sum()):,} "
              f"({(pop['populacao_observada'] == 0).mean():.1%})\n")

    flo = _ler("floresta_muni_ano.parquet")
    if flo is not None:
        cob = flo.groupby("cod_mun")["hansen_cobertura"].first()
        print("### Floresta\n")
        print(f"- Municipios com cobertura de Hansen abaixo de 50%: "
              f"**{int((cob < 0.5).sum())}**")
        ano = min(2024, int(flo['ano'].max()))
        print(f"- Floresta total em {ano}: "
              f"{flo.loc[flo['ano'] == ano, 'floresta_km2'].sum():,.0f} km2")
        print(f"- Perda acumulada 2001-{ano}: "
              f"{flo.loc[flo['ano'] == ano, 'perda_acumulada_km2'].sum():,.0f} km2\n")

    print("### Pontos de atencao para a revisao\n")
    print("- A contagem de municipios mudou? Deve permanecer em 5.570.")
    print("- A prevalencia anual saltou de ordem de grandeza? Pode indicar "
          "mudanca de criterio na fonte.")
    print("- Alguma fonte entrou em defasagem? Ver a etapa de conferencia.")
    print("- A cobertura do Hansen caiu em alguma regiao? Roraima e Amapa ja "
          "ficaram sem dado uma vez, por faixa de latitude nao baixada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
