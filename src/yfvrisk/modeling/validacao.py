"""Esquemas de validacao cruzada em bloco.

Numa particao aleatoria por linha, anos adjacentes do mesmo municipio - quase
duplicatas, sobretudo nas variaveis climaticas defasadas - caem dos dois lados,
e o modelo nunca precisa prever um periodo que nao viu. O desempenho medido
assim e interpolacao temporal, nao previsao.

Este modulo implementa os esquemas que correspondem ao uso pretendido do modelo.

  JanelaTemporal      treina em <= t, testa em t+1..t+h. E o unico esquema que
                      corresponde a pergunta "qual sera o risco no proximo ano".

  BlocoEspacial       leave-region-out por UF ou bioma. Mede transferencia para
                      areas nao vistas, que e exatamente o que a projecao para o
                      Sul exige.

  BlocoEspacoTemporal combina os dois: nem o municipio nem o periodo aparecem no
                      treino. E o teste mais severo e o mais honesto.

Todos devolvem indices, nao dados, para poderem ser usados com qualquer estimador.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class JanelaTemporal:
    """Validacao em janela deslizante para frente.

    Parametros
    ----------
    ano_inicial_teste : primeiro ano a ser previsto
    horizonte         : quantos anos a frente cada dobra preve
    minimo_treino     : anos minimos de treino antes da primeira dobra
    expansiva         : True mantem todo o passado no treino (janela expansiva);
                        False usa apenas os `minimo_treino` anos mais recentes
                        (janela rolante), util para testar se o passado remoto
                        ajuda ou atrapalha
    """

    ano_inicial_teste: int
    horizonte: int = 1
    minimo_treino: int = 10
    expansiva: bool = True

    def split(self, anos: np.ndarray):
        unicos = np.array(sorted(np.unique(anos)))
        for alvo in unicos[unicos >= self.ano_inicial_teste]:
            anos_teste = [a for a in range(alvo, alvo + self.horizonte) if a in unicos]
            if not anos_teste:
                continue
            if self.expansiva:
                anos_treino = unicos[unicos < alvo]
            else:
                anos_treino = unicos[(unicos < alvo)
                                     & (unicos >= alvo - self.minimo_treino)]
            if len(anos_treino) < self.minimo_treino:
                continue
            treino = np.where(np.isin(anos, anos_treino))[0]
            teste = np.where(np.isin(anos, anos_teste))[0]
            if len(teste) and len(treino):
                yield treino, teste, {"ano_teste": int(alvo),
                                      "anos_treino": [int(anos_treino.min()),
                                                      int(anos_treino.max())]}


@dataclass
class BlocoEspacial:
    """Leave-region-out. Nenhum municipio da regiao de teste aparece no treino."""

    def split(self, regioes: np.ndarray):
        for regiao in sorted(pd.unique(regioes)):
            teste = np.where(regioes == regiao)[0]
            treino = np.where(regioes != regiao)[0]
            if len(teste) and len(treino):
                yield treino, teste, {"regiao_teste": str(regiao)}


@dataclass
class BlocoEspacoTemporal:
    """Nem a regiao nem o periodo de teste aparecem no treino.

    O teste mais severo: para cada regiao deixada de fora, o treino usa somente
    os anos anteriores ao periodo de teste, nas demais regioes.
    """

    ano_inicial_teste: int
    horizonte: int = 1
    minimo_treino: int = 10

    def split(self, anos: np.ndarray, regioes: np.ndarray):
        unicos = np.array(sorted(np.unique(anos)))
        for regiao in sorted(pd.unique(regioes)):
            for alvo in unicos[unicos >= self.ano_inicial_teste]:
                anos_teste = [a for a in range(alvo, alvo + self.horizonte)
                              if a in unicos]
                anos_treino = unicos[unicos < alvo]
                if not anos_teste or len(anos_treino) < self.minimo_treino:
                    continue
                teste = np.where((regioes == regiao) & np.isin(anos, anos_teste))[0]
                treino = np.where((regioes != regiao)
                                  & np.isin(anos, anos_treino))[0]
                if len(teste) and len(treino):
                    yield treino, teste, {"regiao_teste": str(regiao),
                                          "ano_teste": int(alvo)}


def particao_aleatoria_ingenu(n: int, semente: int = 42, teste: float = 0.2):
    """Particao aleatoria por linha.

    Existe apenas para servir de comparacao: e a referencia contra a qual se mede
    o quanto a validacao ingenua superestima. Nao deve ser usada para reportar
    desempenho.
    """
    rng = np.random.default_rng(semente)
    indices = rng.permutation(n)
    corte = int(n * (1 - teste))
    return indices[:corte], indices[corte:]
