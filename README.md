# Sylvatic yellow fever risk in Brazil

Season-ahead forecasting of sylvatic yellow fever in Brazil's 5,570 municipalities,
plus the ecological, vaccination and climate-scenario analyses that support it.

> **Live dashboard:** https://dias0202.github.io/yellow-fever-risk-brazil/ ·
> Portuguese version: [README.pt-BR.md](README.pt-BR.md) · Model card: [MODEL_CARD.md](MODEL_CARD.md)

## The problem

Sylvatic yellow fever is not a static property of habitat. The virus circulates
between *Haemagogus*/*Sabethes* mosquitoes and non-human primates and advances as an
**epizootic wave through contiguous forest**. A municipality is hit because the wave
got close, not because its climate changed. After the wave passes, the local howler
monkey population is decimated and the survivors are immune, so the area stays
refractory for years.

Three things follow, and the project is built around them:

1. **Contagion and persistence carry most of the signal.** For a municipality with an
   outbreak within 50 km in the previous season, the probability of an event in the
   next season is **16.8×** the base rate. For one with an outbreak of its own, it is
   **19×**.
2. **The unit of time is the July–June surveillance season, not the calendar year.**
   97.6% of human cases occur between December and May. A calendar year splits each
   wave in half. The Ministry of Health records carry the `MONITORAMENTO` season field
   for 100% of records.
3. **Validation has to match use.** The outcome is rare (0.74% of municipality-years).
   Adjacent years of the same municipality are near-duplicates, so a random row split
   measures interpolation, not forecasting.

## Two models, two questions

| Model | Question | Validation |
|---|---|---|
| **Season-ahead forecast**: L2-regularised logistic regression on 12 variables, no class weights, Platt calibration | Where will the ongoing wave appear in the season that starts in July? | Expanding window by season: train on everything up to T−1, predict T, move forward |
| **Environmental suitability**: random forest on 32 climate, landscape, immunity and demographic variables | Which ecological drivers hold out of sample, and how does projected climate shift risk? | Temporal, spatial (state left out) and spatio-temporal (macro-region *and* year left out) blocks |

### Why not a random split

Same random forest, same panel, same environmental variables. Only the partition
changes (`dashboard/dados/validacao.json`):

| Partition | ROC-AUC | PR-AUC | × chance | Municipalities investigated per event |
|---|---:|---:|---:|---:|
| Random rows | 0.965 | 0.421 | 57.4× | 2.9 |
| Spatial (unseen state) | 0.864 | 0.102 | 13.8× | 10.4 |
| Temporal (unseen year) | 0.746 | 0.033 | 3.4× | 23.1 |
| Spatio-temporal (unseen macro-region and year) | 0.524 [0.505, 0.542] | 0.024 | 2.4× | 20.5 |

ROC-AUC drops by 0.44, while PR-AUC over chance drops by a factor of 24. Lagged
climate shows the mechanism most clearly. Last year's climate is almost identical
between neighbouring rows of the same municipality, so adding it raises PR-AUC by
53% under a random split but only by 15% under a temporal one. **Every result below
comes from validation where the evaluated period never takes part in fitting.**

## Results

### Season-ahead forecast

The model is evaluated prospectively over 21 seasons, using the same variables that
produce the published list (`results/metrics/f6_05_previsao_operacional.json`):

| Ranking | ROC-AUC | PR-AUC / chance | Recall @100 | Recall @500 | Recall @1,000 |
|---|---:|---:|---:|---:|---:|
| Final logistic model | 0.734 | **7.28×** | **0.108** | 0.360 | 0.515 |
| Raw force of infection (no learning) | 0.775 | 8.14× | 0.083 | 0.362 | 0.541 |
| Local persistence baseline | — | 1.98× | — | — | — |

- **Neither ranking dominates.** The learned model is better at the top of the list:
  at 100 municipalities it catches about 30% more outbreaks. The distance-weighted
  force of infection is better in the tail. The output table carries both rankings,
  with the recommendation recorded in the file: use the model for short lists
  (≈100) and the force of infection for wide sweeps (500+). Both clearly beat local
  persistence.
- **Decision curves.** The season-level models have a range of thresholds with
  positive net benefit: 19 to 28, depending on the component (`results/metrics/f6_0*`).
- **Calibration by decile** is monotonic, with Spearman correlation across deciles of
  +1.00. The lowest decile has an observed rate of 0.026%. The top decile predicts
  3.87% and observes 3.90%.
- **Every coefficient has the sign epidemiology predicts.** Distance to the nearest
  focus is negative. Force of infection, forest cover and altitude are positive. Seasons
  since the last event are positive, which is refractoriness.
- **Why it is parsimonious.** With ten ways of measuring "how close the wave was",
  near-collinear predictors cancelled out. The lowest predicted-risk decile then had
  an observed rate of 1.90%, four times the mean. Keeping one representative per
  mechanism, with strong L2 regularisation, removes the cancellation at the source.
- **Why there are no class weights.** Re-weighting positives by ~140× improves
  calibration but destroys ranking, and ranking is what PR-AUC and a priority list
  measure.

**Declared limitation.** The model forecasts the *advance of an ongoing wave* and does
**not** forecast the *emergence of a new one*. In seasons where the wave was advancing
(2018/19, 2019/20, 2020/21) it reaches 23×, 16× and 8.6× chance. In 2016/17, the
season the wave reached the Atlantic Forest, it drops to **1.2×**. Aggregating to
micro-regions was tested and does not help: prevalence rises, but the ratio over it
falls. The forecast for the 2025/26 season, for all 5,570 municipalities, is in
`results/tables/f6_05_previsao_proxima_temporada.csv`.

### Ecological drivers (environmental model)

- In the training set, Gini importance (MDI) almost reproduces permutation importance
  (ρ = +0.81 and +0.91 for the 2014 and 2019 cut-offs). On unseen data it has no
  statistical relation to it (ρ = +0.10 and +0.30; p = 0.57 and 0.09).
- Only four of 32 variables have a permutation-importance 95% CI above zero in both
  temporal cut-offs: `ppt_sazonalidade`, `populacao_log`, `tmin_medio` and
  `ppt_total_lag2`.
- Under the spatio-temporal block, the **South** is the only macro-region where the
  model transfers without local history: ROC-AUC 0.737 [0.697, 0.777] and PR-AUC
  8.0× chance. Every other region stays between 0.45 and 0.56.

### Vaccination

The recorded vaccination coverage is a **consequence of the outbreak**, not a cause
of protection. In an event study, double-centred by year and by municipality, coverage
stays at the municipality's normal level until the year before the outbreak. It
**jumps 4.5× in the outbreak year** and stays high for three years: this is the
reactive vaccination campaign the outbreak triggers.

| Immunity lag | Within-municipality ρ with the event |
|---|---|
| 0 years | +0.082 [+0.073, +0.091] |
| 1 year | +0.017 |
| 2 years | −0.005 (p = 0.35) |

The modelling variable is therefore immunity lagged by two years. In the ablation,
contemporaneous immunity added +0.0199 ROC-AUC [+0.0070, +0.0320] and no PR-AUC,
which is the signature of leakage. With the lag, immunity carries no generalisable
signal, so **no vaccination scenario is reported**.

### Climate scenarios (CMIP6, 2021–2040)

There are five GCMs, chosen to span CMIP6 equilibrium climate sensitivity
(INM-CM5-0 ≈1.9 K to UKESM1-0-LL ≈5.3 K), and four SSPs. The delta method is applied
to each observed year. Uncertainty comes from 30 municipality-level bootstrap refits.

| Contrast | Mean | 95% CI |
|---|---:|---|
| Future − baseline | +0.00083 | [+0.00040, +0.00130] |
| SSP5-8.5 − SSP1-2.6 | +0.00010 | [−0.00006, +0.00023] |

- The climate signal survives, but **the separation between SSPs does not**.
- The choice of GCM moves the result far more than the choice of SSP (dispersion
  across GCMs 0.46 vs across SSPs 0.10, in `dashboard/dados/clima.json`). UKESM1-0-LL under the
  mildest scenario (+17.0%) projects a larger increase than any other GCM under the
  most severe one (at most +9.2%).
- In 83% of municipalities, the dispersion across scenarios is smaller than the
  dispersion across model refits. For that reason, **no municipal scenario map is
  published**.

## Data

Ten public sources, each with provenance declared in [`data/SOURCES.yml`](data/SOURCES.yml).

| Source | Content | Acquisition |
|---|---|---|
| Ministry of Health (CGARB/DEDT/SVSA) | Confirmed human cases and non-human primate epizootics, 1994–2025 | Manual export |
| TerraClimate (Abatzoglou et al., 2018) | Monthly climate, 1/24° | `ingest/baixar_terraclimate.py` |
| WorldClim 2.1 / CMIP6 | Baseline and 2021–2040 projections, 5 GCMs × 4 SSPs; elevation | `ingest/baixar_worldclim.py` |
| Hansen GFC-2024-v1.12 | Tree cover 2000 and annual loss, 30 m | `ingest/baixar_hansen.py` |
| MapBiomas-derived land use (Abdalla & Augusto, 2022) | Land-use fractions, 1985–2020 | Manual export |
| IBGE | Population estimates, 2022 municipal mesh, territorial hierarchy (API) | Manual / API |
| PNI / DATASUS | Yellow fever vaccination coverage, 1994–2022 | Manual export |

## Reproducibility

**What is in the repository:** all code, the tests, every metric JSON and result table
used in this README and in the dashboard, and the dashboard data.

**What is not:** raw inputs and the processed panel (`data/raw/`, `data/processed/`).
Several sources are manual exports (TabNet, IBGE spreadsheets, Ministry of Health
files, MapBiomas derivative) that cannot be fetched automatically, and the climate
and forest rasters total close to 200 GB. Tests that need the built panel **skip with an explicit
reason** when it is absent. The scenario-module algebra and the variable-group
invariants always run.

Expected layout of `data/raw/` for a full rebuild:

```
data/raw/
  fa_casoshumanos_1994-2025.csv        Ministry of Health, human cases
  fa_epizpnh_1994-2025.csv             Ministry of Health, epizootics
  vacin.csv                            PNI / TabNet coverage
  KrempserData/01-stats-1985--2020-land_use_cover-BR.csv
  População ibge/<year>.xls[x]         IBGE estimates, one file per year
  BR_Municipios_2022/  BR_UF_2024/     IBGE meshes
  TerraClimate_raw/  terraclimate/     local copy / files fetched by the downloader
  Hansen_2024_BR/    hansen/           local copy / 10N band fetched by the downloader
  worldclim/                           fetched by baixar_worldclim
```

### Running

Python 3.11. Dependencies are pinned in `src/requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt
export PYTHONPATH=src

# panel (annual)
python -m yfvrisk.ingest.municipios
python -m yfvrisk.ingest.epi
python -m yfvrisk.ingest.vacinacao
python -m yfvrisk.ingest.populacao
python -m yfvrisk.ingest.clima
python -m yfvrisk.ingest.floresta
python -m yfvrisk.ingest.uso_solo
python -m yfvrisk.features.painel

# panel (season) and propagation features
python -m yfvrisk.ingest.temporada
python -m yfvrisk.ingest.relevo
python -m yfvrisk.ingest.regionalizacao
python -m yfvrisk.features.espacial
python -m yfvrisk.features.epidemia
python -m yfvrisk.features.painel_temporada

# analyses (each writes results/metrics/<name>.json)
python analysis/f3_01_comparar_validacao.py
python analysis/f3_02_calibracao_e_bloco_espaco_temporal.py
python analysis/f4_01_explicabilidade.py 2014 && python analysis/f4_01_explicabilidade.py 2019
python analysis/f5_00_diagnostico_vacinacao.py
python analysis/f5_01_ablacao_vacinacao.py
python analysis/f6_01_previsao.py
python analysis/f6_02_ajuste.py
python analysis/f6_03_escala.py
python analysis/f6_04_modelo_final.py
python analysis/f6_06_regularizacao.py
python analysis/f6_05_previsao_operacional.py

# climate scenarios and dashboard data
python -m yfvrisk.scenarios.clima_futuro
python -m yfvrisk.scenarios.projetar
python -m yfvrisk.viz.exportar_painel

# checks
ruff check .
python -m pytest
```

Dashboard: `cd dashboard && python -m http.server 8848`. ES modules need an HTTP
server; `file://` does not work.

CI (`.github/workflows/ci.yml`) runs lint, an import of every module and the tests
on every push. A monthly workflow (`atualizar-dados.yml`) rebuilds the panel and
opens a pull request for human review. It is never merged automatically, and the
scheduled run is gated on the repository variable `PAINEL_DADOS_DISPONIVEIS`
because it needs the manual inputs.

## Repository layout

```
src/yfvrisk/
  ingest/      one module per source: download, integrity checks, municipality-year tables
  features/    annual panel, season panel, spatial graph, propagation features, scale aggregation
  modeling/    block validation schemes, prospective validation, metrics (discrimination, calibration, decision)
  explain/     out-of-sample permutation importance, SHAP, ALE
  scenarios/   CMIP6 delta-method climate and risk projection with model-noise floor
  viz/         export of dashboard data
src/tests/     panel invariants and scenario-module tests
analysis/      numbered analysis scripts (f3 validation, f4 explainability, f5 vaccination, f6 forecast)
results/       metrics (JSON) and tables (CSV), generated
dashboard/     static site (7 pages, hand-written SVG, no chart library)
data/          SOURCES.yml; raw and processed data are not versioned
```

Code identifiers, comments and the dashboard are in Portuguese. This README and the
model card are in English.

## Licence

- **Code:** MIT, see [LICENSE](LICENSE).
- **Derived results** (`results/`, `dashboard/dados/`): CC BY 4.0, see
  [LICENSE-DATA.md](LICENSE-DATA.md).
- **Source data** keep their original terms. Cite the original providers listed in
  `data/SOURCES.yml`.

## Citation

See [CITATION.cff](CITATION.cff). Developed at the Laboratório de Vírus, Universidade
Federal de Minas Gerais, with funding from FAPEMIG.
