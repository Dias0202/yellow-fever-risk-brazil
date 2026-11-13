# Model card — season-ahead sylvatic yellow fever forecast

## Model details

| | |
|---|---|
| Task | Binary: will a municipality record a confirmed human case **or** a non-human primate epizootic in the next July–June surveillance season? |
| Unit | Municipality × season (5,570 municipalities) |
| Algorithm | Logistic regression, L2 penalty (C = 0.003), no class weights, Platt calibration fitted only on seasons before the one predicted |
| Inputs | 12 variables, one per mechanism (below), all computed from information available at the end of June |
| Code | `analysis/f6_04_modelo_final.py`, `analysis/f6_06_regularizacao.py`, `analysis/f6_05_previsao_operacional.py`; features in `src/yfvrisk/features/` |
| Output | `results/tables/f6_05_previsao_proxima_temporada.csv`: calibrated probability and two rankings (model, force of infection) for every municipality |

### Variables

| Variable | Mechanism |
|---|---|
| `forca_k50`, `forca_k200` | Force of infection: distance-decayed outbreak pressure from the previous seasons (50 and 200 km kernels) |
| `forca_epiz_k100` | Epizootic pressure, the sentinel signal |
| `dist_foco_mais_proximo` | Distance to the nearest focus in the previous season |
| `temporadas_desde_evento` | Refractoriness: recently hit areas are less invadable |
| `eventos_locais_10t` | Local history over ten seasons |
| `cobertura_florestal_t1` | Habitat of the enzootic cycle (Hansen) |
| `altitude_media`, `amplitude_altimetrica` | Altitudinal range of the vector; rugged terrain keeps remnant forest |
| `populacao_log_t1` | Exposure **and** notification effort (ambiguous; see limitations) |
| `tmax_temporada_normal`, `ppt_sazonalidade_normal` | Climate normals of the municipality |

Temporal causality: every propagation variable for season *t* uses only outcomes up
to *t*−1.

## Intended use

- Ranking municipalities for **surveillance prioritisation** ahead of a season, in the
  context of an ongoing epizootic wave.
- Use the model ranking for short lists (≈100 municipalities) and the force-of-infection
  ranking for wide sweeps (500+). Neither dominates the other.

**Out of scope:** individual clinical decisions, vaccine allocation by itself,
predicting the emergence of a new wave in an area with no nearby activity, and any
use in place of the official surveillance system.

## Evaluation

The model is evaluated prospectively with an expanding window over 21 seasons: train
on all seasons up to T−1, predict T, move forward. The variables are the same ones
the published list uses.

| Ranking | ROC-AUC | PR-AUC / prevalence | Recall @100 | @500 | @1,000 |
|---|---:|---:|---:|---:|---:|
| Logistic model | 0.734 | 7.28× | 0.108 | 0.360 | 0.515 |
| Force of infection (no learning) | 0.775 | 8.14× | 0.083 | 0.362 | 0.541 |
| Persistence baseline | — | 1.98× | — | — | — |

- **Calibration:** monotonic across deciles (Spearman +1.00). The lowest decile
  observes 0.026%. The top decile predicts 3.87% and observes 3.90%.
- **Heterogeneity:** 23×, 16× and 8.6× chance in 2018/19, 2019/20 and 2020/21, when the
  wave was advancing. 1.2× in 2016/17, when it arrived in the Atlantic Forest.
- **Arrival vs continuation**, split by whether a focus existed within 200 km in the
  previous season. With a nearby focus: ROC-AUC 0.727, PR-AUC 1.95× prevalence. Without
  one: 0.742 and 3.16×. Close to the wave, the model barely distinguishes which
  municipality will be hit.

Metrics: `results/metrics/f6_0*.json`. Why the validation is prospective and not a
random split: see the README section "Why not a random split".

## Training data

Ministry of Health human cases and epizootics, 1994–2025, aggregated by surveillance
season. Climate comes from TerraClimate, forest from Hansen GFC, relief from WorldClim
elevation, population from IBGE. Provenance and known gaps are in `data/SOURCES.yml`.
The base rate is 0.73% per municipality-season.

## Limitations and risks

- **No emergence forecasting.** The model says where an ongoing wave advances, not
  where the next one starts.
- **Population is the second-largest coefficient.** It is ambiguous between real
  exposure and notification effort: larger municipalities are investigated more,
  and therefore report more.
- **No primate variables.** *Alouatta* is the sentinel of the system. Host density
  enters only through forest cover.
- **Recent climate is unavailable at prediction time.** TerraClimate ends in calendar
  2024. Climate-anomaly triggers, the most plausible mechanism for a new wave, are
  excluded, and performance was re-measured without them.
- **Vaccination is not an input.** Recorded coverage is a consequence of outbreaks
  (a 4.5× jump in the outbreak year) and carries no generalisable signal once lagged.
- **Surveillance gaps.** No national epizootic records exist for 2004–2006 and
  2010–2013. These seasons are marked as unobserved, not as zero.
- **Aggregating to micro-regions does not help.** It was tested.
- The forecast is a research artefact. It has not been validated in operational use
  by the surveillance service.

## Licence

Code: MIT. Derived results: CC BY 4.0. Source data: original providers' terms.
