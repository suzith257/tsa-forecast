# TSA Travel Demand Forecasting

Daily air-travel demand forecasting on real public data, with three models built and
compared from the ground up — and every forecast scored against actuals the model never saw.

**Live site:** https://suzith257.github.io/tsa-forecast//

---

## What this is

The U.S. Transportation Security Administration publishes the number of passengers screened
at its checkpoints every day — one of the cleanest public proxies for travel demand: daily
resolution, strong weekly rhythm, sharp holiday structure, seven years of history.

This project forecasts daily demand and checks how close each model came to reality. It is
built as a portfolio piece for workforce-management / demand-forecasting work, so the emphasis
is on **honest validation** rather than a single headline number.

## Method

```
Train 2022–2024  →  forecast 2025  →  score against real 2025
                 →  refit through 2025  →  forecast 2026
```

The year being predicted is never part of the training data. COVID (2020–21) is kept in the
data for context but excluded from training as a structural break.

## The three models

| Model | What it is | Role |
|-------|-----------|------|
| **Manual decomposition** | Hand-built `level × annual × day-of-week × holiday × bridge`, no forecasting library | The model to understand |
| **Prophet** | Facebook's library with full US federal-holiday calendar | The benchmark |
| **SARIMA** | `SARIMA(1,1,1)(1,1,1,7)` on log passengers, weekly seasonality only | A deliberately weak classical baseline |

The manual model uses six Fourier harmonics of day-of-year for smooth annual seasonality,
geometric-mean day-of-week and holiday factors, and "bridge" factors for the travel surge on
the days *around* a holiday (the holiday itself is usually a low-travel day).

## Results

**2025 validation (trained on 2022–2024):**

| Model | MAPE | Bias |
|-------|------|------|
| Manual | 5.27% | +1.08% |
| Prophet | 5.34% | +2.94% |
| SARIMA | 16.19% | +14.84% |

**2026 (locked vs. monthly-refreshed forecast):**

| Model | Locked MAPE | Refreshed MAPE |
|-------|-------------|----------------|
| Prophet | 5.54% | **4.38%** |
| Manual | 5.82% | **4.64%** |
| SARIMA | 14.50% | 23.68% (destabilises under refit) |

The hand-built model matches — and on 2025 slightly beats — Prophet, while staying fully
interpretable. Monthly refresh improves the holiday-aware models by over a percentage point;
SARIMA gets worse, which is itself the finding that model choice governs refresh cadence.

A day-level MAPE around 5% is at the contact-center industry standard for demand forecasting.

## Project layout

```
tsa-forecast/
├── generate_site_assets.py   # scrape → model → validate → charts
├── docs/                     # the published site (GitHub Pages)
│   ├── index.html
│   └── assets/               # generated charts + metrics CSVs
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python generate_site_assets.py
```

The script scrapes `tsa.gov/travel/passenger-volumes` live (with a cached fallback), trains
all three models, runs the walk-forward validation and the locked monthly-refresh study, and
writes every chart and metrics file into `docs/assets/`. Run it any day and it pulls the
latest published figures, scoring the frozen 2026 forecast against all actuals available
to date.

## Data

TSA Checkpoint Travel Numbers — https://www.tsa.gov/travel/passenger-volumes
Public domain. Figures are revised by TSA as late additions arrive, so results reflect the
data as retrieved.
