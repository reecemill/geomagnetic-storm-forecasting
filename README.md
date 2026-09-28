# Geomagnetic Storm Forecasting

## The problem
When a burst of solar wind hits Earth's magnetic field hard enough, it causes a geomagnetic storm. Storms drag satellites down (SpaceX lost about 40 new Starlink satellites to a fairly modest one in February 2022), disrupt GPS and radio, and in extreme cases threaten power grids. Satellite operators and grid operators plan around forecasts, and the standard one in the US is NOAA's: every day at 22:00 UTC, it publishes the probability of a storm on each of the next three days.

## The goal
Find out how good NOAA's storm probabilities are, and whether a model trained on the solar wind and past geomagnetic activity can do better.

## The data
- **Kp index** from GFZ Potsdam, the standard measure of geomagnetic activity: 0 to 9 every three hours. NOAA calls Kp 5- or more a storm (G1) and Kp 9 extreme (G5).
- **Solar wind** from NASA's OMNI dataset: hourly speed, density, and magnetic field, measured upstream of Earth, 1995–2026.
- **NOAA's forecasts** from its daily Report of Solar-Geophysical Activity, 1996–2025: the probability of a minor and of a major storm on each of the next three days.

All three are free with no account.

## Storms, briefly
About one day in eight has a storm, and only a handful of days in thirty years reach G5, including the May 2024 superstorm. Storm days rise and fall with the Sun's 11-year cycle. The solar wind arrives only 30 to 60 minutes after it's measured, so it says a lot about the next few hours but little about tomorrow. One slower signal does reach further: the Sun rotates every 27 days, and long-lived features on it bring storms back a rotation later.

## The bar to clear
Each forecast gives, at 21:00 UTC, the probability of a storm on each of the next three UT days. Scored on the test years 2022–2025 (the current solar maximum) by Brier skill, the improvement over simply knowing the past year's storm rate:

| Next-day forecast | Brier skill |
|---|---|
| NOAA, as published | 11.5% |
| Persistence (today's activity) | 8.5% |
| 27-day echo alone | 0.7% |

NOAA's skill fades with lead time, to about 3% three days ahead. Recalibrating NOAA's numbers onto the planetary Kp index, using earlier years, made them worse, so the published probabilities are the benchmark.

## Can a model beat it?
Four model families, each with its settings chosen on 2017–2021, retrained on 1996–2021, and tested once on 2022–2025 on the same days as NOAA's forecasts:

| Next-day forecast | Brier skill | Beats NOAA beyond chance? |
|---|---|---|
| NOAA, as published | 11.5% | |
| Logistic regression | 14.9% | no |
| Gradient boosting | 16.1% | no |
| Gradient boosting with NOAA's forecast as an input (best on 2017–2021) | 16.5% | yes |
| Recurrent network on 72 hours of hourly data | 19.4% | yes |

- **Next day, every model beat NOAA**, two by more than chance (95% paired-bootstrap intervals).
- **The recurrent network's win isn't a reliable one:** it was the worst of the four on 2017–2021, and two days ahead it was worse than NOAA by more than chance.
- **Two and three days ahead, nothing beats NOAA beyond chance.** NOAA's forecasters can see coronal mass ejections leave the Sun a day or more before they arrive; nothing in this data can.
- **Gradient boosting with NOAA's forecast as an input is at or ahead of NOAA at every lead time**, so the practical result is a correction on top of NOAA's forecast, not a replacement.
- **What the model leans on:** the latest Kp, then the 27-day echo. The echo is nearly worthless on its own but the second most useful input alongside everything else.

## Limits
- **Hindsight data.** The models learn and are tested on final, cleaned data (definitive Kp, quality-checked solar wind). NOAA forecasts in real time from preliminary Kp and live solar wind with gaps. That flatters the models, most of all the next day. A fair operational comparison needs archived real-time data.
- **Different targets.** NOAA forecasts a mid-latitude K index, not the planetary Kp scored here.
- **One solar maximum.** The test is four years of one active period.

## Notebooks
1. [`01_storm_history`](notebooks/01_storm_history.ipynb): how rare storms are, the solar cycle, the May 2024 storm, and the 27-day echo.
2. [`02_baselines`](notebooks/02_baselines.ipynb): climatology, persistence, the 27-day echo, and NOAA's forecast.
3. [`03_models`](notebooks/03_models.ipynb): logistic regression, gradient boosting, a recurrent network, and gradient boosting with NOAA's forecast, against NOAA (a few minutes to run).

Shared code is in [`src/storms.py`](src/storms.py) (loading the data), [`src/features.py`](src/features.py) (what's known at 21:00 UTC), and [`src/sequence.py`](src/sequence.py) (the recurrent network). Notebook 03 checks that changing anything after the issue time leaves its inputs unchanged.

## Setup
```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_data.py      # about 150 MB into data/
```
NOAA's reports for a year are archived once the year ends, so the comparison with NOAA runs through 2025.
