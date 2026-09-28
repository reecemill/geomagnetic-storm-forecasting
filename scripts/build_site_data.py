"""Build the data file for the project's web page from the notebooks' saved results.

Run the three notebooks first: each saves the numbers behind its charts to results/.
The only data read here is the Kp index, for each test day's actual peak Kp.

    python scripts/build_site_data.py ../portfolio-site/storms/data.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from storms import load_kp


def main(out):
    years = pd.read_csv(RESULTS / "storm_days_per_year.csv")
    baselines = pd.read_csv(RESULTS / "baseline_test_scores.csv")
    models = pd.read_csv(RESULTS / "test_scores.csv")
    choice = pd.read_csv(RESULTS / "validation_choice.csv")
    daily = pd.read_csv(RESULTS / "next_day_test_forecasts.csv", parse_dates=[0], index_col=0)
    winner = [c for c in daily.columns if c not in ("storm", "NOAA, as published")][0]

    # Each row is an issue day; the forecast is for the next UT day.
    peak = load_kp()["kp"].resample("D").max()
    target = daily.index + pd.Timedelta(days=1)
    days = pd.DataFrame({"date": target.strftime("%Y-%m-%d"), "max_kp": peak.reindex(target).round(2).to_numpy(),
                         "storm": daily["storm"].astype(int).to_numpy(),
                         "noaa": daily["NOAA, as published"].round(3).to_numpy(), "model": daily[winner].round(3).to_numpy()})

    data = {
        "stormYears": years.round(1).to_dict(orient="records"),
        "baselines": baselines.round(4).to_dict(orient="records"),
        # NOAA's own rows have no difference from NOAA; they become null (JSON has no NaN).
        "models": models.round(4).astype(object).where(models.notna(), None).to_dict(orient="records"),
        "validationChoice": choice.round(4).to_dict(orient="records"),
        "winner": winner,
        "days": days.to_dict(orient="records"),
    }
    Path(out).write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    print(f"wrote {Path(out).stat().st_size / 1000:.0f} kB to {out}: {len(days)} test days, winner {winner}")


if __name__ == "__main__":
    main(*sys.argv[1:])
