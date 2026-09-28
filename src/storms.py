"""Load the solar wind, the Kp index, and NOAA's storm forecasts (see scripts/download_data.py).

Times are UTC. A geomagnetic storm, in NOAA's scale, is Kp of 5 or more (G1 and up).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def g_threshold(g):
    """The lowest Kp that counts as NOAA's G-scale level g. Kp comes in thirds (5-, 5o, 5+),
    and each level starts at its minus third: G1 at Kp 5- (4.67), up to G5 at Kp 9- (8.67)."""
    return g + 4 - 1 / 3 - 1e-6


STORM_KP = g_threshold(1)

# OMNI2 hourly columns used here: (1-based word in each line, name, fill value).
# See https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
OMNI_COLUMNS = [
    (9, "b", 999.9),             # field magnitude, nT
    (16, "by", 999.9),           # By, GSM, nT
    (17, "bz", 999.9),           # Bz, GSM, nT: southward (negative) Bz drives storms
    (23, "temperature", 9999999.0),
    (24, "density", 999.9),      # protons per cm^3
    (25, "speed", 9999.0),       # km/s
    (29, "pressure", 99.99),     # flow pressure, nPa
    (36, "e_field", 999.99),     # -V x Bz, mV/m
    (40, "sunspots", 999),
    (41, "dst", 99999),          # nT
    (51, "f107", 999.9),         # solar radio flux
]


def load_omni(years=None):
    """Hourly solar wind and indices, indexed by the start of each hour."""
    files = sorted((DATA / "omni").glob("omni2_*.dat"))
    if years is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) in set(years)]
    raw = pd.concat([pd.read_csv(f, sep=r"\s+", header=None) for f in files], ignore_index=True)
    time = (pd.to_datetime(raw[0].astype(str), format="%Y")
            + pd.to_timedelta(raw[1] - 1, unit="D") + pd.to_timedelta(raw[2], unit="h"))
    out = pd.DataFrame(index=pd.DatetimeIndex(time, name="time"))
    for word, name, fill in OMNI_COLUMNS:
        values = raw[word - 1].to_numpy(dtype=float)
        out[name] = np.where(values >= fill, np.nan, values)
    return out


def load_kp():
    """The planetary Kp index every 3 hours (0 to 9 in thirds), with whether it's definitive."""
    rows = [line.split() for line in (DATA / "kp.txt").read_text().splitlines() if not line.startswith("#")]
    kp = pd.DataFrame(rows, columns=["y", "m", "d", "hour", "hour_mid", "days", "days_mid", "kp", "ap", "definitive"])
    time = pd.to_datetime(kp[["y", "m", "d"]].astype(int).set_axis(["year", "month", "day"], axis=1))
    time += pd.to_timedelta(kp["hour"].astype(float), unit="h")
    out = pd.DataFrame({"kp": kp["kp"].astype(float).to_numpy(), "definitive": (kp["definitive"] == "1").to_numpy()},
                       index=pd.DatetimeIndex(time, name="time"))
    out.loc[out["kp"] < 0, "kp"] = np.nan
    return out


def _probabilities(text, label):
    """The three day-1/2/3 percentages after a label, e.g. 'Minor Storm 10/01/01' -> [10, 1, 1]."""
    m = re.search(label + r"\s+(\d+)/(\d+)/(\d+)", text, re.IGNORECASE)
    return [int(g) for g in m.groups()] if m else [np.nan] * 3


def load_noaa_forecasts():
    """NOAA's middle-latitude geomagnetic activity probabilities for the next three UT days,
    one row per report and forecast day (issued daily at 22:00 UTC)."""
    rows = []
    for path in sorted((DATA / "noaa").glob("*/*RSGA*.txt")):
        text = path.read_text(errors="replace")
        issued = re.search(r":Issued:\s+(\d{4} \w{3} \d{1,2})", text)
        section = re.search(r"VI\.\s+Geomagnetic Activity Probabilities(.*?)B\.\s+High Latitudes", text,
                            re.IGNORECASE | re.DOTALL)
        if not issued or not section:
            continue
        issued = pd.to_datetime(issued.group(1), format="%Y %b %d")
        mid = section.group(1)
        active, minor, major = (_probabilities(mid, label) for label in ("Active", "Minor Storm", "Major-severe storm"))
        for day in range(3):
            rows.append({"issued": issued, "lead_days": day + 1, "date": issued + pd.Timedelta(days=day + 1),
                         "p_active": active[day] / 100, "p_minor": minor[day] / 100, "p_major": major[day] / 100})
    # A few days have two reports (an amended one); keep the one filed last.
    return pd.DataFrame(rows).drop_duplicates(["issued", "lead_days"], keep="last").reset_index(drop=True)


def daily_max_kp(kp):
    """The largest Kp in each UT day."""
    return kp["kp"].resample("D").max()
