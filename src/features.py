"""What a forecaster knows at 21:00 UTC on the issue day, as model inputs.

At 21:00 UTC the Kp periods up to 18:00-21:00 are in, and so are the solar wind
hours up to 20:00-21:00. NOAA issues its forecast an hour later, at 22:00 UTC.
"""

import numpy as np
import pandas as pd

from storms import STORM_KP

ISSUE_HOUR = 21
SEQUENCE_HOURS = 72
SEQUENCE_CHANNELS = ["bz", "by", "b", "speed", "density", "pressure", "e_field", "kp"]


def daily_storms(kp):
    """For each UT day: the highest Kp, and whether it reached storm level."""
    daily = kp["kp"].resample("D").max()
    return pd.DataFrame({"max_kp": daily, "storm": (daily >= STORM_KP).astype(float)})


def issue_features(kp, omni):
    """Features that don't depend on the lead time, one row per issue day."""
    days = daily_storms(kp)
    before_issue = kp[kp.index.hour < ISSUE_HOUR]["kp"].resample("D")
    f = pd.DataFrame(index=days.index)
    f["kp_today_max"] = before_issue.max()
    f["kp_today_mean"] = before_issue.mean()
    f["kp_last"] = kp["kp"].reindex(f.index + pd.Timedelta(hours=ISSUE_HOUR - 3)).to_numpy()
    for back in (1, 2, 3):
        f[f"kp_max_{back}d_ago"] = days["max_kp"].shift(back)
    storm_today = (f["kp_today_max"] >= STORM_KP).astype(float)
    f["storm_days_last_week"] = days["storm"].shift(1).rolling(6).sum() + storm_today
    f["storm_rate_last_year"] = days["storm"].shift(1).rolling(365, min_periods=300).mean()

    # Solar wind over the 24 hours before the issue time (hours starting 21:00 the day before to 20:00).
    wind = omni.copy()
    wind["southward"] = (-wind["bz"]).clip(lower=0)
    wind["bz_below_5"] = (wind["bz"] < -5).astype(float).where(wind["bz"].notna())
    # Each rolling window ends at the hour starting 20:00, the last one complete at 21:00.
    ends = f.index + pd.Timedelta(hours=ISSUE_HOUR - 1)
    rolled = {
        "speed_mean_24h": wind["speed"].rolling(24, min_periods=12).mean(),
        "speed_max_24h": wind["speed"].rolling(24, min_periods=12).max(),
        "speed_last_3h": wind["speed"].rolling(3, min_periods=1).mean(),
        "bz_min_24h": wind["bz"].rolling(24, min_periods=12).min(),
        "southward_mean_24h": wind["southward"].rolling(24, min_periods=12).mean(),
        "hours_bz_below_5": wind["bz_below_5"].rolling(24, min_periods=12).sum(),
        "southward_last_3h": wind["southward"].rolling(3, min_periods=1).mean(),
        "e_field_max_24h": wind["e_field"].rolling(24, min_periods=12).max(),
        "density_max_24h": wind["density"].rolling(24, min_periods=12).max(),
        "pressure_max_24h": wind["pressure"].rolling(24, min_periods=12).max(),
        "b_max_24h": wind["b"].rolling(24, min_periods=12).max(),
        "f107": wind["f107"],
        "sunspots_27d": wind["sunspots"].rolling(27 * 24, min_periods=24).mean(),
    }
    for name, series in rolled.items():
        f[name] = series.reindex(ends).to_numpy()
    # A jump in speed over the day often marks a shock or a fast stream arriving.
    f["speed_jump"] = f["speed_last_3h"] - wind["speed"].rolling(12, min_periods=6).mean().reindex(ends - pd.Timedelta(hours=12)).to_numpy()
    return f


def lead_features(issue, days, lead):
    """The features for a lead time: the issue-day features plus the 27-day echo of the target day."""
    target_day = issue.index + pd.Timedelta(days=lead)
    f = issue.copy()
    f["echo_27"] = days["max_kp"].reindex(target_day - pd.Timedelta(days=27)).to_numpy()
    f["echo_26_28_max"] = pd.concat([days["max_kp"].reindex(target_day - pd.Timedelta(days=d)).reset_index(drop=True)
                                     for d in (26, 27, 28)], axis=1).max(axis=1).to_numpy()
    f["echo_54"] = days["max_kp"].reindex(target_day - pd.Timedelta(days=54)).to_numpy()
    f["storm"] = days["storm"].reindex(target_day).to_numpy()
    f["target_day"] = target_day
    return f


def hourly_sequences(kp, omni, issue_days):
    """The last SEQUENCE_HOURS hours of solar wind and Kp before each issue time, shape (days, hours, channels).

    Kp is spread over the three hours of its period. Gaps are left as NaN for the model to handle.
    """
    hourly = omni.copy()
    hourly["kp"] = kp["kp"].reindex(hourly.index, method="ffill", limit=2)
    values = hourly[SEQUENCE_CHANNELS].to_numpy(dtype=np.float32)
    position = hourly.index.get_indexer(pd.DatetimeIndex(issue_days) + pd.Timedelta(hours=ISSUE_HOUR - 1))
    out = np.full((len(position), SEQUENCE_HOURS, len(SEQUENCE_CHANNELS)), np.nan, dtype=np.float32)
    for i, end in enumerate(position):
        if end >= SEQUENCE_HOURS - 1:
            out[i] = values[end - SEQUENCE_HOURS + 1:end + 1]
    return out
