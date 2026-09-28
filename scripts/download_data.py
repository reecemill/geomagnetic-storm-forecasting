"""Download everything the notebooks use into data/. No accounts needed.

- NASA OMNI: hourly solar wind at Earth, one file per year, into data/omni/.
- GFZ Potsdam: the planetary Kp index every 3 hours since 1932, into data/kp.txt.
- NOAA SWPC: the daily Report of Solar-Geophysical Activity, which carries NOAA's
  storm probabilities for the next three days, one archive per year into data/noaa/.

    python scripts/download_data.py            # 1995 to this year
    python scripts/download_data.py 2020 2026
"""

import sys
import tarfile
from datetime import date
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OMNI = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
KP = "https://kp.gfz.de/app/files/Kp_ap_since_1932.txt"
NOAA = "ftp://ftp.swpc.noaa.gov/pub/warehouse/{year}/{year}_RSGA.tar.gz"


def fetch(url, path):
    with urlopen(url, timeout=300) as response:
        path.write_bytes(response.read())
    return path


def main(first=1995, last=date.today().year):
    first, last = int(first), int(last)
    for folder in ("omni", "noaa"):
        (DATA / folder).mkdir(parents=True, exist_ok=True)

    print(f"Kp: {fetch(KP, DATA / 'kp.txt').stat().st_size / 1e6:.1f} MB")
    for year in range(first, last + 1):
        omni = fetch(OMNI.format(year=year), DATA / "omni" / f"omni2_{year}.dat")
        archive = DATA / "noaa" / f"{year}_RSGA.tar.gz"
        try:
            fetch(NOAA.format(year=year), archive)
            with tarfile.open(archive) as tar:
                tar.extractall(DATA / "noaa", filter="data")
            archive.unlink()
            reports = len(list((DATA / "noaa").glob(f"{year}*/*RSGA*.txt")))
        except OSError as error:
            reports = f"none ({error})"
        print(f"{year}: OMNI {omni.stat().st_size / 1e6:.1f} MB, NOAA reports {reports}")


if __name__ == "__main__":
    main(*sys.argv[1:])
