"""Download hourly historical weather for the study period (Open-Meteo archive, no API key).

    python scripts/fetch_weather.py

Saved to data/external/weather.parquet, in local Tel Aviv time like the MDS timestamps.
Open-Meteo data is licensed CC BY 4.0: https://open-meteo.com/
"""

import json
import urllib.parse
import urllib.request

import pandas as pd

from oversupply.config import REPO_ROOT, Config

VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"]


def main() -> None:
    cfg = Config.load()
    last_day = pd.Timestamp(cfg.study.end) - pd.Timedelta(days=1)
    query = urllib.parse.urlencode({
        "latitude": cfg.weather.latitude,
        "longitude": cfg.weather.longitude,
        "start_date": cfg.study.start,
        "end_date": last_day.strftime("%Y-%m-%d"),
        "hourly": ",".join(VARIABLES),
        "timezone": "Asia/Jerusalem",
    })
    request = urllib.request.Request(f"https://archive-api.open-meteo.com/v1/archive?{query}",
                                     headers={"User-Agent": "sdes-oversupply"})
    with urllib.request.urlopen(request, timeout=120) as response:
        hourly = json.load(response)["hourly"]
    out = REPO_ROOT / "data" / "external"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(hourly).assign(time=lambda d: pd.to_datetime(d["time"]))
    df.to_parquet(out / "weather.parquet", index=False)
    print(df.describe().round(1).to_string())


if __name__ == "__main__":
    main()
