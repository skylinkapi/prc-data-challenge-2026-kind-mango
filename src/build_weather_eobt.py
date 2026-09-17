"""L7: METAR joined at EOBT_1 time (in addition to the deployed MVT_TIME join).

The deployed features_weather.py reads METAR nearest to `mvt_ts`, which is the
end of the taxi. De-icing and low visibility act at pushback; EOBT_1
approximates that within ~20 min. Emits four EOBT-time columns keyed on
MVT_ID_mvt: tmpc_eobt, vis_km_eobt, wind_kt_eobt, deicing_gate_eobt.
"""
import logging
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_train.parquet")
RANK_FRAME = os.path.join(ROOT, "models", "v2", "cache", "frame_rank.parquet")
METAR_DIR = os.path.join(ROOT, "external", "metar")
OUT_TRAIN = os.path.join(ROOT, "models", "weather_eobt_train.parquet")
OUT_RANK = os.path.join(ROOT, "models", "weather_eobt_rank.parquet")
TOL = pd.Timedelta(minutes=45)
log = logging.getLogger(__name__)


def load_metar(icaos: list[str]) -> pd.DataFrame:
    frames = []
    for icao in icaos:
        p = os.path.join(METAR_DIR, f"{icao}.csv")
        if not os.path.exists(p):
            continue
        m = pd.read_csv(p, low_memory=False, na_values=["M", ""])
        m["ts"] = pd.to_datetime(m["valid"], utc=True, errors="coerce")
        for c in ("tmpf", "sknt", "vsby", "p01i"):
            m[c] = pd.to_numeric(m[c], errors="coerce")
        m["tmpc"] = (m["tmpf"] - 32) * 5 / 9
        m["vis_km"] = m["vsby"] * 1.609344
        m["wind_kt"] = m["sknt"]
        m["wx_precip"] = ((m["p01i"] > 0) | m["wxcodes"].fillna("").str.contains("RA|SN|IC|FZ|GR|GS", regex=True, na=False)).astype(float)
        m["ADEP_mvt"] = icao
        frames.append(m[["ADEP_mvt", "ts", "tmpc", "vis_km", "wind_kt", "wx_precip"]])
    return pd.concat(frames, ignore_index=True).sort_values(["ADEP_mvt", "ts"]) if frames else pd.DataFrame()


def build(path: str, out: str) -> None:
    if os.path.exists(out):
        log.info("cache hit: %s", out)
        return
    log.info("read %s", path)
    frame = pd.read_parquet(path, columns=["MVT_ID_mvt", "ADEP_mvt", "EOBT_1_flt"])
    frame["eobt_ts"] = pd.to_datetime(frame["EOBT_1_flt"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    icaos = sorted(frame["ADEP_mvt"].dropna().unique().tolist())
    metar = load_metar(icaos)
    if metar.empty:
        log.warning("no METAR loaded")
        return
    left = frame[["MVT_ID_mvt", "ADEP_mvt", "eobt_ts"]].dropna(subset=["eobt_ts"]).sort_values(["ADEP_mvt", "eobt_ts"])
    out_rows = []
    for a, g in left.groupby("ADEP_mvt", sort=False):
        r = metar[metar["ADEP_mvt"] == a][["ts", "tmpc", "vis_km", "wind_kt", "wx_precip"]].sort_values("ts")
        if r.empty:
            continue
        m = pd.merge_asof(g.sort_values("eobt_ts"), r,
                          left_on="eobt_ts", right_on="ts",
                          direction="nearest", tolerance=TOL)
        out_rows.append(m[["MVT_ID_mvt", "tmpc", "vis_km", "wind_kt", "wx_precip"]])
    if not out_rows:
        log.warning("no rows joined")
        return
    w = pd.concat(out_rows, ignore_index=True)
    w["deicing_gate_eobt"] = (w["wx_precip"] * (w["tmpc"] < 3)).astype(float)
    w = w.rename(columns={"tmpc": "tmpc_eobt", "vis_km": "vis_km_eobt",
                          "wind_kt": "wind_kt_eobt", "wx_precip": "wx_precip_eobt"})
    out_df = frame[["MVT_ID_mvt"]].merge(w, on="MVT_ID_mvt", how="left")
    cov = out_df["tmpc_eobt"].notna().mean()
    log.info("weather@EOBT coverage %.3f", cov)
    out_df.to_parquet(out)
    log.info("wrote %s (%d rows)", out, len(out_df))


def main() -> None:
    build(FRAME, OUT_TRAIN)
    build(RANK_FRAME, OUT_RANK)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
