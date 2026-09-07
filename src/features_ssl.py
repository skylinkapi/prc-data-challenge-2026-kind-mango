"""Semi-supervised operational-drift features from ranking DEP rows.

The ranking file contains unlabeled 2026 DEP rows with valid sched_delay
values (MVT_TIME - SCHED_TIME, both fields present). Aggregating per
(operator, airport) captures how operators are currently running in 2026
vs their 2025 training baseline. The model can then correct for drift.

Adds:
  ssl_op_apt_ranking_median_sched_delay
  ssl_op_apt_ranking_mean_sched_delay
  ssl_op_apt_ranking_n_flights
  ssl_op_apt_drift_median  = ranking_median - training_median
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKING = os.path.join(ROOT, "submission", "ranking.parquet")

SSL_NUM_COLS = [
    "ssl_op_apt_ranking_median_sched_delay",
    "ssl_op_apt_ranking_mean_sched_delay",
    "ssl_op_apt_ranking_n_flights",
    "ssl_op_apt_drift_median",
]


def _compute_lookup(ranking: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """(op, apt) -> ranking stats + drift vs training."""
    # Ranking stats
    r = ranking.groupby(["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"])["sched_delay"] \
        .agg(["median", "mean", "size"]).reset_index()
    r = r.rename(columns={"median": "ssl_op_apt_ranking_median_sched_delay",
                          "mean": "ssl_op_apt_ranking_mean_sched_delay",
                          "size": "ssl_op_apt_ranking_n_flights"})

    # Training stats for the same keys
    t = training.groupby(["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"])["sched_delay"] \
        .median().rename("_train_median").reset_index()

    m = r.merge(t, on=["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], how="left")
    m["ssl_op_apt_drift_median"] = m["ssl_op_apt_ranking_median_sched_delay"] - m["_train_median"]
    return m[["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", *SSL_NUM_COLS]]


def add_ssl(dep: pd.DataFrame, training_dep: pd.DataFrame) -> pd.DataFrame:
    """dep needs: ADEP_mvt, AIRCRAFT_OPERATOR_flt.
    training_dep must have: ADEP_mvt, AIRCRAFT_OPERATOR_flt, sched_delay.
    Uses the ranking file at submission/ranking.parquet for the SSL stats.
    """
    rank = pd.read_parquet(RANKING, columns=["ADEP_mvt", "AIRCRAFT_OPERATOR_flt",
                                             "PHASE_mvt", "MVT_TIME_UTC_mvt",
                                             "SCHED_TIME_UTC_mvt"])
    rank = rank[rank["PHASE_mvt"] == "DEP"].copy()
    rank["mvt_ts"] = pd.to_datetime(rank["MVT_TIME_UTC_mvt"], errors="coerce")
    rank["sched_ts"] = pd.to_datetime(rank["SCHED_TIME_UTC_mvt"], errors="coerce")
    rank["sched_delay"] = (rank["mvt_ts"] - rank["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    rank = rank.dropna(subset=["sched_delay", "ADEP_mvt", "AIRCRAFT_OPERATOR_flt"])

    lookup = _compute_lookup(rank, training_dep)
    return dep.merge(lookup, on=["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], how="left")


if __name__ == "__main__":
    import glob
    from features_weather import TARGET_ICAOS
    frames = [pd.read_parquet(f, columns=["ADEP_mvt", "PHASE_mvt", "AIRCRAFT_OPERATOR_flt",
                                          "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
              for f in sorted(glob.glob(os.path.join(ROOT, "training", "*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    m = m[(m["PHASE_mvt"] == "DEP") & m["ADEP_mvt"].isin(TARGET_ICAOS)].copy()
    m["mvt_ts"] = pd.to_datetime(m["MVT_TIME_UTC_mvt"], errors="coerce")
    m["sched_ts"] = pd.to_datetime(m["SCHED_TIME_UTC_mvt"], errors="coerce")
    m["sched_delay"] = (m["mvt_ts"] - m["sched_ts"]).dt.total_seconds().clip(-1800, 3600)
    m = m.dropna(subset=["sched_delay", "AIRCRAFT_OPERATOR_flt"])

    sample = m.head(50000)
    out = add_ssl(sample, m)
    print(f"Rows: {len(out):,}")
    print("Coverage:")
    for c in SSL_NUM_COLS:
        cov = out[c].notna().mean() * 100
        print(f"  {c:45s} coverage {cov:5.1f}%")
    print("\nDrift median distribution:")
    print(out["ssl_op_apt_drift_median"].describe(percentiles=[.05, .5, .95]).round(1).to_string())
    print("\nTop drift shifts per airport:")
    print(out.groupby("ADEP_mvt")["ssl_op_apt_drift_median"].agg(["mean", "median"]).round(1).to_string())
