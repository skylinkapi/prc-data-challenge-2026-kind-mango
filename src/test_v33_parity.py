"""H5: parity test. The v33 configuration of predict_v30.main must rebuild
submission/kind-mango_v33.parquet to 0.0 s on all 344,841 rows. Run before
every upload. The run also writes the served-frame dump that the coverage
monitor (H4) reads.

Usage: python src/test_v33_parity.py
Exit 1 when the maximum absolute difference is not 0.0.
"""
import logging
import os

import numpy as np
import pandas as pd

from predict_v30 import ROOT, main
from predict_v33 import R_NORM_FILES

REF = os.path.join(ROOT, "submission", "kind-mango_v33.parquet")
TMP = "_parity_v33.parquet"
DUMP = os.path.join(ROOT, "models", "v33_rank_dump.parquet")
log = logging.getLogger(__name__)


def run_parity() -> float:
    """Rebuild v33 and return the maximum absolute difference in seconds."""
    main(TMP, r_norm_files=R_NORM_FILES, dump_features=DUMP)
    new = pd.read_parquet(os.path.join(ROOT, "submission", TMP))
    ref = pd.read_parquet(REF)
    m = ref.merge(new, on="MVT_ID_mvt", suffixes=("_ref", "_new"), validate="1:1")
    diff = (m["TAXITIME_SEC_mvt_ref"] - m["TAXITIME_SEC_mvt_new"]).abs().max()
    os.remove(os.path.join(ROOT, "submission", TMP))
    return float(diff)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    d = run_parity()
    log.info("v33 parity: max |diff| = %.4f s on 344,841 rows", d)
    if d != 0.0:
        raise SystemExit(f"parity failed: max |diff| = {d} s")
