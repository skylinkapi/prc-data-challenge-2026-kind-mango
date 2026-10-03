"""Hourly watchdog for the 2025 ADSB.lol downloads: restart the fetch when it dies or stops making progress.

Each check counts the points files and the bytes in the raw folder. If no
fetch process runs, or neither number changed since the last check, the
watchdog stops every fetch process and starts one fetch over the whole
2025 range; days already on disk are skipped. It exits when every released
2025 day is on disk or after `WATCH_MAX_HOURS`. Windows stays awake while
it runs.
"""
from __future__ import annotations

import ctypes
import json
import logging
import subprocess
import sys
import time

import pandas as pd

from src_v3 import config as C
from src_v3.fetch_adsb_days import POINTS, RAW, released

log = logging.getLogger(__name__)
PS = ["powershell", "-NoProfile", "-Command"]


def fetch_processes() -> list[dict]:
    """Python and PowerShell processes whose command line runs `fetch_adsb_days`."""
    cmd = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'fetch_adsb_days' "
           "-and $_.CommandLine -notmatch 'Get-CimInstance' } | Select-Object ProcessId, Name | ConvertTo-Json")
    out = subprocess.run([*PS, cmd], capture_output=True, text=True).stdout.strip()
    rows = json.loads(out) if out else []
    return rows if isinstance(rows, list) else [rows]


def progress() -> tuple[int, int]:
    return len(list(POINTS.glob("*.parquet"))), sum(p.stat().st_size for p in RAW.glob("*"))


def restart(missing: int) -> None:
    for p in fetch_processes():
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p["ProcessId"])], capture_output=True)
    log.warning("restart: %d days missing; fetch processes stopped", missing)
    with open(C.ROOT / "logs" / "adsb_fetch_2025_watchdog.log", "a", encoding="utf-8") as f:
        subprocess.Popen([sys.executable, "-m", "src_v3.fetch_adsb_days", "--start", C.WATCH_START, "--end", C.WATCH_END],
                         cwd=C.ROOT, stdout=f, stderr=subprocess.STDOUT,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(C.ROOT / "logs" / "adsb_watchdog.log"), logging.StreamHandler()])
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    days = released([d.strftime("%Y.%m.%d") for d in pd.date_range(C.WATCH_START, C.WATCH_END)])
    last = None
    for _ in range(C.WATCH_MAX_HOURS):
        missing = [d for d in days if not (POINTS / f"{d}.parquet").exists()]
        if not missing:
            log.info("all %d released days are on disk; watchdog ends", len(days))
            return
        now, procs = progress(), fetch_processes()
        log.info("check: %d days missing (next %s), %d points files, raw %.2f GB, %d fetch processes",
                 len(missing), missing[0], now[0], now[1] / 1e9, len(procs))
        if not procs or now == last:
            restart(len(missing))
        last = now
        time.sleep(C.WATCH_INTERVAL_S)
    log.warning("watchdog stops after %d hours with days still missing", C.WATCH_MAX_HOURS)


if __name__ == "__main__":
    main()
