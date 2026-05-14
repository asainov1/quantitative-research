"""Export book events from parquet → pipe-separated CSV for the C++ replayer.

Format:
    event_type|event_time|received_at_ns|update_id|bids_px|bids_qty|asks_px|asks_qty

Array fields are comma-separated. Header row included.

Usage:
    uv run python cpp/export_events.py BTCUSDT cpp/events_btcusdt.psv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from book import load_book_events


def main() -> None:
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "BTCUSDT"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"cpp/events_{symbol.lower()}.psv")

    events = load_book_events(symbol)
    print(f"loaded {len(events):,} events for {symbol}")

    with out_path.open("w") as f:
        f.write("event_type|event_time|received_at_ns|update_id|bids_px|bids_qty|asks_px|asks_qty\n")
        for ev in events:
            bp = ",".join(repr(float(x)) for x in ev["bids_px"])
            bq = ",".join(repr(float(x)) for x in ev["bids_qty"])
            ap = ",".join(repr(float(x)) for x in ev["asks_px"])
            aq = ",".join(repr(float(x)) for x in ev["asks_qty"])
            f.write(
                f"{ev['event_type']}|{ev['event_time']}|{ev['received_at_ns']}|"
                f"{ev['update_id']}|{bp}|{bq}|{ap}|{aq}\n"
            )

    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
