"""Reconstruct top-N orderbook from Bybit snapshot+delta events.

Bybit V5 orderbook stream semantics:
- 'snapshot' event: full top-N book replaces current state
- 'delta' event: partial updates
    - qty > 0: set/replace level at this price
    - qty == 0: remove level at this price

Events are ordered by their `received_at_ns` (local nanoseconds when we
received the message). `update_id` strictly increases within a single
websocket session but resets on reconnect, so we don't rely on it for
global ordering — `received_at_ns` is the canonical timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.dataset as ds


@dataclass
class BookState:
    """Top-N snapshot at a moment in time."""

    event_time: int           # bybit timestamp, ms
    received_at_ns: int       # local ns clock
    update_id: int
    bids: list[tuple[float, float]]   # sorted desc by price: best bid first
    asks: list[tuple[float, float]]   # sorted asc by price: best ask first

    @property
    def best_bid(self) -> tuple[float, float] | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> tuple[float, float] | None:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2.0

    @property
    def spread(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return self.asks[0][0] - self.bids[0][0]

    @property
    def microprice(self) -> float | None:
        """Volume-weighted fair midpoint at the top of book.

        Heavy bid → microprice closer to ask (sellers will get hit first).
        Heavy ask → microprice closer to bid.
        """
        if not self.bids or not self.asks:
            return None
        bid_px, bid_qty = self.bids[0]
        ask_px, ask_qty = self.asks[0]
        denom = bid_qty + ask_qty
        if denom <= 0:
            return None
        return (bid_qty * ask_px + ask_qty * bid_px) / denom

    @property
    def imbalance(self) -> float | None:
        """Top-of-book volume imbalance in [-1, 1]. >0 means bid-heavy."""
        if not self.bids or not self.asks:
            return None
        bq = self.bids[0][1]
        aq = self.asks[0][1]
        denom = bq + aq
        if denom <= 0:
            return None
        return (bq - aq) / denom


class OrderBook:
    """Mutable book that consumes snapshot/delta events and exposes top-N."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.update_id: int = -1
        self.event_time: int = -1
        self.received_at_ns: int = -1
        self._snapshot_seen = False

    def apply(self, event: dict) -> None:
        et = event["event_type"]
        bids_px = event["bids_px"]
        bids_qty = event["bids_qty"]
        asks_px = event["asks_px"]
        asks_qty = event["asks_qty"]

        if et == "snapshot":
            self.bids = {p: q for p, q in zip(bids_px, bids_qty) if q > 0}
            self.asks = {p: q for p, q in zip(asks_px, asks_qty) if q > 0}
            self._snapshot_seen = True
        elif et == "delta":
            if not self._snapshot_seen:
                # delta before snapshot — happens if we missed reconnect snapshot
                # safest: treat as snapshot on first delta
                self.bids = {p: q for p, q in zip(bids_px, bids_qty) if q > 0}
                self.asks = {p: q for p, q in zip(asks_px, asks_qty) if q > 0}
                self._snapshot_seen = True
            else:
                for p, q in zip(bids_px, bids_qty):
                    if q == 0:
                        self.bids.pop(p, None)
                    else:
                        self.bids[p] = q
                for p, q in zip(asks_px, asks_qty):
                    if q == 0:
                        self.asks.pop(p, None)
                    else:
                        self.asks[p] = q
        else:
            raise ValueError(f"unknown event_type: {et}")

        self.update_id = event["update_id"]
        self.event_time = event["event_time"]
        self.received_at_ns = event["received_at_ns"]

    def state(self, top_n: int = 10) -> BookState:
        bids_sorted = sorted(self.bids.items(), key=lambda x: -x[0])[:top_n]
        asks_sorted = sorted(self.asks.items(), key=lambda x: x[0])[:top_n]
        return BookState(
            event_time=self.event_time,
            received_at_ns=self.received_at_ns,
            update_id=self.update_id,
            bids=bids_sorted,
            asks=asks_sorted,
        )


def load_book_events(symbol: str, data_root: str | Path = "data/bybit_linear") -> list[dict]:
    """Load all book events for a symbol, ordered by receive time."""
    path = Path(data_root) / "book" / symbol.lower()
    if not path.exists():
        raise FileNotFoundError(f"no book data at {path}")
    table = ds.dataset(path, format="parquet").to_table()
    events = table.to_pylist()
    events.sort(key=lambda e: e["received_at_ns"])
    return events


def load_trades(symbol: str, data_root: str | Path = "data/bybit_linear") -> list[dict]:
    """Load all trades for a symbol, ordered by event time."""
    path = Path(data_root) / "trades" / symbol.lower()
    if not path.exists():
        raise FileNotFoundError(f"no trade data at {path}")
    table = ds.dataset(path, format="parquet").to_table()
    trades = table.to_pylist()
    trades.sort(key=lambda t: t["received_at_ns"])
    return trades


def replay(events: list[dict], top_n: int = 10) -> Iterator[BookState]:
    """Yield a BookState after each event is applied."""
    book = OrderBook()
    for ev in events:
        book.apply(ev)
        yield book.state(top_n=top_n)
