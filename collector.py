"""Bybit linear (USDT perpetual) L2 orderbook + publicTrade collector.

Bybit V5 public stream — `orderbook.{depth}.{symbol}` (snapshot + deltas)
and `publicTrade.{symbol}` (every trade). One websocket connection,
subscribe op, periodic ping. Buffered flushes to per-symbol parquet
under `data/bybit_linear/{stream}/{symbol}/{date}/{hour}/`.

Why Bybit, not Binance Futures: Binance fapi aggTrade is geo-filtered
from this IP (depth comes through, trades silently drop). Bybit is a
top-3 perp venue, KZ-friendly, similar microstructure flavour.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import websockets
from dotenv import load_dotenv

load_dotenv()

WS_URL = os.environ.get("QR_WS_URL", "wss://stream.bybit.com/v5/public/linear")
SYMBOLS = [s.strip().upper() for s in os.environ.get("QR_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip()]
BOOK_DEPTH = int(os.environ.get("QR_BOOK_DEPTH", "50"))  # bybit supports 1, 50, 200, 500
DATA_ROOT = Path(os.environ.get("QR_DATA_ROOT", "data/bybit_linear")).resolve()
FLUSH_INTERVAL_SEC = float(os.environ.get("QR_FLUSH_INTERVAL_SEC", "30"))
PING_INTERVAL_SEC = 20.0
RECONNECT_BACKOFF_INITIAL = 1.0
RECONNECT_BACKOFF_MAX = 60.0

log = logging.getLogger("qr.collector")


def subscribe_args(symbols: list[str], depth: int) -> list[str]:
    args = []
    for s in symbols:
        args.append(f"orderbook.{depth}.{s}")
        args.append(f"publicTrade.{s}")
    return args


def book_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("event_time", pa.int64()),
            pa.field("symbol", pa.string()),
            pa.field("update_id", pa.int64()),
            pa.field("seq", pa.int64()),
            pa.field("event_type", pa.string()),
            pa.field("bids_px", pa.list_(pa.float64())),
            pa.field("bids_qty", pa.list_(pa.float64())),
            pa.field("asks_px", pa.list_(pa.float64())),
            pa.field("asks_qty", pa.list_(pa.float64())),
            pa.field("received_at_ns", pa.int64()),
        ]
    )


def trades_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("event_time", pa.int64()),
            pa.field("symbol", pa.string()),
            pa.field("trade_id", pa.string()),
            pa.field("side", pa.string()),
            pa.field("price", pa.float64()),
            pa.field("qty", pa.float64()),
            pa.field("tick_direction", pa.string()),
            pa.field("is_block_trade", pa.bool_()),
            pa.field("received_at_ns", pa.int64()),
        ]
    )


def parse_book_event(envelope: dict, received_at_ns: int) -> dict:
    data = envelope["data"]
    bids = data.get("b", [])
    asks = data.get("a", [])
    return {
        "event_time": envelope["ts"],
        "symbol": data["s"],
        "update_id": data.get("u", -1),
        "seq": data.get("seq", -1),
        "event_type": envelope.get("type", ""),
        "bids_px": [float(p) for p, _ in bids],
        "bids_qty": [float(q) for _, q in bids],
        "asks_px": [float(p) for p, _ in asks],
        "asks_qty": [float(q) for _, q in asks],
        "received_at_ns": received_at_ns,
    }


def parse_trade(t: dict, received_at_ns: int) -> dict:
    return {
        "event_time": t["T"],
        "symbol": t["s"],
        "trade_id": t.get("i", ""),
        "side": t.get("S", ""),
        "price": float(t["p"]),
        "qty": float(t["v"]),
        "tick_direction": t.get("L", ""),
        "is_block_trade": bool(t.get("BT", False)),
        "received_at_ns": received_at_ns,
    }


class ParquetSink:
    """Per-(stream_type, symbol) sink. Each flush writes one parquet file
    under data_root/stream_type/symbol/YYYY-MM-DD/HH/."""

    def __init__(self, stream_type: str, symbol: str, schema: pa.Schema, data_root: Path):
        self.stream_type = stream_type
        self.symbol = symbol.lower()
        self.schema = schema
        self.data_root = data_root
        self.buffer: list[dict] = []

    def append(self, row: dict) -> None:
        self.buffer.append(row)

    def flush(self) -> int:
        if not self.buffer:
            return 0
        rows = self.buffer
        self.buffer = []
        now = datetime.now(timezone.utc)
        out_dir = (
            self.data_root
            / self.stream_type
            / self.symbol
            / now.strftime("%Y-%m-%d")
            / now.strftime("%H")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{now.strftime('%Y%m%dT%H%M%S')}_{now.microsecond:06d}.parquet"
        table = pa.Table.from_pylist(rows, schema=self.schema)
        pq.write_table(table, out_dir / fname, compression="zstd")
        return len(rows)


async def sleep_or_stop(stop: asyncio.Event, seconds: float) -> bool:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False


async def ping_loop(ws, stop: asyncio.Event) -> None:
    while not stop.is_set():
        if await sleep_or_stop(stop, PING_INTERVAL_SEC):
            return
        try:
            await ws.send(orjson.dumps({"op": "ping"}).decode())
        except Exception:
            return


async def consume(sinks: dict, stop: asyncio.Event) -> None:
    args = subscribe_args(SYMBOLS, BOOK_DEPTH)
    backoff = RECONNECT_BACKOFF_INITIAL
    counters: dict[tuple[str, str], int] = defaultdict(int)
    last_log = time.monotonic()

    while not stop.is_set():
        ping_task: asyncio.Task | None = None
        try:
            log.info("connecting %s ...", WS_URL)
            async with websockets.connect(
                WS_URL,
                ping_interval=None,  # bybit wants app-level pings
                max_size=8 * 1024 * 1024,
            ) as ws:
                log.info("connected; subscribing %d topics", len(args))
                await ws.send(orjson.dumps({"op": "subscribe", "args": args}).decode())
                ping_task = asyncio.create_task(ping_loop(ws, stop))
                backoff = RECONNECT_BACKOFF_INITIAL

                async for msg in ws:
                    if stop.is_set():
                        break
                    received_at_ns = time.time_ns()
                    try:
                        env = orjson.loads(msg)
                    except orjson.JSONDecodeError:
                        log.warning("invalid json: %r", msg[:200])
                        continue

                    topic = env.get("topic")
                    if not topic:
                        if env.get("op") in ("subscribe", "ping", "pong"):
                            continue
                        if "success" in env or "ret_msg" in env:
                            continue
                        log.debug("non-topic msg: %s", str(env)[:200])
                        continue

                    try:
                        if topic.startswith("orderbook."):
                            symbol = env["data"]["s"]
                            sinks[("book", symbol)].append(
                                parse_book_event(env, received_at_ns)
                            )
                            counters[("book", symbol)] += 1
                        elif topic.startswith("publicTrade."):
                            for t in env.get("data", []):
                                symbol = t["s"]
                                sinks[("trades", symbol)].append(
                                    parse_trade(t, received_at_ns)
                                )
                                counters[("trades", symbol)] += 1
                    except (KeyError, ValueError, TypeError):
                        log.exception("parse error topic=%s", topic)

                    now = time.monotonic()
                    if now - last_log >= 60:
                        snapshot = dict(counters)
                        counters.clear()
                        last_log = now
                        log.info("rate (last 60s): %s", snapshot)
        except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
            log.warning("ws disconnected: %s; reconnect in %.1fs", e, backoff)
        except Exception:
            log.exception("unexpected error; reconnect in %.1fs", backoff)
        finally:
            if ping_task is not None:
                ping_task.cancel()
                try:
                    await ping_task
                except (asyncio.CancelledError, Exception):
                    pass

        if await sleep_or_stop(stop, backoff):
            break
        backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)


async def flusher(sinks: dict, stop: asyncio.Event) -> None:
    while not stop.is_set():
        if await sleep_or_stop(stop, FLUSH_INTERVAL_SEC):
            break
        total = 0
        for sink in sinks.values():
            total += sink.flush()
        if total:
            log.info("flushed %d rows", total)
    total = sum(sink.flush() for sink in sinks.values())
    if total:
        log.info("final flush: %d rows", total)


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("QR_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("data root: %s", DATA_ROOT)
    log.info(
        "symbols: %s, book_depth: %d, flush_interval: %.1fs",
        SYMBOLS, BOOK_DEPTH, FLUSH_INTERVAL_SEC,
    )

    b_sch = book_schema()
    t_sch = trades_schema()
    sinks: dict = {}
    for sym in SYMBOLS:
        sinks[("book", sym)] = ParquetSink("book", sym, b_sch, DATA_ROOT)
        sinks[("trades", sym)] = ParquetSink("trades", sym, t_sch, DATA_ROOT)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await asyncio.gather(consume(sinks, stop), flusher(sinks, stop))
    log.info("shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
