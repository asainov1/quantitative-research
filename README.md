# qr — quant research

Pet-проект для входа в quant/HFT: сбор L2 orderbook + trades, фичи на микроструктуре, бэктест с моделью очереди.

## Stage 1 — collector (текущий этап)

Собирает с **Bybit linear (USDT perpetuals)** через V5 public stream:
- `orderbook.50.{symbol}` — snapshot + delta-обновления топ-50 уровней
- `publicTrade.{symbol}` — каждая исполненная сделка

> Изначально планировался Binance Futures, но `aggTrade` через `fstream.binance.com` с этого IP молча режется (depth идёт, trades — нет, geo-фильтр). Bybit — топ-3 perp-биржа, KZ-friendly, схожая микроструктура. На сервере с другим IP можно вернуться к Binance, поменяв `QR_WS_URL`.

Складывает в parquet (zstd) с разбивкой по символу и часу:

```
data/bybit_linear/{book|trades}/{symbol}/YYYY-MM-DD/HH/{timestamp}.parquet
```

### Data volume (ориентир)

При `book_depth=50` для BTC/ETH/SOL — ~3000 событий за 30с (~100/сек). Прикидка: **~5 GB/день** в zstd-parquet для трёх символов. Если упирается в место — снизить до `QR_BOOK_DEPTH=1` (top-of-book), это в десятки раз меньше.

### Запуск локально

```bash
cd ~/projects/qr
cp .env.example .env   # подправь символы / depth при необходимости
uv run python collector.py
```

Логи раз в минуту показывают rate по каждому стриму. Останавливать — Ctrl+C, final flush сохранит остаток буфера.

### Деплой на Ubuntu+RTX сервер

```bash
git clone <твой-github-url> ~/qr
cd ~/qr
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
cp .env.example .env
```

Запуск под systemd — см. `scripts/qr-collector.service`. Скопировать в `/etc/systemd/system/`, поправить `User` и `WorkingDirectory`, затем:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qr-collector
sudo journalctl -u qr-collector -f
```

### Проверка собранных данных

```python
import pyarrow.dataset as ds
trades = ds.dataset("data/bybit_linear/trades/btcusdt", format="parquet").to_table()
book = ds.dataset("data/bybit_linear/book/btcusdt", format="parquet").to_table()
print("trades:", trades.num_rows, "book events:", book.num_rows)
```

Schema:
- **trades**: event_time (ms), symbol, trade_id, side (Buy/Sell — taker side), price, qty, tick_direction, is_block_trade, received_at_ns
- **book**: event_time (ms), symbol, update_id, seq, event_type (snapshot|delta), bids_px[], bids_qty[], asks_px[], asks_qty[], received_at_ns

Book даёт **дельты**, не партишн-снапшоты. Для top-N в любой момент времени нужно реконструировать книгу: применять delta-события к последнему snapshot. Это будет в Stage 2.

## Roadmap

- [x] Stage 1: collector
- [ ] Stage 2: book reconstruction + features — order flow imbalance, microprice, queue depth, Lee-Ready trade signs
- [ ] Stage 3: GBDT (LightGBM) предикт return @ 100ms / 1s / 10s, walk-forward CV, purged k-fold
- [ ] Stage 4: event-driven backtest с моделью очереди, маркауты (PnL @ 1с/10с/1мин), maker/taker fee + slippage
