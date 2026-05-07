# qr — quant research

## Контекст
Pet-проект для подготовки к позициям quant researcher / HFT. Цель — публичный артефакт на GitHub с честным end-to-end пайплайном: сбор tick-data → реконструкция книги → фичи микроструктуры → ML-предикт → бэктест с моделью очереди.

Бэкграунд владельца: ML extraction (lease_int, government_invoice), C++ только базовый, микроструктуру учу с нуля. Этот проект — учебная база, не претендует на реальную торговлю.

## Текущий этап
**Stage 2a — book reconstruction + первые фичи (готово).** `book.py` восстанавливает top-N книгу из snapshot+delta. `analyze.py` строит features DataFrame и графики (`plots/`). На 10 мин данных BTCUSDT уже видно монотонную предсказательную силу top-of-book imbalance на горизонте ~1.6 сек.

**Stage 1 — collector.** Bybit V5 public linear stream (`stream.bybit.com/v5/public/linear`): `orderbook.50.{symbol}` (snapshot+delta) + `publicTrade.{symbol}` для BTCUSDT/ETHUSDT/SOLUSDT.

Не Binance Futures: с этого IP `fstream.binance.com` aggTrade молча режется (geo-фильтр), depth идёт, trades нет. На сервере с другим IP можно вернуться через `QR_WS_URL`.

## Стек
- Python 3.11, uv для зависимостей
- websockets, orjson, pyarrow
- Запуск: `uv run python collector.py`
- Деплой: Ubuntu+RTX сервер, systemd unit (`scripts/qr-collector.service`)

## Конвенции
- Комменты — только когда непонятно ПОЧЕМУ. Что код делает — видно по идентификаторам.
- Времена: `event_time` / `transact_time` в миллисекундах (как у биржи), `received_at_ns` в наносекундах (локальное).
- Никаких хардкодов символов/глубины — всё через ENV (`QR_*`).
- Данные (`data/`) НЕ коммитим.

## Особенности данных
- Bybit orderbook стрим — это **дельты**: после snapshot приходят incremental updates. Для top-N в момент T нужно применить все delta событий до T к последнему snapshot.
- Trades приходят **массивами** в одном envelope (`data` — список словарей).
- `side` в trades — это сторона тейкера (агрессора). `Buy` = тейкер купил по ask. Это уже trade-sign, не нужен Lee-Ready на perp.

## Roadmap
1. **Stage 1: collector** (in progress).
2. **Stage 2: book reconstruction + фичи** — построить из дельт top-N книгу в любой момент, посчитать OFI, microprice, queue depth, spreads.
3. **Stage 3: ML** — LightGBM на предикт return, валидация — purged k-fold или walk-forward.
4. **Stage 4: backtest** — event-driven с симуляцией очереди, маркауты (PnL @ 1с/10с/1мин), maker/taker fee + slippage.

## Что почитать (для контекста)
- Larry Harris — "Trading and Exchanges" (микроструктура, словарь)
- Cartea/Jaimungal/Penalva — "Algorithmic and High-Frequency Trading" (LOB формализм)
- Bouchaud — "Trades, Quotes and Prices" (impact, queue dynamics)
- Lopez de Prado — "Advances in Financial ML" (meta-labeling, purged CV)
