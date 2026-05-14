#include "book.hpp"

#include <algorithm>
#include <stdexcept>

namespace qr {

std::optional<double> BookState::mid() const {
    if (bids.empty() || asks.empty()) return std::nullopt;
    return (bids.front().price + asks.front().price) / 2.0;
}

std::optional<double> BookState::spread() const {
    if (bids.empty() || asks.empty()) return std::nullopt;
    return asks.front().price - bids.front().price;
}

std::optional<double> BookState::microprice() const {
    if (bids.empty() || asks.empty()) return std::nullopt;
    const double bq = bids.front().qty;
    const double aq = asks.front().qty;
    const double denom = bq + aq;
    if (denom <= 0.0) return std::nullopt;
    return (bq * asks.front().price + aq * bids.front().price) / denom;
}

std::optional<double> BookState::imbalance() const {
    if (bids.empty() || asks.empty()) return std::nullopt;
    const double bq = bids.front().qty;
    const double aq = asks.front().qty;
    const double denom = bq + aq;
    if (denom <= 0.0) return std::nullopt;
    return (bq - aq) / denom;
}

namespace {

void replace_side(auto& side, const std::vector<double>& px, const std::vector<double>& qty) {
    side.clear();
    for (std::size_t i = 0; i < px.size(); ++i) {
        if (qty[i] > 0.0) side[px[i]] = qty[i];
    }
}

void apply_delta(auto& side, const std::vector<double>& px, const std::vector<double>& qty) {
    for (std::size_t i = 0; i < px.size(); ++i) {
        if (qty[i] == 0.0) {
            side.erase(px[i]);
        } else {
            side[px[i]] = qty[i];
        }
    }
}

}  // anonymous namespace

void OrderBook::apply(const BookEvent& ev) {
    if (ev.bids_px.size() != ev.bids_qty.size() || ev.asks_px.size() != ev.asks_qty.size()) {
        throw std::invalid_argument("bids/asks px and qty size mismatch");
    }

    if (ev.event_type == "snapshot") {
        replace_side(bids_, ev.bids_px, ev.bids_qty);
        replace_side(asks_, ev.asks_px, ev.asks_qty);
        snapshot_seen_ = true;
    } else if (ev.event_type == "delta") {
        if (!snapshot_seen_) {
            replace_side(bids_, ev.bids_px, ev.bids_qty);
            replace_side(asks_, ev.asks_px, ev.asks_qty);
            snapshot_seen_ = true;
        } else {
            apply_delta(bids_, ev.bids_px, ev.bids_qty);
            apply_delta(asks_, ev.asks_px, ev.asks_qty);
        }
    } else {
        throw std::invalid_argument("unknown event_type: " + ev.event_type);
    }

    update_id_ = ev.update_id;
    event_time_ = ev.event_time;
    received_at_ns_ = ev.received_at_ns;
}

BookState OrderBook::state(std::size_t top_n) const {
    BookState s;
    s.event_time = event_time_;
    s.received_at_ns = received_at_ns_;
    s.update_id = update_id_;
    s.bids.reserve(std::min(top_n, bids_.size()));
    s.asks.reserve(std::min(top_n, asks_.size()));

    std::size_t i = 0;
    for (const auto& [p, q] : bids_) {
        if (i++ >= top_n) break;
        s.bids.push_back({p, q});
    }
    i = 0;
    for (const auto& [p, q] : asks_) {
        if (i++ >= top_n) break;
        s.asks.push_back({p, q});
    }
    return s;
}

}  // namespace qr
