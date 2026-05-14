#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace qr {

struct PriceLevel {
    double price;
    double qty;
};

struct BookState {
    std::int64_t event_time = -1;
    std::int64_t received_at_ns = -1;
    std::int64_t update_id = -1;
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;

    [[nodiscard]] std::optional<double> mid() const;
    [[nodiscard]] std::optional<double> spread() const;
    [[nodiscard]] std::optional<double> microprice() const;
    [[nodiscard]] std::optional<double> imbalance() const;
};

struct BookEvent {
    std::string event_type;
    std::int64_t event_time;
    std::int64_t received_at_ns;
    std::int64_t update_id;
    std::vector<double> bids_px;
    std::vector<double> bids_qty;
    std::vector<double> asks_px;
    std::vector<double> asks_qty;
};

class OrderBook {
public:
    void apply(const BookEvent& ev);
    [[nodiscard]] BookState state(std::size_t top_n = 10) const;

private:
    std::map<double, double, std::greater<>> bids_;
    std::map<double, double, std::less<>> asks_;
    std::int64_t update_id_ = -1;
    std::int64_t event_time_ = -1;
    std::int64_t received_at_ns_ = -1;
    bool snapshot_seen_ = false;
};

}  // namespace qr
