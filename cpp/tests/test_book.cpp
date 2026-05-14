#include "book.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

bool approx(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

void test_snapshot_then_delta() {
    qr::OrderBook book;

    qr::BookEvent snap{
        .event_type = "snapshot",
        .event_time = 100,
        .received_at_ns = 200,
        .update_id = 1,
        .bids_px = {100.0, 99.0, 98.0},
        .bids_qty = {2.0, 3.0, 5.0},
        .asks_px = {101.0, 102.0, 103.0},
        .asks_qty = {1.0, 4.0, 6.0},
    };
    book.apply(snap);

    auto s = book.state(10);
    assert(s.bids.size() == 3);
    assert(s.asks.size() == 3);
    assert(approx(s.bids[0].price, 100.0));
    assert(approx(s.asks[0].price, 101.0));
    assert(approx(*s.mid(), 100.5));
    assert(approx(*s.spread(), 1.0));
    // microprice: bid_qty=2, ask_qty=1, mp = (2*101 + 1*100)/3 = 100.6667
    assert(approx(*s.microprice(), (2.0 * 101.0 + 1.0 * 100.0) / 3.0));
    // imbalance: (2-1)/3 = 0.3333
    assert(approx(*s.imbalance(), 1.0 / 3.0));

    qr::BookEvent del{
        .event_type = "delta",
        .event_time = 110,
        .received_at_ns = 210,
        .update_id = 2,
        .bids_px = {100.0, 99.0},
        .bids_qty = {0.0, 7.0},  // remove 100, replace 99 -> 7
        .asks_px = {101.5},
        .asks_qty = {8.0},  // new level inside
    };
    book.apply(del);
    s = book.state(10);

    assert(s.bids.size() == 2);
    assert(approx(s.bids[0].price, 99.0) && approx(s.bids[0].qty, 7.0));
    assert(s.asks.size() == 4);
    assert(approx(s.asks[0].price, 101.0));
    assert(approx(s.asks[1].price, 101.5));
}

void test_delta_before_snapshot_becomes_snapshot() {
    qr::OrderBook book;
    qr::BookEvent del{
        .event_type = "delta",
        .event_time = 1,
        .received_at_ns = 1,
        .update_id = 1,
        .bids_px = {50.0},
        .bids_qty = {1.0},
        .asks_px = {51.0},
        .asks_qty = {1.0},
    };
    book.apply(del);
    auto s = book.state(10);
    assert(s.bids.size() == 1);
    assert(s.asks.size() == 1);
}

void test_top_n_truncation() {
    qr::OrderBook book;
    qr::BookEvent snap{
        .event_type = "snapshot",
        .event_time = 1,
        .received_at_ns = 1,
        .update_id = 1,
        .bids_px = {100, 99, 98, 97, 96, 95},
        .bids_qty = {1, 1, 1, 1, 1, 1},
        .asks_px = {101, 102, 103, 104, 105, 106},
        .asks_qty = {1, 1, 1, 1, 1, 1},
    };
    book.apply(snap);
    auto s = book.state(3);
    assert(s.bids.size() == 3);
    assert(s.asks.size() == 3);
    assert(approx(s.bids[0].price, 100.0));
    assert(approx(s.bids[2].price, 98.0));
}

}  // namespace

int main() {
    test_snapshot_then_delta();
    test_delta_before_snapshot_becomes_snapshot();
    test_top_n_truncation();
    std::cout << "all tests passed\n";
    return 0;
}
