#include "book.hpp"

#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>

namespace {

std::vector<std::string_view> split(std::string_view s, char delim) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == delim) {
            out.emplace_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

std::vector<double> parse_doubles(std::string_view s) {
    std::vector<double> out;
    if (s.empty()) return out;
    for (auto tok : split(s, ',')) {
        out.push_back(std::stod(std::string(tok)));
    }
    return out;
}

std::int64_t parse_i64(std::string_view s) {
    std::int64_t v = 0;
    std::from_chars(s.data(), s.data() + s.size(), v);
    return v;
}

qr::BookEvent parse_event(std::string_view line) {
    auto fields = split(line, '|');
    if (fields.size() != 8) {
        throw std::runtime_error("bad line, expected 8 fields, got " +
                                 std::to_string(fields.size()));
    }
    qr::BookEvent ev;
    ev.event_type = std::string(fields[0]);
    ev.event_time = parse_i64(fields[1]);
    ev.received_at_ns = parse_i64(fields[2]);
    ev.update_id = parse_i64(fields[3]);
    ev.bids_px = parse_doubles(fields[4]);
    ev.bids_qty = parse_doubles(fields[5]);
    ev.asks_px = parse_doubles(fields[6]);
    ev.asks_qty = parse_doubles(fields[7]);
    return ev;
}

void print_state(const qr::BookState& s, std::size_t show = 5) {
    std::cout << "update_id      : " << s.update_id << "\n";
    std::cout << "received_at_ns : " << s.received_at_ns << "\n";
    std::cout << "bids:\n";
    for (std::size_t i = 0; i < std::min(show, s.bids.size()); ++i) {
        std::cout << "  " << std::setw(12) << s.bids[i].price << "  "
                  << std::setw(12) << s.bids[i].qty << "\n";
    }
    std::cout << "asks:\n";
    for (std::size_t i = 0; i < std::min(show, s.asks.size()); ++i) {
        std::cout << "  " << std::setw(12) << s.asks[i].price << "  "
                  << std::setw(12) << s.asks[i].qty << "\n";
    }
    std::cout << std::fixed << std::setprecision(6);
    if (auto v = s.mid())        std::cout << "mid        : " << *v << "\n";
    if (auto v = s.spread())     std::cout << "spread     : " << *v << "\n";
    if (auto v = s.microprice()) std::cout << "microprice : " << *v << "\n";
    if (auto v = s.imbalance())  std::cout << "imbalance  : " << *v << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: book_replay <events.psv>\n";
        return 1;
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 1;
    }

    std::string header;
    std::getline(in, header);

    qr::OrderBook book;
    std::size_t n = 0;
    std::string line;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        book.apply(parse_event(line));
        ++n;
    }
    const auto t1 = std::chrono::steady_clock::now();

    const auto dt_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "events processed : " << n << "\n";
    std::cout << "time             : " << dt_ms << " ms\n";
    if (dt_ms > 0) {
        std::cout << "events/sec       : " << (1000 * n / dt_ms) << "\n";
    }
    std::cout << "\n--- final top-5 ---\n";
    print_state(book.state(10), 5);
    return 0;
}
