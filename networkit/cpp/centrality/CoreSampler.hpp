#ifndef NETWORKIT_CENTRALITY_CORE_SAMPLER_HPP_
#define NETWORKIT_CENTRALITY_CORE_SAMPLER_HPP_
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <networkit/Globals.hpp>

namespace NetworKit {
namespace CoreDecompositionDetail {
// reset() is called only between parallel sampling phases.
class Sampler final {
public:
    void reset(count newExpectedHits, double newRate) {
        expectedHits = newExpectedHits;
        rate = std::min(1.0, newRate);
        threshold =
            static_cast<uint32_t>(rate * static_cast<double>(std::numeric_limits<uint32_t>::max()));
        hits.store(0, std::memory_order_relaxed);
    }

    bool sample(uint32_t randomNumber) {
        count current = hits.load(std::memory_order_relaxed);
        if (current >= expectedHits || randomNumber >= threshold) {
            return false;
        }

        current = hits.fetch_add(1, std::memory_order_relaxed);
        return current + 1 == expectedHits;
    }

    count numberOfHits() const { return hits.load(std::memory_order_relaxed); }

    count numberOfExpectedHits() const { return expectedHits; }

    double sampleRate() const { return rate; }

private:
    std::atomic<count> hits{0};
    count expectedHits{0};
    uint32_t threshold{0};
    double rate{0.0};
};

} // namespace CoreDecompositionDetail
} // namespace NetworKit
#endif
