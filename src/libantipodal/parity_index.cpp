/**
 * parity_index.cpp
 *
 * Implementation of parity-stratified index
 */

#include "parity_index.h"
#include "backends/faiss_backend.h"
#include <algorithm>
#include <stdexcept>

namespace antipodal {

ParityIndex::ParityIndex(size_t dimension, const std::string& backend_type)
    : dimension_(dimension) {

    // Create backend instances
    if (backend_type == "faiss") {
        i0_ = std::make_unique<FaissBackend>(dimension);
        i_plus_ = std::make_unique<FaissBackend>(dimension);
        i_minus_ = std::make_unique<FaissBackend>(dimension);
    } else {
        throw std::invalid_argument("Unknown backend type: " + backend_type);
    }
}

void ParityIndex::insert(const Vector& v, size_t id) {
    // Compute energy partition
    auto [alpha_plus, alpha_minus] = energy_partition(v);
    float alpha_max = std::max(alpha_plus, alpha_minus);

    if (alpha_max > K_STAR) {
        // Structure-dominated: use parity index
        if (alpha_plus > alpha_minus) {
            // Insert to I₊
            Vector v_even = project_even(v);
            i_plus_->add(v_even);
            id_map_iplus_.push_back(id);
            size_iplus_++;
        } else {
            // Insert to I₋
            Vector v_odd = project_odd(v);
            i_minus_->add(v_odd);
            id_map_iminus_.push_back(id);
            size_iminus_++;
        }
    } else {
        // Entropy-dominated: use quotient index
        Vector v_can = canonical(v);
        i0_->add(v_can);
        id_map_i0_.push_back(id);
        size_i0_++;
    }
}

std::vector<SearchResult> ParityIndex::search(const Vector& query, size_t k) {
    // Compute energy partition for query
    auto [alpha_plus, alpha_minus] = energy_partition(query);
    float alpha_max = std::max(alpha_plus, alpha_minus);

    std::vector<SearchResult> results;

    if (alpha_max > K_STAR) {
        // Structure-dominated: search parity index
        if (alpha_plus > alpha_minus) {
            // Search I₊
            Vector q_even = project_even(query);
            auto backend_results = i_plus_->search(q_even, k);

            // Map back to external IDs
            for (const auto& r : backend_results) {
                results.push_back({id_map_iplus_[r.index], r.distance});
            }
        } else {
            // Search I₋
            Vector q_odd = project_odd(query);
            auto backend_results = i_minus_->search(q_odd, k);

            // Map back to external IDs
            for (const auto& r : backend_results) {
                results.push_back({id_map_iminus_[r.index], r.distance});
            }
        }
    } else {
        // Entropy-dominated: search quotient index
        Vector q_can = canonical(query);
        auto backend_results = i0_->search(q_can, k);

        // Map back to external IDs
        for (const auto& r : backend_results) {
            results.push_back({id_map_i0_[r.index], r.distance});
        }
    }

    // Sort and return top-k
    std::sort(results.begin(), results.end());
    if (results.size() > k) {
        results.resize(k);
    }

    return results;
}

size_t ParityIndex::memory_bytes() const {
    return i0_->memory_bytes() +
           i_plus_->memory_bytes() +
           i_minus_->memory_bytes();
}

}  // namespace antipodal
