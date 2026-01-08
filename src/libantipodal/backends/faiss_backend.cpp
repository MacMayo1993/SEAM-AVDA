/**
 * faiss_backend.cpp
 *
 * Implementation of FAISS backend
 */

#include "faiss_backend.h"

// Note: This requires FAISS to be installed
// Install with: pip install faiss-cpu
// Or use stub implementation if FAISS not available

#ifdef USE_FAISS
#include <faiss/IndexFlat.h>
#else
// Stub implementation for when FAISS is not available
namespace faiss {
    class Index {
    public:
        virtual ~Index() = default;
        virtual void add(long n, const float* x) = 0;
        virtual void search(long n, const float* x, long k, float* distances, long* labels) const = 0;
        virtual long ntotal() const = 0;
    };

    class IndexFlatIP : public Index {
    private:
        size_t dim_;
        std::vector<std::vector<float>> data_;

    public:
        explicit IndexFlatIP(size_t dim) : dim_(dim) {}

        void add(long n, const float* x) override {
            for (long i = 0; i < n; ++i) {
                std::vector<float> vec(x + i * dim_, x + (i + 1) * dim_);
                data_.push_back(vec);
            }
        }

        void search(long n, const float* x, long k, float* distances, long* labels) const override {
            // Simple brute-force search
            for (long q = 0; q < n; ++q) {
                std::vector<std::pair<float, long>> scores;

                for (size_t i = 0; i < data_.size(); ++i) {
                    float score = 0.0f;
                    for (size_t d = 0; d < dim_; ++d) {
                        score += x[q * dim_ + d] * data_[i][d];
                    }
                    scores.push_back({score, i});
                }

                // Sort by score (descending)
                std::sort(scores.begin(), scores.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

                // Take top-k
                for (long i = 0; i < std::min(k, (long)scores.size()); ++i) {
                    distances[q * k + i] = scores[i].first;
                    labels[q * k + i] = scores[i].second;
                }
            }
        }

        long ntotal() const override {
            return data_.size();
        }
    };
}
#endif

namespace antipodal {

FaissBackend::FaissBackend(size_t dimension)
    : dimension_(dimension) {
    // Create IndexFlatIP for inner product search
    index_ = std::make_unique<faiss::IndexFlatIP>(dimension);
}

FaissBackend::~FaissBackend() = default;

void FaissBackend::add(const Vector& v) {
    if (v.size() != dimension_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    // FAISS expects row-major float arrays
    index_->add(1, v.data());
    count_++;
}

std::vector<SearchResult> FaissBackend::search(const Vector& query, size_t k) {
    if (query.size() != dimension_) {
        throw std::invalid_argument("Query dimension mismatch");
    }

    if (count_ == 0) {
        return {};  // Empty index
    }

    // Limit k to index size
    k = std::min(k, count_);

    // Allocate result buffers
    std::vector<float> distances(k);
    std::vector<long> labels(k);

    // Search
    index_->search(1, query.data(), k, distances.data(), labels.data());

    // Convert to SearchResult format
    std::vector<SearchResult> results;
    for (size_t i = 0; i < k; ++i) {
        if (labels[i] >= 0) {  // Valid result
            results.push_back({static_cast<size_t>(labels[i]), distances[i]});
        }
    }

    return results;
}

size_t FaissBackend::size() const {
    return count_;
}

size_t FaissBackend::memory_bytes() const {
    // Estimate: dimension * count * sizeof(float)
    return dimension_ * count_ * sizeof(float);
}

}  // namespace antipodal
