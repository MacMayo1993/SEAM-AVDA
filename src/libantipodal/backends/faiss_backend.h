/**
 * faiss_backend.h
 *
 * FAISS backend implementation for AVDA
 */

#ifndef FAISS_BACKEND_H
#define FAISS_BACKEND_H

#include "../parity_index.h"
#include <vector>
#include <memory>

// Forward declarations to avoid full FAISS dependency in header
namespace faiss {
    class Index;
}

namespace antipodal {

/**
 * FAISS backend using IndexFlatIP (inner product)
 *
 * Uses FAISS's efficient vector search implementations
 */
class FaissBackend : public IndexBackend {
public:
    explicit FaissBackend(size_t dimension);
    ~FaissBackend() override;

    void add(const Vector& v) override;
    std::vector<SearchResult> search(const Vector& query, size_t k) override;
    size_t size() const override;
    size_t memory_bytes() const override;

private:
    size_t dimension_;
    std::unique_ptr<faiss::Index> index_;
    size_t count_ = 0;
};

}  // namespace antipodal

#endif  // FAISS_BACKEND_H
