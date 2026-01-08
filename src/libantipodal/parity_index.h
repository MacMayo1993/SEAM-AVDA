/**
 * parity_index.h
 *
 * Parity-stratified index structure for AVDA
 *
 * Three-index architecture:
 * - I₀: Quotient index (canonical representatives only)
 * - I₊: Even parity index (α₊ > k*)
 * - I₋: Odd parity index (α₋ > k*)
 */

#ifndef PARITY_INDEX_H
#define PARITY_INDEX_H

#include "quotient_space.h"
#include <memory>
#include <vector>
#include <utility>

namespace antipodal {

// Forward declaration for backend interface
class IndexBackend;

/**
 * Search result: (index, distance)
 */
struct SearchResult {
    size_t index;
    float distance;

    bool operator<(const SearchResult& other) const {
        return distance < other.distance;  // For top-k sorting
    }
};

/**
 * Parity Index: Three-index AVDA structure
 *
 * Automatically routes vectors to appropriate index based on energy partition:
 * - High structure (α_max > k*): use parity index (I₊ or I₋)
 * - Low structure (α_max < k*): use quotient index (I₀)
 */
class ParityIndex {
public:
    /**
     * Constructor
     *
     * @param dimension Vector dimension
     * @param backend_type Backend type ("faiss", "milvus", etc.)
     */
    ParityIndex(size_t dimension, const std::string& backend_type = "faiss");

    /**
     * Insert vector into appropriate index
     *
     * Algorithm:
     * 1. Compute α_±
     * 2. If max(α₊, α₋) > k*, insert to I₊ or I₋
     * 3. Else, canonicalize and insert to I₀
     *
     * @param v Vector to insert
     * @param id External ID for this vector
     */
    void insert(const Vector& v, size_t id);

    /**
     * Adaptive k-NN search
     *
     * Algorithm:
     * 1. Compute α_± for query
     * 2. Choose index based on structure:
     *    - If α_max > k*: search I₊ or I₋ (parity-stratified)
     *    - Else: canonicalize and search I₀ (quotient-only)
     * 3. Return top-k results
     *
     * @param query Query vector
     * @param k Number of neighbors
     * @return Top-k results
     */
    std::vector<SearchResult> search(const Vector& query, size_t k);

    /**
     * Get statistics
     */
    size_t size_i0() const { return size_i0_; }
    size_t size_iplus() const { return size_iplus_; }
    size_t size_iminus() const { return size_iminus_; }
    size_t total_size() const { return size_i0_ + size_iplus_ + size_iminus_; }

    /**
     * Memory usage (bytes)
     */
    size_t memory_bytes() const;

private:
    size_t dimension_;

    // Three indices
    std::unique_ptr<IndexBackend> i0_;       // Quotient index
    std::unique_ptr<IndexBackend> i_plus_;   // Even parity
    std::unique_ptr<IndexBackend> i_minus_;  // Odd parity

    // Index sizes
    size_t size_i0_ = 0;
    size_t size_iplus_ = 0;
    size_t size_iminus_ = 0;

    // ID mappings (index_id → external_id)
    std::vector<size_t> id_map_i0_;
    std::vector<size_t> id_map_iplus_;
    std::vector<size_t> id_map_iminus_;
};

/**
 * Abstract backend interface
 *
 * Allows swapping FAISS, Milvus, etc.
 */
class IndexBackend {
public:
    virtual ~IndexBackend() = default;

    virtual void add(const Vector& v) = 0;
    virtual std::vector<SearchResult> search(const Vector& query, size_t k) = 0;
    virtual size_t size() const = 0;
    virtual size_t memory_bytes() const = 0;
};

}  // namespace antipodal

#endif  // PARITY_INDEX_H
