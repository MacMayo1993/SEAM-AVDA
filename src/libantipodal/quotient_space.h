/**
 * quotient_space.h
 *
 * Core operations for quotient space ℝPⁿ⁻¹ = ℂⁿ / ℤ₂
 *
 * Implements:
 * - Canonical representatives σ(v)
 * - Parity operators P₊, P₋ (gauge-relative)
 * - Energy partition α_±
 */

#ifndef QUOTIENT_SPACE_H
#define QUOTIENT_SPACE_H

#include <vector>
#include <utility>
#include <cmath>

namespace antipodal {

using Vector = std::vector<float>;
using EnergyPartition = std::pair<float, float>;  // (α₊, α₋)

/**
 * Canonicalize vector to quotient representative
 *
 * Algorithm:
 * 1. Normalize vector
 * 2. Find first nonzero element
 * 3. If negative, flip entire vector
 *
 * @param v Input vector
 * @return Canonical representative σ(v)
 */
Vector canonical(const Vector& v);

/**
 * Compute energy partition α_± for vector
 *
 * Measures alignment with canonical/anticanonical representatives.
 * Returns (α₊, α₋) where α₊ + α₋ = 1.
 *
 * - α₊ > k*: structure-dominated (use parity index)
 * - α₊ < k*: entropy-dominated (use quotient index)
 *
 * @param v Input vector
 * @return (α₊, α₋) energy partition
 */
EnergyPartition energy_partition(const Vector& v);

/**
 * Project vector to even parity subspace
 *
 * P₊(v) = ½(v + S_σ(v))
 * where S_σ is gauge-relative involution
 *
 * @param v Input vector
 * @return Even projection
 */
Vector project_even(const Vector& v);

/**
 * Project vector to odd parity subspace
 *
 * P₋(v) = ½(v - S_σ(v))
 *
 * @param v Input vector
 * @return Odd projection
 */
Vector project_odd(const Vector& v);

/**
 * Compute seam score Ψ(v) for boundary detection
 *
 * Ψ(v) = |α₊ - α₋| / max(α₊, α₋)
 *
 * High score → vector near seam (semantic boundary)
 * Low score → vector interior to region
 *
 * @param v Input vector
 * @return Seam score ∈ [0, 1]
 */
float seam_score(const Vector& v);

/**
 * Helper: Normalize vector to unit length
 */
Vector normalize(const Vector& v);

/**
 * Helper: Compute dot product
 */
float dot(const Vector& a, const Vector& b);

/**
 * Helper: Compute L2 norm
 */
float norm(const Vector& v);

/**
 * Threshold constant k* ≈ 0.721
 * Derived from equal-cost boundary in phase diagram
 */
constexpr float K_STAR = 0.72134752044f;

}  // namespace antipodal

#endif  // QUOTIENT_SPACE_H
