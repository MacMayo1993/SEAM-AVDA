/**
 * quotient_space.cpp
 *
 * Implementation of quotient space operations
 */

#include "quotient_space.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace antipodal {

float norm(const Vector& v) {
    float sum = 0.0f;
    for (float val : v) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

float dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have same dimension");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

Vector normalize(const Vector& v) {
    float n = norm(v);
    if (n < 1e-10f) {
        throw std::invalid_argument("Cannot normalize zero vector");
    }

    Vector result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] / n;
    }
    return result;
}

Vector canonical(const Vector& v) {
    // Normalize
    Vector v_norm = normalize(v);

    // Find first nonzero element
    for (size_t i = 0; i < v_norm.size(); ++i) {
        if (std::abs(v_norm[i]) > 1e-10f) {
            // If negative, flip entire vector
            if (v_norm[i] < 0) {
                for (size_t j = 0; j < v_norm.size(); ++j) {
                    v_norm[j] = -v_norm[j];
                }
            }
            break;
        }
    }

    return v_norm;
}

EnergyPartition energy_partition(const Vector& v) {
    Vector v_can = canonical(v);
    Vector v_norm = normalize(v);

    // Compute alignment: cos(θ) between v and canonical
    float alignment = dot(v_norm, v_can);

    // Energy partition based on alignment
    float alpha_plus, alpha_minus;

    if (alignment > 0) {
        // Same hemisphere as canonical
        alpha_plus = alignment * alignment;
        alpha_minus = 1.0f - alpha_plus;
    } else {
        // Opposite hemisphere (anticanonical)
        alpha_minus = alignment * alignment;
        alpha_plus = 1.0f - alpha_minus;
    }

    // Ensure normalization
    float total = alpha_plus + alpha_minus;
    if (total > 1e-10f) {
        alpha_plus /= total;
        alpha_minus /= total;
    }

    return {alpha_plus, alpha_minus};
}

Vector project_even(const Vector& v) {
    Vector v_can = canonical(v);
    Vector v_norm = normalize(v);

    // Check alignment
    float alignment = dot(v_norm, v_can);

    if (alignment > 0) {
        // Already in even subspace
        return v_norm;
    } else {
        // In odd subspace → project to zero
        return Vector(v.size(), 0.0f);
    }
}

Vector project_odd(const Vector& v) {
    Vector v_can = canonical(v);
    Vector v_norm = normalize(v);

    // Check alignment
    float alignment = dot(v_norm, v_can);

    if (alignment < 0) {
        // In odd subspace → flip to canonical
        Vector result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = -v_norm[i];
        }
        return result;
    } else {
        // In even subspace → project to zero
        return Vector(v.size(), 0.0f);
    }
}

float seam_score(const Vector& v) {
    auto [alpha_plus, alpha_minus] = energy_partition(v);

    float alpha_max = std::max(alpha_plus, alpha_minus);
    if (alpha_max < 1e-10f) {
        return 0.0f;
    }

    // Seam score: |α₊ - α₋| / max(α₊, α₋)
    return std::abs(alpha_plus - alpha_minus) / alpha_max;
}

}  // namespace antipodal
