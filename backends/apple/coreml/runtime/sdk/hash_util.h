//
// hash_util.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <functional>
#include <type_traits>

namespace executorchcoreml {
inline void hash_combine(size_t& seed, size_t hash) {
    // Combiner taken from boost::hash_combine
    seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <class T> inline void hash_combine(size_t& seed, T const& v) { hash_combine(seed, std::hash<T> {}(v)); }

template <class T> inline size_t container_hash(const T& values) {
    size_t seed = 0;
    for (auto it = values.begin(); it != values.end(); ++it) {
        executorchcoreml::hash_combine(seed, *it);
    }

    return seed;
}
}
