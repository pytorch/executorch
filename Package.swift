// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This package manifest is for frameworks built locally with CMake.
// It defines dependencies and linker settings for Executorch components.
//
// To use prebuilt binaries instead, switch to one of the "swiftpm" branches,
// which fetch the precompiled `.xcframeworks`.
//
// For details on building frameworks locally or using prebuilt binaries,
// see the documentation:
// https://pytorch.org/executorch/main/using-executorch-ios

import PackageDescription

let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

let products = [
  "backend_coreml": [
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [:],
  "kernels_optimized": [
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_portable": [:],
  "kernels_quantized": [:],
].reduce(into: [String: [String: Any]]()) {
  $0[$1.key] = $1.value
  $0[$1.key + debug_suffix] = $1.value
}

let targets = [
  "threadpool",
].flatMap { [$0, $0 + debug_suffix] }

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v10_15),
  ],
  products: products.keys.map { key in
    .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
  }.sorted { $0.name < $1.name },
  targets: targets.map { key in
    .binaryTarget(
      name: key,
      path: "cmake-out/\(key).xcframework"
    )
  } + products.flatMap { key, value -> [Target] in
    [
      .binaryTarget(
        name: key,
        path: "cmake-out/\(key).xcframework"
      ),
      .target(
        name: "\(key)\(dependencies_suffix)",
        dependencies:([key] +
          (value["targets"] as? [String] ?? []).map {
            target in key.hasSuffix(debug_suffix) ? target + debug_suffix : target
          }).map { .target(name: $0) },
        path: ".Package.swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  } + [
    .testTarget(
      name: "tests",
      dependencies: [
        .target(name: "executorch\(debug_suffix)"),
        .target(name: "kernels_optimized\(dependencies_suffix)"),
      ],
      path: "extension/apple/ExecuTorch/__tests__",
      resources: [
        .copy("resources/add.pte"),
      ],
      linkerSettings: [
        .unsafeFlags([
          "-Xlinker", "-force_load",
          "-Xlinker", "cmake-out/kernels_optimized.xcframework/macos-arm64/libkernels_optimized_macos.a",
        ])
      ]
    )
  ]
)
