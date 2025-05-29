// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250528"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "sha256": "6659c7918575eeaaf8c612e79f1e01edb192a1896159f2718703a46188e24376",
    "sha256" + debug_suffix: "db81a7b73449363a3d8f3699a2f3a7389825ebe876cd31ab1616fdff823fd3ca",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9a1887310273773ed300afbb3b8e7e9ee9ec4146d0e2ec6092215ac44f949b9e",
    "sha256" + debug_suffix: "7fa331ecb70b655a1f306a1d8d336120596e9cf44e06fba8acd71fd20ae1fb82",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4dd221334514656b94680e807a5314fd026e3917615942129800f28af543314a",
    "sha256" + debug_suffix: "18edb4140e38fb49b059620695b7633c01a4021d4652de13054a86c8e4df401a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "42e877aa3a040394d6ceb16dcdb433f93903f8e6e9f7fbfa801eea7c7bf54c92",
    "sha256" + debug_suffix: "5eeaf8aa22f061c6e995aa839096a08e6383086929cf4799a4cde9f0788d9d54",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "410d44fb0595cb88b8de5372344f537c908c2b87bb24d6a882a1008b98c26eef",
    "sha256" + debug_suffix: "3b644bc52de93580c5f15211c7c2e99eefd9b7c28fefbafb8803568a6bca1bf8",
  ],
  "kernels_optimized": [
    "sha256": "215bacc2a01d4c39aa4c3f7758a01621e36f7f0e330dc8fa96aecee1972463fd",
    "sha256" + debug_suffix: "4444300e8166a77ab88c117d2866b7bf4c5bfd77efc2f3a5a711684171fef8c0",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_portable": [
    "sha256": "e65c7e8cf7dccceba00d4a53178be7906444388cb574fb43a9adb78dd2eab39d",
    "sha256" + debug_suffix: "8ef8178a1448c6af8ca8ff536e5d5d3d9fc2f519b9681915a8dae182fd3e57e2",
  ],
  "kernels_quantized": [
    "sha256": "612f831b5d9646872cc182794ccd58b7f8f8fa46579c130bbc15874cee89428f",
    "sha256" + debug_suffix: "b96ac5f82315cf2eb6188c1de02c5dbb5287cef82ef1c1acc67d1963b424e782",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "90a5c1e76e131f876afd8ba1deb63b2790497b05035eb56e6ca8574b922cad2b",
    "sha256" + debug_suffix: "248218196aec0256aa276062e3f885d65034fb0371ac1b88ccc80033a9bf62a1",
  ],
])

let packageProducts: [Product] = products.keys.map { key in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

let packageTargets: [Target] = targets.flatMap { key, value -> [Target] in
  [
    .binaryTarget(
      name: key,
      url: "\(url)\(key)-\(version).zip",
      checksum: value["sha256"] as? String ?? ""
    ),
  ]
} + products.flatMap { key, value -> [Target] in
  [
    .binaryTarget(
      name: key,
      url: "\(url)\(key)-\(version).zip",
      checksum: value["sha256"] as? String ?? ""
    ),
    .target(
      name: "\(key)\(dependencies_suffix)",
      dependencies: ([key] +
        (value["targets"] as? [String] ?? []).map {
          target in key.hasSuffix(debug_suffix) ? target + debug_suffix : target
        }).map { .target(name: $0) },
      path: ".Package.swift/\(key)",
      linkerSettings:
        (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
        (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
    ),
  ]
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v10_15),
  ],
  products: packageProducts,
  targets: packageTargets
)
