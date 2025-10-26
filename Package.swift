// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251026"
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
    "sha256": "b5bc5cec298c956d0e81f9171d054c81930624905658e6f0466fab0a07439cfd",
    "sha256" + debug_suffix: "942fc07e7c12fd5455417df1335956264caeb429ba7a565806474b92fa9b4b5b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9594918b25dde68f29caffb1f152082e026065b2c7a91782d69fb6158e6a2eae",
    "sha256" + debug_suffix: "bcac1b3a8989cd75864dd8fb1bf79a9046edd3410c4e8c5aa39acac1085ce4d6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "02b5ca22eaef84d0262eaa03f31cd09a65cccc2230ec18f265d13b10284094c5",
    "sha256" + debug_suffix: "cd74c0d5d4a8eac490660eb015ed8254d91f420ec4ee784263997980fcff2fe7",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "891d995f68ff7e77531bf5a887140f03bb5dfb08daa1daad933bbd53c923c53c",
    "sha256" + debug_suffix: "35cac0110585bccd40e0b9ddbdde99518328339059831addb556caf3499f00a1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "1c4b4611e0d40fe677c51a8e3bc3ac5661a5e67f899ecbcda30ff1c1d0c9747d",
    "sha256" + debug_suffix: "92a050e8204e5704a019a1d60f728b0f70ff487665c010bb96522b8ddde616f1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a2626dd6be98b6359e1ce2bad37b7892b60d464b9a175eb709f3eeb966a68780",
    "sha256" + debug_suffix: "64d35f6ba2b08468e94a3498c6278660af4e39eb4b876ed0c2f74b703c0c4622",
  ],
  "kernels_optimized": [
    "sha256": "aefa3be9568056d28c0ea2d3dd07a2379e131b4e0b8ef1f41559c4498d7af82f",
    "sha256" + debug_suffix: "bd159c51e243e071b0a6b8243d9a889c007c8b5a57c681742144f5fef3164d25",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "683f5db9cee3945de85fc809c1353ed4b4d9ae0f03561dc27f7ac70f5006e976",
    "sha256" + debug_suffix: "3226d3ed0fcde816e1830952064a546bfc575047bffa97f0c77945110de20e85",
  ],
  "kernels_torchao": [
    "sha256": "5d55623cb0db1a2e1c6e0beb029536d06945bca8ed42c0c73ba4eb47c725e55e",
    "sha256" + debug_suffix: "d1cf2ea6f370680462e78ed960edf31dd8c01459f8b317e978eefd4cee2b1bfb",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d4d35474f901cc35ea5f6943de3f25eed233aeef92e6164d3946436c45b5542c",
    "sha256" + debug_suffix: "6d21960f3147f3f6d066e7ae77b96ee9cfe784ef6e486411278196e7a400bc57",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
