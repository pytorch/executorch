// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0"
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
    "sha256": "3011f833a917c3963b578900328b59a841cb57055928862d866c8f5e1c0bc6ae",
    "sha256" + debug_suffix: "e7f585ba71e496da368d1655cfaa79879a8de9b3f7eb79ef2b15277184927d76",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "33651fceeae0655b32b5d592717c1459d6533b62680ed63d7c6ce0ef55126a26",
    "sha256" + debug_suffix: "b01924caae9866c5debf866b44bbb96a925a82cfdef63071587192d5c2b7e026",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8eff82f3c07f35b7b0ef00f4bf0c46c92d28cbd700021f266ac51e4f5d992b0b",
    "sha256" + debug_suffix: "99dacea0819203311ef7ee7053f0e1e2c3324eeb246852f492750916301ba42e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d50caba210ba241c1dbb61dd3fbfa0b0e140af814b7f2feae39a2f7a06dc346e",
    "sha256" + debug_suffix: "0218f47d1a9601fb0622fe92687cd3b79648bd095fb11740444497d9c9c47dfd",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "c4764e576f996aa35392f39360be55e148270ad685e3ad612cac42b720ca4533",
    "sha256" + debug_suffix: "a1860ce330a129adf207e51d130c1e8769edf5f77bb94c7f08a4fa37708dc314",
  ],
  "kernels_optimized": [
    "sha256": "f1b0543ebb9ff58c601f3617e9ea89d4bc301a3090d3a1f0c65756fe8cf6e0ac",
    "sha256" + debug_suffix: "fab1d6940f61080d3194a476162717489ac101dbb39ee938d842fe3645a33933",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "bae3ce97eedcb6c70d7c9618020df487995ee34995cfc3cd04fb9d2cbd0acceb",
    "sha256" + debug_suffix: "0597527fa758048db657cbadb38f0a090d8483285db2bdd3803a72ea1b00106d",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b4c7d667c494e85e3a0318946b441e38fc97f72ba5283ab06bfad8972b1a2d6f",
    "sha256" + debug_suffix: "499ff18ab9193c5d4385334bd493ecd006ea38326d27b5333d1c34548b6c4874",
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
