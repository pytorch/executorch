// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251106"
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
    "sha256": "986538675640fe08b6eccb97c911056073f5fd64de548f442a7349ce771098cd",
    "sha256" + debug_suffix: "07d05c69cafbbf7abb109b8619d63569ec302cb7883e6348599696c07ae73bbb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0a92b1e6eb07a9887916b1275ef94939788d585a0a95a6a02b7b405a6d40050d",
    "sha256" + debug_suffix: "1d644cfc8b51d06d292fc44b3bdc22379cbf85e8fc085dec305d446290c7b86e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5fe11e817ab3ff12aaef36488e2653e8eaa2f6edeb3ee21730218913c8213d4b",
    "sha256" + debug_suffix: "718136519e3f19ed04b73549c99c25378c533e5803ea72a3b6bc83c316111b92",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e25b0e4597e131cbde8a1699805ef3841d19d3af2eee994f6dcd5f14ef95d8ef",
    "sha256" + debug_suffix: "9d11d56c06ad8784d1c1a67cac8bce0e4b62f4f7a42f23265b66fa7c6723b3cd",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0cf4f82293106a4b0edf5336e2f768d2cdebff5255b90321c52f21c5df8af73e",
    "sha256" + debug_suffix: "fc690cc4d7a71deb95bd7f2855777411a3aff0c4cb9dc7e80bbc1e732d4fb34e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "fc81c170c5617771f29860fa7e3a6c67e786082b7220bf20bb693d7630ead88f",
    "sha256" + debug_suffix: "1a666404e93b0d949da526ef05790d7ef2ce9431007a35c7dd4d8b9d4011472f",
  ],
  "kernels_optimized": [
    "sha256": "ff1980c46204fb1617a63613696093f48c8363efc9530e6260728c6a116602e2",
    "sha256" + debug_suffix: "966205a879f85d7fdffa02fff4be8ab2212b9df4144ecc1d961884efadaddbe7",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ce5b9ea1a74fd4d743c004bf2d09907d0073ce792a867331c4c84c92ef6da6b2",
    "sha256" + debug_suffix: "1727a952fb6c39f002a7766d34c4fd73dd54c6750c87d09baf0d2b60877e36ed",
  ],
  "kernels_torchao": [
    "sha256": "2b14766d3342681310e52e89968a69d01b78e954ac101af697143170245875d4",
    "sha256" + debug_suffix: "df222f9faf4d52f251b1ab009fa16babc68ce6a56237b5236ed8f3fba773a19c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a307263ad5f8d1add7b808847208db172339d111575041f6caeef96e27157137",
    "sha256" + debug_suffix: "9b9544f98145067f5d936b8736f353d8ac40a39ced4ddfa854577b577b1bf950",
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
