// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251005"
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
    "sha256": "1112eed5f4f7d093047e5ae4353353f85f9575f3ec05bedac887d130d9c53cde",
    "sha256" + debug_suffix: "554b662fe5fd28227f811e61b23b74dae04d768900f03eff395c9e69c68bdbad",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "90c21bed0eea21c58fc54d63e9638b51cf376669b815e6721f017117c4068980",
    "sha256" + debug_suffix: "bca7c0bcb91a61a08eff601a8280c3df6d0617da88f41ab2c8cd672757b27020",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "da665dd0181b6d43bb34f7bc21dcbf609110060c57fc4944314f672effcd9fa2",
    "sha256" + debug_suffix: "b911173b21274ce822fce4ace3049cbda94ab3c83206d43e104a41918f374eff",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a67cd237192e8b3f3e4a6e133cdf7f4287914d221f7b26b6a751f32d6e420613",
    "sha256" + debug_suffix: "2dcaf082adb5d1a799056e37e11ed48007bdf32d1531756cb80c4723914d14be",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "30575c875d67b0bce0508b34f3c26c818947cb5ea06b38fb784d143dc2ab0b6c",
    "sha256" + debug_suffix: "4a26a73df1d486c1dcfc53fe9c8505ef57e28bda905ad68607eb475b748c75dd",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "55c702f46a6db25bac88ac7f2cd7d4bc3e5a8a3b8464179a6b38edf548371f53",
    "sha256" + debug_suffix: "30ecd2f30473bcf3c9b7320af4fe6f073092bdfe1b2667e607cd0ce0a0cef9b9",
  ],
  "kernels_optimized": [
    "sha256": "5ec8aefcdd9a215f4fbcf2f837765905e62398fec2f9954a5d691e21fa1106a1",
    "sha256" + debug_suffix: "d84c428a0b4f67f30fdb901649952894195fd301ae521c741f449b77ed9d01bd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ab6b0b945e80a2c04cddc906e202b34eab05a1cc4e73e1bfba89336c6260042f",
    "sha256" + debug_suffix: "e2f8cfb8204aa607e504a7083eb4f4dc4bd2544af15e5b0217afc3f4516c0f2b",
  ],
  "kernels_torchao": [
    "sha256": "6837bab235f5a9484f1db3ff5e24a362cff065e95ad6654e7175248dde52ab13",
    "sha256" + debug_suffix: "99d4c2605076c5bce97c3b77bdba3bf71a47f644fa1431172e1b4f6d6baeddf5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f793c62f66f9f84aebe15cbb92d15430c3810dc3c6bcc358779656e447d7bb3e",
    "sha256" + debug_suffix: "8eca604d8a2f60206818c2635bf98fd6061aa974b01d7ab77c16f28054bc21ab",
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
