// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260802"
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
    "sha256": "cac26e78897b5286b688b6a71f76c1faa5b1f342f3d7de3f66482c89e41ff6c5",
    "sha256" + debug_suffix: "77d78ee12289a3c14f04b6f3b5586e2a1bec317aaebb299e1210144f87d612ce",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "61e1bc0913dff7c4b83b814b8e4082d043844f3b608a3d1097fb9d1a920838b7",
    "sha256" + debug_suffix: "65231594a61fd59ba6ca1c55f373843106f69d37b7dd72e2582be694f6724ff0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "77bede412e380bb958e8b63a730d02991dcd3be758559d56c2cbd37f74342cac",
    "sha256" + debug_suffix: "915eff73f6ff1463020d0e25d1c8e190a3b1557a590e88c80514f5eb6cbaf59f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "159dff7f02515985588df6e90c969b7b3e228569ffc8f04756432d7be4431fa2",
    "sha256" + debug_suffix: "45c2e6562d06a10fbac0f8f456b98773dcd46b94277cc1d2db13a9834f1b1688",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "bae330dff4a7ec1f976b566bc851a4697b26f2391641e4b69c2074ffa08c318a",
    "sha256" + debug_suffix: "b373579d8171acb3a62eada2f9060151e79511e823065cfcbe251fdae2a96583",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8d753d1b9a68aa5656178e06ca0e3e1995f866b4b9720a778a442b4c12353078",
    "sha256" + debug_suffix: "21266735ec82b0476dca7662cac2171df83869abaeea7be8f3eb736b4a8e4fc5",
  ],
  "kernels_optimized": [
    "sha256": "9599ce9eb00f3c1cd41020ef7315adf26c074c01529a7354f6bee2bd494d7e5e",
    "sha256" + debug_suffix: "2c9337c7e2263ee8e1b17a6f477ee6962522221cfb475735dbe379a583012eb8",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b94519f94c1e6c65bef1614304dd48a31e0ab04d3b170ee6fb62b518f7de17e0",
    "sha256" + debug_suffix: "b757de416155c86d195602b6f608b10cc51ecee59a94e83f507da2a9885e218d",
  ],
  "kernels_torchao": [
    "sha256": "4ef733bd923b2841f0d2b5d0317e98ef5350f7d606e85b7a0e68e4a458a22aef",
    "sha256" + debug_suffix: "1d8dde4842a32f062e681ae2b7350bd8d132cd625e6c9a3445764abf60e150e3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cac04988b0edb740a9c088ac0804045b86e565f4889e792a31c0d27506465b6d",
    "sha256" + debug_suffix: "663b59ed8e9fd4b2a34039a6f03a3fb098a45e112fe71593858f551e77faa964",
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
