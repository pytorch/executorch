// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260119"
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
    "sha256": "ea6f04aef980bcdc3935b9f1ed9eabaffcca4dd05a7f633c72e9edd2315c5b1d",
    "sha256" + debug_suffix: "fdd08bb48ccc47f02c4f888420af8c7be21c64eed483d6a6ab61441dfc2929bd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "060d46fbed5107ba6ae4c094b7ce1b1dae5a6dff2fa24c84fea1d633c18735a7",
    "sha256" + debug_suffix: "0ca0616d6c54f8bfaae1bf25ce931d3d0dbfc04692b63bbbfefcc2ff9f5d4f50",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3e69b4086e1ae618c25bb5aaf1b808dc352cc2e918c963218c967d7abb817686",
    "sha256" + debug_suffix: "ef961988b9e6fb214561975e697b581d7354abdc2a26e40162c8b9afad44a1ed",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "cc5de143dfa400430cdcbbdc144925b8b00957c50666cc3b8bba74c5717949cd",
    "sha256" + debug_suffix: "a96a42c15329e8555ad89f60342787672cf0dd458e5bd774e37e196cfbce2ff7",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "37ad1c98f68328cbc4760b9b08cccde6f796758bf17eaad5fc0024772c39c20e",
    "sha256" + debug_suffix: "fe9a4734cca130b5734da4d03c7d09f5f854e98d9909c4845c33dcf3ee339e27",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ba5d9a520dda89374453c067c4131d0ad9cbfe4ee88e9183736e36b9c4257c62",
    "sha256" + debug_suffix: "ffb21f45fdb1df810fb19f19d73f172157c90ae89cfbe191729e9c8b76d977f7",
  ],
  "kernels_optimized": [
    "sha256": "379804c0b0db6a76e55b6ead06e9c4a49598c510e617c241948751ffac19af86",
    "sha256" + debug_suffix: "2b66425439be70b34df86dedaea1e45d3aa8cfda24c988622bd3f46558e980ef",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "76af1519d4ebc5abea930fb7fcb861d341775cb8c5ad2fcdbb25fb640f00bc60",
    "sha256" + debug_suffix: "5ebc4a024ef55197e825232b0f836a16553d18e2c58c5b2d721fed85731c5df9",
  ],
  "kernels_torchao": [
    "sha256": "596fcfcd38961c23d90c66397286460d5825ea26d976a76f52f76418efed1154",
    "sha256" + debug_suffix: "f6c54ff85010d4a948c77e83176feefc7e5e80ab93bc8c3051deba4f43890c7b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "78a68f1ce39deffa65392a0957b641abf1de52eb6618c402bc3b0f335f1ff9b9",
    "sha256" + debug_suffix: "66917d1a0b8e78a7fb4a1992e6ba638589dff420d84af8f5fc64340278959597",
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
