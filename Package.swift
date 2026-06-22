// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260622"
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
    "sha256": "286bc29fc2d5454e82c73ec89fb7f94878a762d51b4d88643df8003ea01d4317",
    "sha256" + debug_suffix: "5b1fff1cd611b017384e594a9a2bda8da02c5cea9e473c2e7379c0d59563ab62",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "40ea10ae41df2edba1e9362dd4b1358540e5b4f8d51ef66e59da74171a6f57af",
    "sha256" + debug_suffix: "26ffe93781267758ff33b4d53671b4576eb3ffb97923208115993550d6cc9043",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1d9701011419a9cbafe833a42aad0ebb3b6a26ea64386effd5360e64624283b9",
    "sha256" + debug_suffix: "71b131444eef5a988f61e5910283f8171f19066451c25b5594c2d2dc62b42a60",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4bc08b7b0dabe7e361602db107440a5e3fcfc82227227db2fa8dec8d1662bf19",
    "sha256" + debug_suffix: "365c7a6b26ff733a35aea5b33af8b37e7db4c9dcf1c67ffb4d693bb588c9ac9e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a3e7d8a5d6050c092084cb6ef446554e8ce5cc29876de67921b53f57bab3420a",
    "sha256" + debug_suffix: "6ed30696899858a0bfb28ec35342c31b4927796b568330809ac3748b3343ab50",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "abf03f19fed0f96d7b7158444e673c9b038570319740d19a8471cf3d0fb45fc3",
    "sha256" + debug_suffix: "4ce6223d63e7e70a390ecc167b51581c144877fc9f66046a82e33782adf71a7f",
  ],
  "kernels_optimized": [
    "sha256": "30cefdcb9a227a5b8b58256e750870cbb295704d747124914bdbce98348eb65b",
    "sha256" + debug_suffix: "7553d8d12a188e1ba475ac0ec0b4ac974d01966820594bfa147307a0b4fef094",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "921b29a18dbdc885a0c68f209bb2e268d1ad249d6f0d52fe6e844776bfd26271",
    "sha256" + debug_suffix: "0304a2acf90fbab69aa31d3c8347c24286d27cc19f45eef6d8cd32e824dbafa0",
  ],
  "kernels_torchao": [
    "sha256": "4ad94868c067f22a5154f2c39901e179141b8bd73c420658e4c792923957d017",
    "sha256" + debug_suffix: "9b16d1b00f7d949e04bb2bc30fc24b67139708d69216a16b232028e43f85a627",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b1d9c885eadccd01e1760ec46482801423000d99182eee9dbd6232828bb8e9d4",
    "sha256" + debug_suffix: "743150f13d0d85ea89d1ab07132afbe56591f3a7ea6e6f65dcc844d0071da3a2",
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
