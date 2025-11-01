// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251101"
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
    "sha256": "9937d2c96f83d60d0690f23aa89251e4816641f364aa8cdbf3025bd9a5223d51",
    "sha256" + debug_suffix: "aae6bf2cf14aebbc9c40ec48509193e19f8c8df2e57aecff7d6b1f723251f834",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "98689a5a39326b179a8e130662f4913fba2b1cd18ba20f01615c057cb46b7169",
    "sha256" + debug_suffix: "ae0843fae26b33305a928b9b3433fb53b96a110f766faca6dfb01a44c8421d75",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "11d70f36a6a87e32d478fe809883a3147175f73f24d3066bfe142f37e07e96b0",
    "sha256" + debug_suffix: "6e5b770959d6c044c1a1ea4c2a1e4251b94c390a1ed21d91514184d9b872712e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ccb542ce41a6adbebe5d1595e3a59e75fe0555a40cde340521b7a19be4fcba61",
    "sha256" + debug_suffix: "549fb813a9b510981021c0ee01ddaf99dc3448e1d9e6c3fb7ccdec801386ac12",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "34dd14f00e9955345257009684a3b23b41f204f3fb69107d94f4ceb535744083",
    "sha256" + debug_suffix: "a2935a7d5d2dda66cbb60c8306f12d3746f34f6444a2c98950e620a84fd64c5a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "40444b2f400e18d4abb5e60783fed931913871803cc2c0215ae8855d740ac373",
    "sha256" + debug_suffix: "065daecd9a29ca1d45d9fb76a6209117059bcedfdc67bfa6095e12af1e8ad76a",
  ],
  "kernels_optimized": [
    "sha256": "7347929b307328e61198165d7b83599d130d4b1a278cf0f1e579fc065653d745",
    "sha256" + debug_suffix: "9bca3e0fa41e4f639e6df37984e0e98de110c6b5d15b95a083bb582471f28e25",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6521905b4a6c34e65afdd62fb2364068f1888f16db1b1a66f7f6683284f600b9",
    "sha256" + debug_suffix: "bf397e12e850f2af4af3c1766b867e1e5fd807f0752de893bc0d50b9a12db100",
  ],
  "kernels_torchao": [
    "sha256": "c0afb883bf15c277484be919f55f8175ed909d0a16e7f5e41f4ac15293f2d7cd",
    "sha256" + debug_suffix: "2e32d37d39640ee1ee918c00f8edced4eefeda5233933665db6ff053d3662adb",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cf2eea33792e03e936b293e96f94899a6deb30381ff41442ef5f36bb7ea88e70",
    "sha256" + debug_suffix: "e987b41847af4f82cf74c72fab40c7399859f236cdd89d01a262ade32caabff4",
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
