// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260312"
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
    "sha256": "d43de3902b24a1f532fd16801e9b67c17418a003928b760fccc69547ee77c893",
    "sha256" + debug_suffix: "04cc54a3cfa1ea84cc50151648478c5d0c2013542c38871427f077734d6d914f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7d990940a75f1a2d3691cd752b9673bee878b2cb3ca544aefd192d576fa026bf",
    "sha256" + debug_suffix: "19962e7273ac669db8f4dddcde101655daccac117900097f07c1e5a0a539d889",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "140681b0801c64f113f4f7e690ff3cb74429c063b9d612769647ababae2571c9",
    "sha256" + debug_suffix: "32109cba7f0c57ce1ec713f6979219e372ff057eb3fd64ac7cc5b583dee7166a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5ad1e0e283226122c4212487fe70975650ff647dd3db09d7adda3735e8e2eb8c",
    "sha256" + debug_suffix: "be316f7ea30378e1e4c2565b1e4c62d163fd9487d14f9084cdfa03e619e1b283",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "07759bb5f02e10b5651c3578d71821628886ad201474021a45e7d50cab6ac37d",
    "sha256" + debug_suffix: "a4215c6cf435fa90085a53ac130e3a793fe325c86ac20b6d558907799946e3c8",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "cd3ea7d7f53f4bf6e9f170b0afe2ea752cb66eb54ceac5a7053561a39d6d8360",
    "sha256" + debug_suffix: "4f2479fe2fba60d70435054f2d955f55469f9f31c142487f1001cbbf93c9ee63",
  ],
  "kernels_optimized": [
    "sha256": "2dcc38e2ab5ce99f48ad7684537b38734fdcf48e717402a0a9b96d96556e674d",
    "sha256" + debug_suffix: "06396dfd8b9c5124c9309a547e4ba70881e84e8c70256342a9799ff998796de8",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "66e1f6d637c8d5311c437023f151495fb3dc5ac0b0738c11483d0f4729c48127",
    "sha256" + debug_suffix: "bc5abc1d33aa8fc4a8c383adeee41e3a65640b8af0bbb9f23bfa188137f50624",
  ],
  "kernels_torchao": [
    "sha256": "965e2cf44bf1428af82cad15fef88df4d0a7001d11396b23ba4de6f70e0f5d8c",
    "sha256" + debug_suffix: "7e4d1c9c82ae2cbe0885595d618ec88acece7db5f7a06704c86f156b917524c8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6dd7b47f2dcf92f418b8718aa3aec34e0d5400a7015d4117ecc37a60d647de10",
    "sha256" + debug_suffix: "a9bd454b476fe78007080b47f6071b140229e3ee5775ccc1e34cf5a26e760bbe",
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
