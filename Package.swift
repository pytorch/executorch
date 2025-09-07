// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250907"
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
    "sha256": "778466546a882c07ff5c07b6b2001ddbc2a5bfde65dc28fcb36bbfe21df6b940",
    "sha256" + debug_suffix: "26724e1b67474b5e48ed52bb9d1e9fad9dca8e2e5618dba43d8f82d31867bb88",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "c030794b5ce59d88c10d940cb7cb82151100416d576ff6f78c908adcd08472a4",
    "sha256" + debug_suffix: "689da7dd2a85eb936d81907559edfa738a5cb27df3f013d30bbd0394e797e57d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e24f7fa18b2758da2cc09352cb45a70401d8ffa7b52026ed040d817a1b136cdc",
    "sha256" + debug_suffix: "bcaa46dc7d262dff7e583fc7ec82d3edc3f0a46398600ae8e72a867fb4840b64",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2910d42b3383a8c30dda2eaa5c315f653c6ef8ee0fd7a8d1884613dd45c8c3de",
    "sha256" + debug_suffix: "b467d8b82c784ce5ba675a1df1692863ca151865f4ef47186f0de127cd5a9bd6",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5e9dfa3915903da1e2b3b5730888bb3be28446e0a7799921b99ad73e5351bfac",
    "sha256" + debug_suffix: "b10560dd735e5355243fe73dc4e78de06311c94e3d774892c002c80417af8241",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7323cb691fbb5d760e76184fc1bf5dcf85f6a2cb689591fcfbfc64732edc1fbe",
    "sha256" + debug_suffix: "60c152f55d16d96700b0852ab0231b737e3b3393dce2f2b44d10c8eabe85eef5",
  ],
  "kernels_optimized": [
    "sha256": "c7e363ce515abd67dcc02d3df9a90d82562e1ec8f1840e006ca5ff34f15ccdb1",
    "sha256" + debug_suffix: "cde7c2dfd60f8459a60b3ee0b3ca6214f977c71d8c539f5638550dbe955a90dd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ea040118563cd816af66c216d8728b74fc96a5baf6b36f7c4afd2309368cd6da",
    "sha256" + debug_suffix: "308b74f2bb62fa753a256692dd7a352e3751b8956d7343ce76f88e5fce3daa92",
  ],
  "kernels_torchao": [
    "sha256": "e36ed4e29ff043ec978afb89f3b11f70b316a532d84367526a56c68e7a6a3221",
    "sha256" + debug_suffix: "dfa89a185459b6518e50355015feb7cda95e5013b41b718f51c6ac64cd4ab9f1",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5b0e0b0f46812c71d8491d5c5a169879210235a5038b74a1a358740d96b90a2b",
    "sha256" + debug_suffix: "3f8e2a28bccd513ae2fc611c27389a30fd32952b1dbf2c1f0cbedd670dd95d9a",
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
