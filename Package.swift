// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250914"
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
    "sha256": "28c0148a5d942a075a5fe7c7dc7de32efe971d614787bbd273749c78e0797f7e",
    "sha256" + debug_suffix: "0bc4b1c205284f5b81717ab7a84f7e3b8df0cff3d32046eedd834ce60da1e77c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "dd4ffbed010678588d2306cc91669bc46f7c0403481b9854b14e82b85b0b3d3f",
    "sha256" + debug_suffix: "34dcffa9216b9d6ad3634a2a0babffb32431c64d72a842e7f094df431985d6a3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a09cd22b865437d58df74a8b2e16addf7c1676fb362fc9a92272bb1e7b1ba9c8",
    "sha256" + debug_suffix: "4030bc548f6cf8db6a9f1c909a259dc1cd859390093da402ef690de8ff2e7aa6",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b1f2ad85c1f321689f4b6183a5cdec37bf5f503d53ae2b2d6762e1a3476a16ea",
    "sha256" + debug_suffix: "0894976b64d648e602c29e32463b67be492c94193d01dfb12710da1f3944949d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f9cc228869f654eb68fd4a30fe778ea0c2c1930a9fbaa00704e82a829f3d1a4b",
    "sha256" + debug_suffix: "8a2c45c2c86ec0439513da4a0ca7dcd6deb5b1b256a60e7df090ef6e06ddeb32",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7ca1a93a69ca1b0100a568b06bb89b6f47eda8217ba870272af982efc821d01a",
    "sha256" + debug_suffix: "08ad66a83c0fcd4d05f840242e121eae4ad13e831fdaa66f430be19234c85ed4",
  ],
  "kernels_optimized": [
    "sha256": "7087ea8be5ba15dcb53e2ff967b4855fd784e92304fae093997cc33e731050fb",
    "sha256" + debug_suffix: "6f52294ca33b6166c571cc56c377693beaddb785c708fbedb3591e58c45cfad0",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "bdb348c871a2defe09d976a0fcfef97e877a832662ac7a7524b4f05f40e7329b",
    "sha256" + debug_suffix: "90167eb4654a85e68a59619032e8f2028898c66e48c9d345b6b98b8bc1eb61bd",
  ],
  "kernels_torchao": [
    "sha256": "bdcf4ec65ca73cb01358e426307ba1051cad078ab082ccb27491da1baf5b43ed",
    "sha256" + debug_suffix: "f273cba7d0a20316f3ef7bea8ca6a91cc5626702ace237a8a767f9c858acc757",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1caacabd41c5a7af34ac101844a56c08c3a1c296a867e92b5b1275d71aa5629a",
    "sha256" + debug_suffix: "33b20d5ca8829809cc0493d45ce63468b4b3110e3687d4509c71ae0183204fd8",
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
