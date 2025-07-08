// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250708"
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
    "sha256": "93b3e0e51708607d0fa96baf335d99fbb3319ab81960d2907573d3ae30df1f6e",
    "sha256" + debug_suffix: "1573befd9de6b795fbf17d0ef0b04bc58ba743bd2c9ea14ca34eed1ba0222029",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5db703b20db5af6fa22d08cf65cb5941ed55427bcb777f0dcda708cdb6f3f358",
    "sha256" + debug_suffix: "7f182b4f11c05ad9bd43824e3d7901a49fabb24cc336e5232390afab83300f0c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d1424207d175f133cc049a7e0a88b3fa222bbd6333589408893c56fb2b205f90",
    "sha256" + debug_suffix: "2a0485bffe45cb4174c6c15128a9f427ca1a4ee9a8fe914d8ac68cd11b300d0f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "efd3a32be71d412cd08c685b32442946f31db934e1287de32ec30ded11e2817e",
    "sha256" + debug_suffix: "c0d86d0c25a8574aed1d9aeac4fbc69bdc49299e547d38c476b02252f85533b8",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "3234fc68c6bf2ba3d54d3b42b7eca9f8d32c7a0ca07e5b4ca8d3bba138a54ee2",
    "sha256" + debug_suffix: "048ff6ad28f9b71c2384b733461dfd57993e896ff64839d3b4d91c7870747ac4",
  ],
  "kernels_optimized": [
    "sha256": "b90264a8ded93ad5f34327a8793c50fc797c4f9c1a32af8db15ca9ec54410d32",
    "sha256" + debug_suffix: "3dfcfcd75cc841fd30505167a7f1912ee2378663ed21f83c16ea24d400a223fe",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3fb94ea2046b3e475777de650eda6cb868dc225e0bf8940e3b058e6c78af7fc9",
    "sha256" + debug_suffix: "536d30e351fab259de4acd8f45f06db4dc04ef930434c01dee45281e15293ca5",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1775d9d63d81355fb86ac4067b47412dd6cb39a65421b78179e5e53c8c9332d2",
    "sha256" + debug_suffix: "12c07ecb854abf3e0b8fdef6111276854c911d56b3d4dd31cbbe0ca90794aa59",
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
