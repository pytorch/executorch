// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250629"
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
    "sha256": "2953f60f52232bb449c117d8bd8610517a66f0f7a0e1b6d3111a4728f33b966b",
    "sha256" + debug_suffix: "bdc167bf34e6db52530331c027e4297e6799c8eca0fb7e957b5f08ed8aa5e8cb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e5b0f07974b4f583512254b1338a95795e87ac583eb9f0b6ddb737c078910e81",
    "sha256" + debug_suffix: "db62e8f8d3fe3b4221b398afe4457f00cb1e6f0a5902ee024abec53529b5d433",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b39877f4929a70254cc77e644fa9221e7b1336bcf38b68cc21560478cb8e4752",
    "sha256" + debug_suffix: "1c88113d9469f08660c9e2f23598b7679c8df36e54f9eb2854cdc47271c2d2bc",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9680f6446051b07b5a90953f047a9dd96e3075f750ba5da22323c05b5df40b20",
    "sha256" + debug_suffix: "b5a033bb082c9504664451d3d4f46521d6723e7fd1adc4d9cbe7900bdd4388c0",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "855d78f3958dff70680777cf36689faa0e606f35041e847e6b404378fa73f185",
    "sha256" + debug_suffix: "d86bd9cc7e23536e096b74e404efff7cf8d47802572d27b96c33668c80575e19",
  ],
  "kernels_optimized": [
    "sha256": "e79c75d8c3e48c9b966fd7ff531d5f5f1d098cf150d04029da466e32f2e60c29",
    "sha256" + debug_suffix: "417da0b5d70560288dcd861864864e9f2f2a8892feaeb932d08489b646344ebe",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a675a36de644b11fea93286de54cbd7f08c0ce50e8609d250d2c7d4f771d2df0",
    "sha256" + debug_suffix: "c2dd075b5c2127aa82e08880d8cd9342bebb341239054a5c1459adee9044205b",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e1270d1685012cdb5db9b8edbde2783cc88b420d897ea82ab2aef8de28307c2a",
    "sha256" + debug_suffix: "ee937961d110361d361d88424f77103a3c31748e66001be61c7fe20735bcb4f1",
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
