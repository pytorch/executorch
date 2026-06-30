// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260630"
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
    "sha256": "853862fa9b59c1d3dc88da2a7529bca9203e8cebbc56e7030fe25d3c49fffe9f",
    "sha256" + debug_suffix: "0690adf101571fe60b2be87d4f04cde2a75d0e3afe38839fb0c80df7a8c46ff1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "77b97a65793ce5b03c588d1eab779f2af137034b7a7d75a16bd95f0373c3931b",
    "sha256" + debug_suffix: "b169b5a7568b790db997b83b0a0564d802be58294542ede0b0ab959a88e3e390",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f21599cf1945ab5fd9affc7751eb27c66790fbb4b2a8d83cf803bc2e1b8b0523",
    "sha256" + debug_suffix: "a60c5d990cb322c774e0d8439f3698e51e2f7d57e7a4496851e0b478c4a73c07",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9215f00d1107932438bea43ad094fc048f9ae4bac923bfdd08f38f4102cc12f4",
    "sha256" + debug_suffix: "c7e19782906326cdd914eadc8558df39224b6c714cd949454ab2d08a1550fe5a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "252f86c4fb788cc2431ba5d79462bad15010d3e3d967fb76758677edeb591fac",
    "sha256" + debug_suffix: "e226734e1aa5fda387f58e836ed546b10d63744e982db7d101d639c005f9075e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "24efb56b93bf83dc11ae19574da25e5eefcc022420d33a417c07cece99d3eb72",
    "sha256" + debug_suffix: "463d125546c04ff762bae116f1165cb2037b8f29ae3211afc6769b91365a61cb",
  ],
  "kernels_optimized": [
    "sha256": "ac0dca570bc8ee24c7725fc33fd677f66e4fc8b8a7975827eac9f86e3ba788ae",
    "sha256" + debug_suffix: "492b2a2bee15f6b1451a09d1008dc4255f69144811dfb45149516e3a21d663c5",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c2c4f6f946bebff3b568f13b84e6fb6f6d8dc41d818cce084d16124ae1dfd453",
    "sha256" + debug_suffix: "c36f0f7cf30cb105ca3534d859321be14c6046fa42d9ae91413ce0c1c1b04ad3",
  ],
  "kernels_torchao": [
    "sha256": "503ef0d5f195b8448cd5838fa9651b85e4339eb8714ed865e2a1bcf3d4cc7e5a",
    "sha256" + debug_suffix: "666a7f34c9a144bd515e9461e6eb5c71b25a30ac6e1d955aaa756629d3cc8869",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3590d162e4889269da7ec38b546a0e6f9212b601b306ba1ca920f8b613c2e340",
    "sha256" + debug_suffix: "26b10449cc94b8b541747b02d0d4092155bc4719fa68dae85d7a54652d188a1c",
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
