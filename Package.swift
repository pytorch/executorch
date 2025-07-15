// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250715"
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
    "sha256": "e7840d89bf34428c32d4ffa46f8cfa7c4f0c3c324c401eeaed8c1ab61b08cc0f",
    "sha256" + debug_suffix: "80349fa8fda82d1d1fd1a00ca7350399d87cf7c511f05c4e4181a9f07ca1d4a0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2582b12220fbccde499daf57d256e2a63a9f95a313ecb0459ac9d1cb9f5f43d4",
    "sha256" + debug_suffix: "87e5012afbd742a23bc2669bc7ae90e31bbe9ac305bbb5d0342a54e4e3db3ed4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "40e6e151ea317dd2fb72cba53358113da7e454278b215bc7feecabb9b7261376",
    "sha256" + debug_suffix: "73eb2bcbdf4e76e367065b4a7c1060cce909b18432c2dff6b9760d9311803af8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8b0616cb2b9099a8bbc0f7ab3c527564fc60311cee9206e6699a9bcdf610b413",
    "sha256" + debug_suffix: "d1f261c3d058aead0111e65b9003cded5b27adf115b37b37c5255b557bce24c2",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_llm": [
    "sha256": "6193febf789bfd860e2420542cc51ee901580f191bed691fd93c8fe0ecf7149b",
    "sha256" + debug_suffix: "f0858815f10c38c36436b32ef0324306512dc82352262db25bb32b9097d13d15",
  ],
  "kernels_optimized": [
    "sha256": "500bdeca676906362dbb326b02584022e946dfe7e4052767bec188246bdd1201",
    "sha256" + debug_suffix: "3137d8f175d38616e663662a8eb5f14e796690b29122bb2618c2eb6d12942628",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a45c3ae5b85d98145279d3932fd2b47b5d5bd633fb8b175ea3fd73f130bd1ca0",
    "sha256" + debug_suffix: "4ba7599665af3fe17b9a92628e698a189d0fd4f439aabe1bf0b74ad942838b82",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2aa2e95360aa2d094e1921dc786de44a7d8f4a9a7c2c62510f1bba37190c5adc",
    "sha256" + debug_suffix: "8b8036ea8a2a3f0ca1b863b25d678c80f91a1db8c8bd75a37bf74dc1aeb46099",
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
