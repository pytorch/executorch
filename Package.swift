// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260719"
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
    "sha256": "e3589f6ff336d9fd6d2d85ac37ed1be5378b2e6ac3069bb081cdd2ebc317d3f8",
    "sha256" + debug_suffix: "b66de05879b8a295759c691123b5c457833fc6d8cbcef27af13985b5e1e0be9b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d15e94261fc6f3b979c284d10d59d849049cb958076e9604d67c44f110e2ef08",
    "sha256" + debug_suffix: "b966d7efd27c363f595a1cae9acaf0ddc6f6f835262cec93f5dc34d21ccd6ce2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f46b9dc244aad066fe9059f86acda1c793fcee20cf1c775d8b165ffd535cf2c9",
    "sha256" + debug_suffix: "1328a0ec8f38eadf1833a4d08fdce93fc144e7564b00806cf529ba2df07e15a1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a41672fd702e0891109167b3e8701c407136ee62d4f2dad4f970c4946e6ef317",
    "sha256" + debug_suffix: "f61fcdc2267eb32dabc4f9a57a34a4916df1eb8d094baa9494af3db4a64c1a07",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e30039d952da46f3d4ea3803fc5339b459dd6badb08eadb6a0d633f9997148bd",
    "sha256" + debug_suffix: "9b9b985b308eade274d89691a69a18b0f4797c1ab1bc73f09eaa93da0d3f210f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "afa5516871682d7d6b7ca8514eead731daaf5cabf81b92fea1e8d0e7894b8cf0",
    "sha256" + debug_suffix: "4d22a8ab8c75e92f362f9b9c1db2fa4d292b203904ca9f63b2357e3547ca92e6",
  ],
  "kernels_optimized": [
    "sha256": "fe749488a3ab7508b0a3836ffd587e06aa42bf328124dc514afb4a7b64955095",
    "sha256" + debug_suffix: "604edf628107df83560274ec775824b69d16ca5c461fb09545768f7336750015",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "834cd21a071a826fbf736a0872259c2f3ffff648fd42efb6f7c6809a207fc093",
    "sha256" + debug_suffix: "1e8f71857c9f958ee14b5b7006994eb7fc22b791a304c37266316f4cc73ea941",
  ],
  "kernels_torchao": [
    "sha256": "77e93a80f6668376d88883870e0db21e7a6ebe7ee370c6458d2b957033d1eeba",
    "sha256" + debug_suffix: "25bd1639f625419e54e50b8918f1a2ef86b4679d9000a07e867abcdf5a5a486a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3758114fcf98201a58ebeb1aa5b74ac1cca122f265a4eaa01c37d9a2285ee6bc",
    "sha256" + debug_suffix: "679f0c2ae3ba191bc03f27632b7fc34f6c8ea69985b12bf4a651e5ac196a2021",
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
