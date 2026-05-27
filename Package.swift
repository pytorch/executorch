// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260527"
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
    "sha256": "9df70299d340c7cf911e88c40d6596b1fc67c550f751af9f8f1c09c3dc9fc2e9",
    "sha256" + debug_suffix: "b0f697b6878937144068bd596cdb9983278b943638abdb5c0ab524744f0161b5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8da5d32dcccf0f982105f6538f7e71374a72b9ea3575b8359cb2982ecb83cab9",
    "sha256" + debug_suffix: "4c6508808ecb18b640fdfc3a512da260d7192b4a465c49defc035ed48fa8a418",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3823c6d7f576bf65e19872ee60e7617f3cf026904fa6fbd86ea5351c8f953c3c",
    "sha256" + debug_suffix: "f03c78788884bced81b581f1efcf2caac83fa0fb8b1b60f9693b3a85026d504b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "72b52a341415aaf419f59f113e7682b2f55514aa6e67971d7484e5ccaf7fdc39",
    "sha256" + debug_suffix: "051e52a338f0e52a0bd726a7768753b4156f8c1fd3ad1a868a51f8644663cc07",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d07c1a5830d71942af8eaa1de1ac48eadbbba6b337575289d4e21bbe81193edc",
    "sha256" + debug_suffix: "8a3d31b2ed9a1c8a427c76adac51ceb19758a3a71528218511c84a9c4a49df52",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "efd7c7b90d9b3e755d2315c9d7f9dcfe560cee54313ae58083c8d2758876d983",
    "sha256" + debug_suffix: "caa94850c02f1ce3a47f35455d9fb7f1667c45013ab090bb9609bc3ed6a42894",
  ],
  "kernels_optimized": [
    "sha256": "a6af8f035e313a5d6e9e68e3ffea08385ebb893f0dcf59fdd7ee591f0654c2e0",
    "sha256" + debug_suffix: "553b216499ed1aac4a5f41a3d64e350c9d83e3e5ecf10578b522b974efb0cfad",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f17024fa4ab441dd60dbaf12b620ab545e97a9414be9eb1757cb4ea68d0c5144",
    "sha256" + debug_suffix: "c4f84929049252dc279769377e2e4b0466866c58d338cb895b26c8c185cc4830",
  ],
  "kernels_torchao": [
    "sha256": "87672dacbf212723d6629254e135c7ea7a84e755340d878c973490c8358f0a36",
    "sha256" + debug_suffix: "43126fbe0da1117bce61ca139dc3a53854555dde741bffc1bf171e72c38563f1",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "256c21e83cb74906fe956a8cf6af0660226b9e58c3f1da6b2dcc3ffd0e23d668",
    "sha256" + debug_suffix: "bea20084c091ee90860db0f1c6dd965c4ac801ea952ca7a4bc0eb48ac577e827",
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
