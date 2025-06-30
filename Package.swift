// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250630"
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
    "sha256": "b9090193677559645e8ac422431ee6571be9089710d5aed3b0495245c9b4b2cc",
    "sha256" + debug_suffix: "312dc44f8616568d564d39ac6b43036cc771a38076cfb90cbce7e63bb58ed045",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3d967fe0772addac1ecd74a517667bd568eee621901987c4c5fe8d60e7319c26",
    "sha256" + debug_suffix: "14f321446810ab832f1da6ad957507a4c9f3ae92d1470445e75f1ab31f5c9866",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4e52239a02dddcc0f401a2b5d7df6cb7145fbd8dd407fabf839f8d4784dcb433",
    "sha256" + debug_suffix: "fdadce9c472f97881fc4c0a512dc482db281beaf859acc0da845887ed5d008be",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "241ee54be298194d2a34a300633cc5641cdd75da27a398e4263420484fab7c02",
    "sha256" + debug_suffix: "f63cd27ec70918cdc3d3fa6daecedcb3a474cebc20a6813931e4568c3606cb54",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "8abe7c7499d27df2b39107d5520ee6209286c2c2541ca4d65abd901994ca7f13",
    "sha256" + debug_suffix: "d7c5765af620d4472c6744806bb9255a5f92c77a335e77df3246f1c1bcd7ec17",
  ],
  "kernels_optimized": [
    "sha256": "58cae64eadf2a01223fd5917c3eb0fc4df72cf503b83ac44d0b5b32fbf73478d",
    "sha256" + debug_suffix: "f86083817dc2d551d68933d46effc9ac4facece90395478764eeb353e010322b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d54c3e671fed247c87a2b605a063f065c8e70fa0f3edfa0e28cf2c363743c0a4",
    "sha256" + debug_suffix: "0e946dae0969f345ee0cd894f91576400e0e17ec41be7ea33c15460904b3f888",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "09f3032d77952cd70ad27d1b28d658689bb4c0d3a2d009cdfdecd27ba4c43b31",
    "sha256" + debug_suffix: "da97f323ab74eb58a6e03af3aefbe658689cf9d9fea1ad848c581a22c2d6acbe",
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
