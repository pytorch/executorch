// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241207"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "a41f7fab95644ab69c4a920c14526a0f3fc97591c0bd3c959454f582d20715f0",
    "sha256" + debug: "25f298fed604c4cebbc91591d3d548d0c3173bcccb8baf9411bdc46659f8b46d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fda48b3526a94519316e76d88c07a9fbd6dfcb5b391df54118e03a6db4361655",
    "sha256" + debug: "305b45314157767a79df25dfbc838392d9fd135191dd0b1bbd8075c7734c87b9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c1464fec150069505bbed6e8eeb9e0d845ef39ee65ef7af7ff18c9ca215a7cf6",
    "sha256" + debug: "edee14f194e268df1b1d48da30ff279ef5e696b62eb5147d9b6adf7c4867775e",
  ],
  "executorch": [
    "sha256": "b25ea1aec8096da6147924604336b0647b4e35285b98d561a5fb694c3113c8c5",
    "sha256" + debug: "46248f2c200c3cf6782fe9a6afdab601659c4724736097df9a79e015835c2609",
  ],
  "kernels_custom": [
    "sha256": "4b70526cde758ef46d1aea91120ec9d3dd4259e7eadf767fe72af7099ebfdc44",
    "sha256" + debug: "11cafe58e849a103fc4d0f05c90c9b0fdf041c46e860e0b1041e3d3102485ba6",
  ],
  "kernels_optimized": [
    "sha256": "0e0f7588f8d0a4550a40fe4f7348e99604b9d6512a988b6b446a8648adb7b28d",
    "sha256" + debug: "6d583b2fb237bd9ea2de5ab51a1882c2fad13617b57fdd85232382b3a8e1a4b5",
  ],
  "kernels_portable": [
    "sha256": "7516916120493fc7809ee64702a34f99dc393cdb7ed112ac72fc9166371cf942",
    "sha256" + debug: "99be8a796a2b8d90a78a32cdd79428d0e9a96bfe6dc9892f2ad62c9b47c5c2ec",
  ],
  "kernels_quantized": [
    "sha256": "db6cc74e0e0a6ce4126ac891aad7ccca5b7f4d961ebe57bec2f321318ea6b566",
    "sha256" + debug: "3f6945256dcc3c77bac06939a676b1e42705b52cb14ad7b1d21fb66cf5e45c99",
  ],
].reduce(into: [String: [String: Any]]()) {
  $0[$1.key] = $1.value
  $0[$1.key + debug] = $1.value
}
.reduce(into: [String: [String: Any]]()) {
  var newValue = $1.value
  if $1.key.hasSuffix(debug) {
    $1.value.forEach { key, value in
      if key.hasSuffix(debug) {
        newValue[String(key.dropLast(debug.count))] = value
      }
    }
  }
  $0[$1.key] = newValue.filter { key, _ in !key.hasSuffix(debug) }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v10_15),
  ],
  products: deliverables.keys.map { key in
    .library(name: key, targets: ["\(key)_dependencies"])
  }.sorted { $0.name < $1.name },
  targets: deliverables.flatMap { key, value -> [Target] in
    [
      .binaryTarget(
        name: key,
        url: "\(url)\(key)-\(version).zip",
        checksum: value["sha256"] as? String ?? ""
      ),
      .target(
        name: "\(key)_dependencies",
        dependencies: [.target(name: key)],
        path: ".swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
