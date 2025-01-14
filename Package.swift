// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250114"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "614406abb5d859492781df44e17de78a8a3fe970fa8f9b59e2a37edde6fad78c",
    "sha256" + debug: "14bb48472bb0b25f57541c368ac32b2960e7c27ea4a8313b5d30bec9da7e88bd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "696e537e3bb607262fa33ed583fade639c60be24c56f3fb829bda0c90963dead",
    "sha256" + debug: "0f7b85d9f8c5b55c15864739caa5ade151460f9855d39c8ea007a996ba57637e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2300b9546c282cccc3467910b9c60c8b286d549e7d604576572fe24b791a1b2d",
    "sha256" + debug: "517f7127ab644c2f9877e69b2cd11d37d820f3e5c28930067d37edcea8885d8a",
  ],
  "executorch": [
    "sha256": "3a983ef9259f1152b8d9548bcee342510b989612ec26c8442e7fbc74a4fd92a9",
    "sha256" + debug: "bc13e09b88edc17cf4cc1e1048203c144bca6cbf269d1fe6de58d6cae9bbcca8",
  ],
  "kernels_custom": [
    "sha256": "f6f325954cf2cd3c160686ea1bc4031fae1d1417219f9d4e66bc039b002d71c3",
    "sha256" + debug: "3a8fb79b00bd30b302ab9eeb108511429e060fee9401d6a6bee4cadc7351c7f1",
  ],
  "kernels_optimized": [
    "sha256": "62e2b97a7658038f22c1909001a4bce7d792ff504fddd0b84914e954124a99c1",
    "sha256" + debug: "3bc8d19cbe9321f8ac7a97c5872d0f4865f9d6f1ed268ef8bbd48f4c1b414e79",
  ],
  "kernels_portable": [
    "sha256": "2ef5be325b97d50c6026f1bd0853530fdd91b74e8462894485a12583c38bba87",
    "sha256" + debug: "11210201a2e0dd985f0506153eeaca8fd629ff1db683c59a603ce0ca815b5860",
  ],
  "kernels_quantized": [
    "sha256": "6370b6d1879fec281affad4f97bc3e40486e1b764b073b07a565c532ededd968",
    "sha256" + debug: "2763bf58fd77fa14276b3d743dcbc92d12e65110051a2b01f8e474e5725c6bf2",
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
