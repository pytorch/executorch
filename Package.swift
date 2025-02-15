// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250215"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "be66481f511b41e30497fe15238e8d2c90dcb159378cd354d730eb799ed4a7a1",
    "sha256" + debug: "70ef6e5117f1476084d234e44a4bd21737109efad4bec7141a6b8696db38eb42",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4bc7bf4842b329b53160c7fd8c769807cab513cb3696940e5fd2c930a92838e4",
    "sha256" + debug: "ae661d9c2874575d03c3b5c145d5fbbb6f4a294b8dc7243d6145258b9cea4f9c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0b29d8cedf4fd8cc5795c385574f5800d8c9ce35c05eb48e11248257d285bda8",
    "sha256" + debug: "cd44bb006f1ea6b95b9914223788d8c944a2c2adbc3e0a4a7d79dabfeffad831",
  ],
  "executorch": [
    "sha256": "088039806a20f9a91291769bca35e159b9da25a7c999af1d5da01dda542b603a",
    "sha256" + debug: "17d727986dbefae1e558554fbec1ae504c3091e3b2f3688469d50579881d5f85",
  ],
  "kernels_custom": [
    "sha256": "c803d7bad81275b3f880bf158a463834f37202e60a62d87550c54e01b6169f95",
    "sha256" + debug: "a2615204bd283000045deb36f98ca0ced94f67adb5413f11eebfbf5046b4e3ce",
  ],
  "kernels_optimized": [
    "sha256": "829ccbdb91fbb18157d15dcdc436e593d3d07143fb868248ddaec3149839d43f",
    "sha256" + debug: "baa5ab610ee8f0e4bca0a4707b5c7aad004335d741adf7df3e8c78be554b49d8",
  ],
  "kernels_portable": [
    "sha256": "2acf571cea9b76de4d286b8c472fc1252726f72ef040db94682d8ddb010bb6d6",
    "sha256" + debug: "f9cddd3eaa2e9420304c72f2628302f58bcd39bd2ee63c09ec83e660fc4489e5",
  ],
  "kernels_quantized": [
    "sha256": "6918aa4acd289d8d16eede92c3aa276686dd66979aba589542430e585d9b0d8b",
    "sha256" + debug: "ebf5e7dd89897476ef8158a2959b8ed5a7fef58a4044de7a59e9fd5f8bacc7c8",
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
