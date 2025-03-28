// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250328"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "d03493aade12b5a5d1c4bc1b70866a16b6db7c199a4f4fee4f58b9dd93072b9a",
    "sha256" + debug: "f498ec5f3d2af4027b27b953561a8ee7a50a89c1e32b0fcabdbc944651cf7e9d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "233303b675c08f8cd87487ad47347f770de02e1690401dbc705566d86a507016",
    "sha256" + debug: "45c01146de365cd78fe3f085e810ea3517222fb75cfa165f78ac14b4f3900ab3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d2b644cf423609647bd3d9451fe9a96c4e57fa3396429b88e8ffd400dcdcedbb",
    "sha256" + debug: "bfbac215fc12df3e914e3e9b451a46479b9a47ee5604faf30d399056cf90030f",
  ],
  "executorch": [
    "sha256": "f0ac1c933bc586f3f1c23275bd185fe1e108a20122afb5240682a0d0bd4dcf59",
    "sha256" + debug: "8b847819ee0405c077489b7fd5eb32a3f5eb88b46edbcb3f4259c6e4228f513d",
  ],
  "kernels_custom": [
    "sha256": "d30852d9b3d26e9c68ceb929ea95b711c9948977d0a90c80285809c9f656adb6",
    "sha256" + debug: "677b7bbe5fb2cdbed23a09f423b2105019748f994d5662c7be746dde1e324e52",
  ],
  "kernels_optimized": [
    "sha256": "067569e82776377c968cdae890ecda0d77c2c120f42f4b9561ce83428e4a3c96",
    "sha256" + debug: "c1c4017d43ffcfc9406823f5a1d7309891d8a98b3e9ea4b092347565ec18b8ee",
  ],
  "kernels_portable": [
    "sha256": "28622f84cc0d014a83c9bc11bdaebd77c5c239b3963b89e6393180eba1674b48",
    "sha256" + debug: "81e997149a29ca7ee3117c633a6d85aef904cadfbe05d604e803c8dfac3dfd7b",
  ],
  "kernels_quantized": [
    "sha256": "c8042f5b38a6f3ac9ea1530d5d52ceb845ec8f7e39fc9ada3268fa3460a43b25",
    "sha256" + debug: "a3aa170b334eeff61f7115a3181781794ca607be0b06cb293eae78aeb92bd201",
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
        path: ".Package.swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
