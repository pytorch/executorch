// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250122"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "d8045c97ca4a0ce7bd6527ab501d87da2b246331e89d648c49e6188bed300ec3",
    "sha256" + debug: "19c4ce2ae5d091963027b93870c865ad176ed3b45cb63d285f47df9b1391017d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0934cbeabf946b9ca5d11bc19c0760250865377fd938882ce4eb7099182f839e",
    "sha256" + debug: "3c5343f52d6ed10d95de391a3e3943cc4aa7832fd17ddf78ba223fd8d5782dad",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "732c61d9a4f1c373f6c25d146f3fe807b4e9abf53e8a629a564cf5339bdce632",
    "sha256" + debug: "f850868e0a4a5fcfde7b08058c41e9f286f767a84eb2726849c9423ea0199bbc",
  ],
  "executorch": [
    "sha256": "28950cf5d5076404a43262e82d1a923589403e38b4674d12e79973511f5a8b5d",
    "sha256" + debug: "bc0719d6ad9a0595c11e454424b3d70c0057657cbb60886ff825a925ac2e1c78",
  ],
  "kernels_custom": [
    "sha256": "0e6d21e675b4fb6f335d184519f41af64b2352b9409f8b9d20f39960ce141133",
    "sha256" + debug: "503a388957e88c830e467fef3ded78571a1bae5b4b97ca335fe5c3ad17282009",
  ],
  "kernels_optimized": [
    "sha256": "9e98e64b0fcb7fdb9dfc5e4bcc94ca186bee4ff07bac234ae66e0770e0f06811",
    "sha256" + debug: "618233f046063ae45c8297aacb9e2e0ca239b8e91911ab19f636da3508564f0b",
  ],
  "kernels_portable": [
    "sha256": "daecf9c6d14cf6137d25c00bf13c0b0c276c6338d31515af0226e6de89b2435b",
    "sha256" + debug: "83060e620852579b1502b6c54e727a19e354336162f7258256ec1eb75715bce2",
  ],
  "kernels_quantized": [
    "sha256": "7d1c87bd57a3ecb778a9157e52563ddc6ed7bfcd699c2e8f8b10494f088373db",
    "sha256" + debug: "9f68980f30ae400bf459f4e461936bbf801ae0da1daebb99bffdcbbb818c4587",
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
