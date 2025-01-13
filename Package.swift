// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250113"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "2204e70fc2f4520bcbad63f7f65bcf171ffb668056c108132351a324f270e2a8",
    "sha256" + debug: "488994693e74470a34864f9d1bd552e978cbe8d33ec7a2a3ce3ce222498ce1df",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7dc1bf425d78b194da17da4253f09d085706d4657958fb841cb3f0562a07cd90",
    "sha256" + debug: "90918bc57e8a940e37c1c68e7695209b440506b97b352ee146de0b667f1925ad",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bd40995bdc70aa4fed2f2ef6e5e9976ff580a47f92ffdf5e70b320a317f89227",
    "sha256" + debug: "c2cd80d6c0bf670c919a687bfc9c152810029d97d1833fddfd6707fff9f0f0dc",
  ],
  "executorch": [
    "sha256": "4fc16ad5b791ebc83d2c74b80db2a32509d598975df83db8262733196547b197",
    "sha256" + debug: "8aecfb4ba9f4c79368afd03c466e99c7c456cae0f55e00498cc8a24932861060",
  ],
  "kernels_custom": [
    "sha256": "75a5384fc1ded79cd3fe640c72089e0ecd34c001f144018d106e986cc6bb6379",
    "sha256" + debug: "5e0c7dc30bd6d7f2f2d00b1773841b050c6a8f7d1ec1c7875d7bccd2683eab67",
  ],
  "kernels_optimized": [
    "sha256": "3d1529f846dd9d933baa4bf267cdec8a71914a493a6c77aa6c2c765812b5b61f",
    "sha256" + debug: "c95d212a3b0e9a36a287097170136856c586e2112e6f25402ed554a4ccc4d382",
  ],
  "kernels_portable": [
    "sha256": "a671af1a7eac8f96608f4f5bd5ae18cfc0e1ba02919eedbae607f55f8fc0b19b",
    "sha256" + debug: "0b58ce51b38db40eed280d76b932ceec6009f7ab7c8502a2ace7f0c44028d728",
  ],
  "kernels_quantized": [
    "sha256": "c21f0850ba5d68c9688f0d110c6501d42a8df54f6e01fa73563fc9acbe08286d",
    "sha256" + debug: "4d3b853170d4ce968e630382bd8896f226ab48cd4cc08f69b909e070d1bcd0d5",
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
