// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250506"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "d5124ddb61560446763c0f2a528495b717ac919884f0ff41753ded54ed60e569",
    "sha256" + debug: "b2aa170706fb6031851c68aab721eca4da6bc9731e2d9d82a1c60466e3dc6ce3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "79186037ad791cd47b5c9d2171423eeeda872fd5b70443a4aa1cc05dfd985ed6",
    "sha256" + debug: "b65948aa60d3ef762709cdf321d0ac5696f24f0cac83f34d4c9090a53565cdf7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bbbf639c01e8612447fefbdca9652da71c2531d213e9548b205f4b0698adb976",
    "sha256" + debug: "3df93e42e629761db4bdab8c9b25c0b81285ce6a1e21dc513f34a68e661fdc90",
  ],
  "executorch": [
    "sha256": "d5d17e71686701f1cfc66b077949ba4c27d99ca7588dca329f6396f97bb9c2b5",
    "sha256" + debug: "17cdc28958d44e9ad2f77c923d2d32b10e2414ecad7806c7fc8b253c4911c70f",
  ],
  "kernels_custom": [
    "sha256": "75e8dd176bd3e677cb0e5a22e784fbfb16802a0ac26956c94733049fe92c4e72",
    "sha256" + debug: "c91efe91b4ad41c280bd7562e847464b423df24bc54431a7940296692ae99f4f",
  ],
  "kernels_optimized": [
    "sha256": "62814ca46530f8ef113f634ac1a582235d04ff31a2d8021efd067fdae26b06dc",
    "sha256" + debug: "ee56a9ffb84b4b681ade5c78df9958f177e55cb60085e7c63791d5ca3ccd1ce7",
  ],
  "kernels_portable": [
    "sha256": "7ef6b3e9a458b9ede3f6a4be41e5c3700739596e77b6bc68c69a1f58a74135af",
    "sha256" + debug: "35b91fe73ab88d337873ecd222e2807efa39c271c620b0274bc1bb8ff5b3f129",
  ],
  "kernels_quantized": [
    "sha256": "56b271b6aaf1c202ed0a1280b4282b628e6ca964ea8f725a267952ba4a999919",
    "sha256" + debug: "118da7f0537dadbdf3a26a60dd703581993ea42faa1fc41cdb6aa3883473abed",
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
        linkerSettings: [
          .linkedLibrary("c++")
        ] +
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
