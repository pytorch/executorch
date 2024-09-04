// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "54c3fa8663870da7d3761424f9ef1d0b47e98d078eca31d1febcf1a2b5f66ae8",
    "sha256" + debug: "bfb1f05f01783fdcd3cecbfde2fb83b6ae5dc627d7cc5b6fe7495d9346c83eeb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e8ea47733407d8fd00522ff6b7f17a46290f217462e72ab86f8b6b34141c6c90",
    "sha256" + debug: "fe39c95986b5b722618685a2e25b3ec161f89ac8f2d157ab77e35164dafc5e08",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "58f66e6b8ff5c9048adfa6577fe7ed927f19869e6f912d030e4ea1166f0e27bc",
    "sha256" + debug: "bddddd252696f65a65d77398f1148c81cae667465a04a44c6f3acca404fb1340",
  ],
  "executorch": [
    "sha256": "97841c38c4d89597cea31b3d0be6f83e23024268986f89f23d86ab537d9a468b",
    "sha256" + debug: "a5ba5f595e4a1e7e4827d835e9a13383e417cb90074e25e9fdad77c89bb94794",
  ],
  "kernels_custom": [
    "sha256": "283a8d0e85029149f06fa604230dfe3e8c267ec4516b00519c9353ba0ef68b4e",
    "sha256" + debug: "d3f9cb8fcbab6845a0d7112c07c7e2ee686dac8accb5454852501b9d295d9350",
  ],
  "kernels_optimized": [
    "sha256": "e54a455a5af69fd28c1e0dd6a9032af9c537b9c556524a318bc7e4d44d5edb3d",
    "sha256" + debug: "8984a1b1c03309fd4d931b54313f6747045a7d22abee38edd85a0d684107c12f",
  ],
  "kernels_portable": [
    "sha256": "f68ca6de5366d0181411e8ad85a20165fa137c7e97256cb603d0584ff8f382f3",
    "sha256" + debug: "77e7a29c91ec556bee86226a503e1f4ca4abc728dc1f5efc14adf43f55c666f9",
  ],
  "kernels_quantized": [
    "sha256": "503dd27248233707d41e5c5ef7dc805dc6215985724c3e457ef87dfbd39cb306",
    "sha256" + debug: "79bc48f166a3eff310589257e8961e19f4be625b2a81656cb202fcfbd6ab5edc",
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
    .iOS(.v15),
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
