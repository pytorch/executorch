// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250310"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "fb93097a0a8c9e8aee7db6c15382b1ffa073265158ee731e1b413816a1c9a316",
    "sha256" + debug: "348e5f221dc42ac9ebe12d6a69f699590379906b8b9c59b3e47c76fc19eadfa0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "562df299f6811c81806c6b688c9c2968108e77fc77240da2a1b3e2771ec950ef",
    "sha256" + debug: "b6462be596721ec98246cc381ef952017490a1f8d0296e3440b386ea3eb41144",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "65d8f3fcfd5cb0a54e3dd7f382fc180c85551de4c6f29997ba51d65ff25a2a3e",
    "sha256" + debug: "721a142c1a68b5ef9604aeb890c483f072b65761e6b550bc6e8620bc95d4c84b",
  ],
  "executorch": [
    "sha256": "1a39f37df81ee7001baec46958c858ae0713faac33df966dcae99322f0e10f49",
    "sha256" + debug: "185b771b4d66a53e58111d2f857cde99d266796610f21cfb106c778998333f71",
  ],
  "kernels_custom": [
    "sha256": "439ae236adb7f074ebba5e179834fcba63826f1d21ce918d9165cd684ac55af2",
    "sha256" + debug: "86b22f0af228dfa520c6e2a48904fe03c5ff6b6b44361e5e71ebc3741539e9cc",
  ],
  "kernels_optimized": [
    "sha256": "ddb88c2e17414910537d57c9d9ee63fe6a592f75ab0cbe356fe9383a7903c93b",
    "sha256" + debug: "8ac9118b5efc7eea501b1a08bbe337f7bdecc7e576f67b31be86af27dbb40d45",
  ],
  "kernels_portable": [
    "sha256": "e4229d732d402bad5f4d195fa45f12d3a2fb73ddc0f97ccbda9b16723af32e13",
    "sha256" + debug: "a74f6fe9ed118e47e0630224eae4d5e92d596288ff3ca4eab7f7ff25fb659dbd",
  ],
  "kernels_quantized": [
    "sha256": "dd54404e46022aca2eb75b8ba84df98547cd064f8ae758f82514f3bd2adc4b61",
    "sha256" + debug: "b9cc295bc8b93b27e0d194881d766eb31e183133252e07cfa15bf834c968b4f3",
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
