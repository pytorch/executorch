// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250421"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "2bf34b2eca973eea8b7f2c07e899abdff970a027b402a20daec565c9d971e07a",
    "sha256" + debug: "f484cc18a44a8859add707dff9d86710215fc4e2815f68b2dad63f69e02d93f8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b9bacd948434a7bed628fdaa37e3f419eb51120b6e52adc76e41ebfc86e2bcf1",
    "sha256" + debug: "54bd7227ec8e1ec19df32c51bc8174d118ad87616e502bcbc02c1a935084b535",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "30c8b989806baf5192f7cad522101723166ec9d473c2db7d19e94ee6d8f68a3c",
    "sha256" + debug: "b8eab646003e0479751a0870ee7435d47e1b8b5908387caa162199c9c2138c53",
  ],
  "executorch": [
    "sha256": "b838164c06eda8339e3407ad0ccdf970432b5eebd62dd39c195693c1c4acdc8e",
    "sha256" + debug: "fa3a964c50c48ad92cb05036fb02fd12ac74dc3276da8f1a9b961a3e0203e52b",
  ],
  "kernels_custom": [
    "sha256": "85e96a16afb744552be4f2830b41223d4623dffc96a21696d2a1423820a0876c",
    "sha256" + debug: "43b8a2a10b69a9e5f3cfc5197a4c2874221119dd42716b560d3c829286935889",
  ],
  "kernels_optimized": [
    "sha256": "16206c66f650d2bf04c128400e99d55615f136260020e44179b405175b41d7c7",
    "sha256" + debug: "8635b703f2d33cf25197a294d15cf59318e529a9ff6f97677ed70ae4f094663b",
  ],
  "kernels_portable": [
    "sha256": "56408981b3c6ec79c3e9c86f4a71fa37053970a4bef445279261ba99b28b8656",
    "sha256" + debug: "8b9a3a52f03e7ebe6e15d8ce5892c4d5462a03e5cc2bc7d17e023849938328bf",
  ],
  "kernels_quantized": [
    "sha256": "a8c62c3f9711124341089817e61634ddaab987bd1474a0772e79fba2998b8a2b",
    "sha256" + debug: "78541711f14b096c44033cec6e968870843314d28d25db701659c5f4db33b744",
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
