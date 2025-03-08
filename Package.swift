// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250308"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "782e5e8025fa75fda270628b04bd59e860de14bb595e89dce8899fba903bef78",
    "sha256" + debug: "e3f4e5691667172fec1691bb5607ccf593e851a0f6a339f7fc37c0fe65fbba01",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "58ebd1b74b5a35b61de7cdda8d565923911298472dd0763a3c266a2323f393c9",
    "sha256" + debug: "41bd3706fbbbc73a87fdea4d5414e219cf8918276efe70a2158d6a5326114b49",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "30dc5a5282eb82460fc7c77566cc569fd8f5c9b11a356cf7fb20985f5a968ca8",
    "sha256" + debug: "87e1448943febb83547e8aa4177ea11357904e50b89586cd7ff3430d115ccd18",
  ],
  "executorch": [
    "sha256": "8f66b828340bb6ded56d6635cd8a244738cfa9dc3fab8c7687a99886b3a1cf78",
    "sha256" + debug: "9abfc3fb8af1dc25e14de7f7985bdd32a41e5a3fd2d1eaf2fe70d0f80f6b81c7",
  ],
  "kernels_custom": [
    "sha256": "fcb263c6065974b81302bea4d7166850b1f0b5cb904388b6136052cfac335725",
    "sha256" + debug: "285c259890bc1af0f83c57e9281ff13481923b36aca0b327d9aecea459cb90fe",
  ],
  "kernels_optimized": [
    "sha256": "4b6963541fb415c07e349d772119973f270f994d619fec38f848510e38227bc2",
    "sha256" + debug: "3d26d36eee9a4f8b3feac99f6778440cf702bac84682472b7cd5de3e64217b53",
  ],
  "kernels_portable": [
    "sha256": "e5ec276510d99f6ddec068c70cedd9000a278a95ebc18025a300a7b1eeabce69",
    "sha256" + debug: "9e63c020b49f0c198f2f6018b27f6a794fbb32d9be24034ce1cd5496933986d4",
  ],
  "kernels_quantized": [
    "sha256": "027c6b6cc8cdbf49284ea81ff21d216dfd6b3c5ec3f4913b33d8bc491a004865",
    "sha256" + debug: "94ad9810fa6a391309e9fc1bb1e30655e38cbc7120ff5b53da7805e6ba641328",
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
