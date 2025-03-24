// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250324"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b36362d63a32ab423b0ced8bae1435c235138d6c4d82303e96239e0ee66fbdac",
    "sha256" + debug: "b63d220e937e567f2a12d6bda358428b6bf6318658a59c978427687a36195e39",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e44d6f5513eb527ad5a797a55a1941e4138f6a564f02f794783838aeb1cba50e",
    "sha256" + debug: "8e92b37fb2e54a69e7915a74035ba5451ea5cab71a0b93a3d13567cdabcfde66",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fef454d755b861bd1fe5de71350c7dff77e45a6b9f66a711902d58b7c71a5c74",
    "sha256" + debug: "621935e274b0059aff4110003cdff8be5359f24b9ca4fddd8d9d5b8bdfba4b18",
  ],
  "executorch": [
    "sha256": "2691090134eab250270378af1bfe239811676111a61e044813248ed2f010121d",
    "sha256" + debug: "42b21cfe3d179874b1051aa7a2377b0978fee0d1a23f744cac5a3b20336fb39d",
  ],
  "kernels_custom": [
    "sha256": "34910471768a36e487d378af8039da3d55f1a2ba5526743fee12a4e2ec675bf3",
    "sha256" + debug: "f044e8e930e6a334b3f81ac1516eea2fee438ac6db7244b696f796cc20142c34",
  ],
  "kernels_optimized": [
    "sha256": "9d088cf2b6f75b5081da5ddccfa190f770b6795eda2f083aa1efd0d10a765fcc",
    "sha256" + debug: "997a432b95280587068c0a424887b7660e1baf73a94d77bc83db5386b3334d7c",
  ],
  "kernels_portable": [
    "sha256": "ab103c4647db287c32254f42f9f3c376b9d6b9ed6baaea0f9b5f7abb26e2e9c5",
    "sha256" + debug: "460e05de44e2ac13f74387dda23a4ee29a81073c19c1201f92b5aee00a4c9972",
  ],
  "kernels_quantized": [
    "sha256": "45a49b3eb460e03d989a4b05fea640806cf46c7fdd3dc03357a486cdb616f8e6",
    "sha256" + debug: "85885af8958661ea20ee6e3dbd11697f104c620825c641a24ca32e676da4d39c",
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
