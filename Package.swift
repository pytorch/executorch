// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250522"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "6f5c8ace3387d6e617996e31a517d461b8348d6e96cccc282bc2a530b5895d76",
    "sha256" + debug: "f2df15e777165b9f6b5a36858da471baf0aafc02f93173d30170162530456264",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1f59ac0b8d17ea0abf7f998266ac12922c4f7049813b588210b5bf6cca7c386f",
    "sha256" + debug: "d117831c3c75eed2e3c9a91b34c4e6c3cab68d1a5283ec3cd4bc5e3282c62fbf",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "35d77f94354358f728d66edff26f35c6785066e0efd59d27db8042353c1085ca",
    "sha256" + debug: "57b9cf485b70fd80f49fedcdd9a2c926ff9a115c39300c2f45862a6f77ade476",
  ],
  "executorch": [
    "sha256": "5d7d9a1da0eb0eda0065f4dbd65d5933f3321dc789e3a815c80339ea649a9ffb",
    "sha256" + debug: "1f5e24384ade4857a05d5a21a2c8b6192263f4b97ee371d48dc5bd54f5a641db",
  ],
  "kernels_custom": [
    "sha256": "7c15bdaf96db0e40ae34c64c19bb941575bd9a4b2b57a5c98a773a832b961c18",
    "sha256" + debug: "47378ef0321997e44c8982cab2be9ab9d0fddea14cd3f932cc4fad3dd46eb522",
  ],
  "kernels_optimized": [
    "sha256": "1c62a27e4097f1bf1475562093d02736e47b04810e2ccc3e5dc5aeda493836fc",
    "sha256" + debug: "ff459b802e2baa10e5a64df4b0f457ffbf71f9b73bc855fbc98f7d06a01c75ca",
  ],
  "kernels_portable": [
    "sha256": "914fa4a96614fe240fdaf28715f9685cb9f25aaf39c8696d58574041c00d96df",
    "sha256" + debug: "4f18a0719665a7f3ab4a4141b9eb9a1ecb45f101f3acee4d13347fb6f3ee5703",
  ],
  "kernels_quantized": [
    "sha256": "333cf66b908d95a827a0a3cfd4585ec180a8586e2e0f0504466f65f501a8c6dd",
    "sha256" + debug: "f50a45ae024396e6bc8f9bd866f34921c80b8c5f95fb8e4db4d1c3c3aff41980",
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
