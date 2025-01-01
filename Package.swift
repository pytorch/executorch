// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250101"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "34839d345ef02e27f67ce1365d4a4d4b93dc40160f8b60d11d180c16958cbeb5",
    "sha256" + debug: "7eec9979135efd6ce92119f365c966e9dcf8bac1d46b427554e51557c013b961",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b66e3c575ca8ebb03d64027514e3ccf6d06d8d871cc60c5455b1241685a89972",
    "sha256" + debug: "072fca1e136d81d86c86971b1d7e78f3aee68052af4078245e8680c9583dc487",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9849375f19eba4348c016d5bcbb3fce9af52a6adf8f31e0a343aaebd469b9552",
    "sha256" + debug: "bed0812f040a072c7c2b24a1ac243d13ff6c4ae6f96f24234680d785d80da49f",
  ],
  "executorch": [
    "sha256": "f50f3469db6ab72fea94f44120a3dd6c460d0e27f934879560ce5c835ac121ab",
    "sha256" + debug: "15cff93b44c7cb9e8bcb5aca85bcea795c3ac135771ddecc2673473e36da57ee",
  ],
  "kernels_custom": [
    "sha256": "f11966e2f152472f5470ea024b4e9bb66f5b50c042d598c02823814c2315d27d",
    "sha256" + debug: "5482d9ea88441585649ea930ca1db9f1f89ee1352dd121d7f4e49a3a3f4b2eb4",
  ],
  "kernels_optimized": [
    "sha256": "77f5df1f739af25040c791c09b86bf3440e710e7acfe9ce5a80fcf2530dc40ea",
    "sha256" + debug: "15ec7000f9cc38862450f5496d565bfafe4ed7a479b9aedd2a7cef78c9c2e615",
  ],
  "kernels_portable": [
    "sha256": "6c297197bdc548dd2ba761f4ef0b8e47d244762d1f27c36270ce0ac2d6bdb59c",
    "sha256" + debug: "ef6dd62521f2cb1f191f2852b92a6e6b96e90824b7c8b68d34f4128120266d18",
  ],
  "kernels_quantized": [
    "sha256": "9a12731aaf70cde5ef76db458d7dde8f9228c479646985a1523ec85c5f041664",
    "sha256" + debug: "36bb23a35fd8efbccb9ac9674201b8480382e32d59b266cccc63020ea020591f",
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
