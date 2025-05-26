// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250526"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "1a3202ad95894c423227a50e4d12d1579ab940b27cc1951a032ac6ccedc49b81",
    "sha256" + debug: "f25875042d2e4594b299301024cc2483cd5b589cb5723a325c2cd7e698f2b776",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "25d937d5fae3541b57d0d0826e62256dd59a571ffccd50df841a80ff24a4f968",
    "sha256" + debug: "c1c8858c9eb9a5b86f9683162d4dec5288e786719e60ea1c7b3d163fca094394",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e26be53fc332bf69e1ad52f273a39770b6bcf06cf00c4a5af0ad02ffdd45b5d2",
    "sha256" + debug: "4dccd0a331eb55066163811b0b5013eda929eab1702152cae3c9e435b92a6b27",
  ],
  "executorch": [
    "sha256": "27da2b42be3d8eec58e97864e89d1ad203cbaa944e51a1343e580f975db2d8ee",
    "sha256" + debug: "e110e2da078d5e61f4f6d1f44bd24a36dafbb0a1f2a79dbc4dd512d94038c891",
  ],
  "kernels_custom": [
    "sha256": "609373368baf47f8633c85f8c928745a85ac10b02c15d184010e0ef93ad6521f",
    "sha256" + debug: "30ac19f2afa6f11e18f2917819c7c3d8bc7f6551eed1893567961ab02355115a",
  ],
  "kernels_optimized": [
    "sha256": "77d767258c04cb735893a9110431c78bb428c7fe61a48dbd3b9d2dfb9cfdcb6e",
    "sha256" + debug: "83b4b3bfeaff8dfe5b635a9f79a105c929b6ff4618206e9cbaf61f4637599ccf",
  ],
  "kernels_portable": [
    "sha256": "ef941d0e97e63c28d12ea246bd7af9787327d02a4f1547cb6c485d9a6e0d2dc7",
    "sha256" + debug: "55a25b320826f7404bdd394baef2aec9a1ae71bd8426f9f2e27954e19c9018fb",
  ],
  "kernels_quantized": [
    "sha256": "dc7a45924a34c34f47caf4acb894aee38acdfa92dda60dc94b9fd792c6c6865b",
    "sha256" + debug: "b71a34fffe5e2fad286656e7c8b4ac7ec49d96a64ac96a09d55e85af97310894",
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
