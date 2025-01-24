// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250124"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "95f2d01011338681d2e5ba6ad42e47f126de968d92a45dc0d840f9a5b49c3510",
    "sha256" + debug: "8835173a2cf57589f8622b7c9d89b3f496cebe2f270e60a62128c4992888a6d0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3d1b398b67637e67c9ef53e17418cdeb20897a17d1dada7e0f084ec7a84aaf8d",
    "sha256" + debug: "f5bed4f6420ce93303beee7596e601b36b885dac265497022b58550cec0aa221",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "25848a7667b403c139801c6dc7cb1cf821bae23b699b674de39ea71a6f3a8139",
    "sha256" + debug: "adadccd2aa9a1f0a488de70cec7305559943ac81fe48f1ce87a3f5cfc745f0f4",
  ],
  "executorch": [
    "sha256": "aece35ceeff308bc3912d90390656487f85743ec692300f65ce1326b2cf51a95",
    "sha256" + debug: "fb1c588546b9369751197928ab4c996f81f27f590e108a6e4d72ae2bcc0eb626",
  ],
  "kernels_custom": [
    "sha256": "592bd1b2825f25c5ad899cc4e2295bf0a4e945eb39b912d21d3db125cb529614",
    "sha256" + debug: "689af31538f1f61de3066120637ac881ad88aa9db58de949ea64827929ee8709",
  ],
  "kernels_optimized": [
    "sha256": "491bcdc4a96d92053a773fe1d8e0446e6f53f718620063ccbda613590d1345bd",
    "sha256" + debug: "d0ce3b088e0582e4bdeee604296b97ef10ada5833c02d2ac5f11db336f6a1ecb",
  ],
  "kernels_portable": [
    "sha256": "4f85f00af0782f8b1447a944a00b2ad5559849cf278cd521ecda69488d581e69",
    "sha256" + debug: "619b58cedb8da989b463235ef224c601a02683084c15cf534f26291ac74fd997",
  ],
  "kernels_quantized": [
    "sha256": "7261a371959043cce98a0a5bbc208b395c8b653ce3e3e063fb1a12498bab0bf4",
    "sha256" + debug: "4c0de7a50f720837d18925372d0f31e3d380e41f85ee32da9d0d9f140fb35e6f",
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
