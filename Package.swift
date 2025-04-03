// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.6.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "4340abb883e9f7fe132c414bff3a77212c1824983685385a903541f413c4d71b",
    "sha256" + debug: "f9a2e06a383a4d3f6c613499c529df49cc3f1f6cab55a5660ed0f15df5124884",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d4001714310b996f41a7310a92ba5b06ba5121c4845f3206002f77895f11134a",
    "sha256" + debug: "7998639e11f108fb57e77de1e1881274a34390653248572b53cb20d917b209c6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0870dc162bbf207c17a5d76518a5e3f1729e387c9825658c99777381801f9884",
    "sha256" + debug: "a9e9dc1db920423580f86b8758a3dd2a8483eece475deb8c6e5bc55943ecf3ca",
  ],
  "executorch": [
    "sha256": "fb2a6c70abb7689ce81767d2f1b17173fb0263414925d948989536a334d23750",
    "sha256" + debug: "7ea2892eee522a2ae148c866a38e12694960984cf59c6195900c62584226b59f",
  ],
  "kernels_custom": [
    "sha256": "a85ccf909e17c32c22cb97c71a309e650e19d20b3e0b7eef00c4b74c9d9460cd",
    "sha256" + debug: "ef5e4b08d1af0b1824cfb1add15e05207e26d4962b974cc039b34f4575ddccbe",
  ],
  "kernels_optimized": [
    "sha256": "209487e9da5740f109e48cda476f39434a09ef0e1a77ffef9d9cebbb45d5f545",
    "sha256" + debug: "46a33abd41ee8b1d475d2a016b71a04c3a76666ec6da9065c2df431975e580bf",
  ],
  "kernels_portable": [
    "sha256": "a66b87d7fcf20b90c821ec129f02018ac3d91cdf88be9d5ab6c971216e618425",
    "sha256" + debug: "36cb510768b94d5251cd8c31fc79ffa4c3c529ca989dd1e15174b84b8965ddbb",
  ],
  "kernels_quantized": [
    "sha256": "bd48d8bafafa15f007cfdcb67d6a551934c888c0e59cdcc7df076dfade7355d0",
    "sha256" + debug: "c3dae5b62c710b7fc829aede3d94254ac2b697c58147a445e1d17fbf0d7ad338",
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
