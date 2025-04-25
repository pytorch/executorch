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
    "sha256": "2b7acb66e11b24b2578ec571919e6d7135f717f0e6458a89b7268fd5ef1cb327",
    "sha256" + debug: "b0423340b044a55c837d55892a3c2878a56e0cc6334a10537189336c34b49e9e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9aae4bad0274b1fc03b7266979f18d7feed9f09ac8ee89091499358d2725d2b7",
    "sha256" + debug: "60ba786d05abc137f96db04d662c7c15760b42e0589769768e27a81a438515d4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "78e49c506d4e52d7495af9b6647cfbd6c2bf49d90fe2fa663a0a46006115d1a1",
    "sha256" + debug: "7ddbaccc21cc6777040a433a85df5fcadee049458d231d1027a48e156cbb0953",
  ],
  "executorch": [
    "sha256": "e1edee971876a7db448b0c3a4f077965d879f10c347a1a8d457de6d2cf07267e",
    "sha256" + debug: "eb047daaa5ce1d6000ce5ed57ddfdf80ba6e0484812ceeb89dfd4d57e8163fb7",
  ],
  "kernels_custom": [
    "sha256": "53937344949bd3d54d1635f77bdc13690d09087d86869bfd423fb9b092679020",
    "sha256" + debug: "a8d7d605816673a2b5c387754b8e1753420dec5a14c6338372c05d8e17165f9b",
  ],
  "kernels_optimized": [
    "sha256": "21136206b97b6f2dfd85cfdaefb5dbf63bcd192de0ecd8df53b25a5b4d70c070",
    "sha256" + debug: "be586812ee567c9fa2b959745a0e4e31b33981fb9fce672eaee7e4bdde157343",
  ],
  "kernels_portable": [
    "sha256": "86584eac738b99edcc61063799f093623e042bcdbcc828ec16c74621b955a662",
    "sha256" + debug: "d6cae50e8a3dff6d9d8a1ea752f80bd4c85ffc199ff02cd4a3029075cf5d626f",
  ],
  "kernels_quantized": [
    "sha256": "be447b2fddffc9791562a74192fe180642bfc86204502c5a6ca6d06034981c53",
    "sha256" + debug: "69e15d51045ccee079aeff621caeb5e32e8608d064151bfe737bbea72329698c",
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
