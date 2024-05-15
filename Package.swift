// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "e32d6b28753f9d187c8d0f64bf602158c896dbaaeed22d5c0a82b4cdd909a58a",
    "sha256" + debug: "4be23e3b043de643cab577649ae473ebdb6fb134cecf54d5c6abae4957619fe4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2d89b7310aac357bb796ed99cc050b46d0db6f4b71097e2f2bfb18a4413a7d98",
    "sha256" + debug: "8b85cc395f19e1272659e00f4fc2648cf921a1f9346fef25d23aa55f6f40309a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ea167c829383ff90b0e650e6e91f9981aa12139b0bce528138a658044e0ace20",
    "sha256" + debug: "6c8569aea6c4793e16663a07386d72fedbd4ede9a225f48c49616f6a9191db49",
  ],
  "executorch": [
    "sha256": "b03baa35c5f05bb6305bd339eec4efc59f97fc7f2c5cbd40599b963e1fbe49e6",
    "sha256" + debug: "b5b258803b6b30637d09426cae1ce622c47e6be7906b3202ad71603379e6a84f",
  ],
  "kernels_custom": [
    "sha256": "77105e74e998545a86dcd74e77f1ec2bb2922227723cb1245e4c479c57a2afe3",
    "sha256" + debug: "2ad702186ba47ef2c6ef4cc9d47b3f428ce236f8b51e03abb20b34cb6388bbd4",
  ],
  "kernels_optimized": [
    "sha256": "159e643369950f6aee00dfff90b2d2d455b2b5fbad57bee3f62475b7b8466f89",
    "sha256" + debug: "8cd79914a19aba5209238a3bb91917072ac65f977f1b551d3deaab9e8848ca2b",
  ],
  "kernels_portable": [
    "sha256": "ca006cdd022e5934052f3641c23fe3150a32c4c365f252714b051499be5e1347",
    "sha256" + debug: "c32726cc395935c3206dd3679ab271b92ac1c86c9c6cf0da011de074350a8313",
  ],
  "kernels_quantized": [
    "sha256": "5c71ceff9ee3e4cbfafc400a78d27c2c72b65f1f44ef37ccc4ff18bad234c65f",
    "sha256" + debug: "b1ff5ee764c33a49d0d060da860ac59d01235093f84db2029b3044c76ca0e68b",
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
    .iOS(.v15),
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
