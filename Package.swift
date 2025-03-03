// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250303"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "e9786ed76104c8f3f569d4fcc0d87f0a7c1b6338b5d231a648912503c3fc592e",
    "sha256" + debug: "f04457a123ee65c94df7b0f55b5d99dc8446fe96edf148bde7c540b6b2aa0343",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "85a0b322fd92715071cdc3f859e7ad869881662872d7b646ce787f1f0ad75536",
    "sha256" + debug: "cc0e59ad3a7a4193da13a1852178b672242d946cd50a597b3a162dfcf8bf2454",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9ced13985cad909bbd4e0618c6628c0b06a14d7858fbc17f30ab8eed566704e3",
    "sha256" + debug: "607f0746ea962904deed426c11b9c57fc29838418aa45f4b5f1a5a183704b614",
  ],
  "executorch": [
    "sha256": "ea2c3fcdd589ce26002d2a02a31185be150be70f942761f3a06d02a0f810d4f2",
    "sha256" + debug: "b269bf4f049c00b97fe7e5e02ce352707ef191138a830c9d812e1e8142decc53",
  ],
  "kernels_custom": [
    "sha256": "0d1baaf0b26d4e4c7ce426cef6822cd1c5c72e44d5ed6cb53e83cc467c3ef720",
    "sha256" + debug: "6b0e8e11110a9dd2c190a680e89c611056c5d0c225f48d64f53fe0cbcdddcfc4",
  ],
  "kernels_optimized": [
    "sha256": "e10f389aebf3c3f52846b00688410643d9f59e3b57cd994d83dd6e12581c5a4d",
    "sha256" + debug: "16c9e4618fdfe46b83477af58f6966fb524f5e4b33ac56d7c1b3e644542ccc0e",
  ],
  "kernels_portable": [
    "sha256": "73f8c04d4c4efc7efc1272f6b3734b849e3a200c653f0576179202109310949a",
    "sha256" + debug: "10116f345f726cc179abe23ae5ab10aae0bfa057f95004d408bdfb6b393bd2ce",
  ],
  "kernels_quantized": [
    "sha256": "1266bfbebcce0e21620348d41c884b3509025a92e7b810e33a63454bda89aa85",
    "sha256" + debug: "4f4013849ee41e79aa9be2ca39ed4ea12f340400828e87af5311cb9ab4c93c7b",
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
