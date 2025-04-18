// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250418"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "3e10234e532f56e9ae3cf8098d47ce6cea0fbdfd41a9a68593dd600b0bd985bc",
    "sha256" + debug: "bef6b5e43c2d081673a3ddcfb474476fdd084ad7dba9264a592edd6c526ab4d3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2965a78ab019b6bfee76d6211c7992395efd30db51ccdea101f8f23db7098708",
    "sha256" + debug: "e2f6263b98be3ffdd1588319695b9bd6d5004ad30c0d3a74a6515881ae2ba04a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3d5e6a0c8f9d2e87aa54a0e7e3e94d2c099863b354771a4f8ea0a64ca665fd23",
    "sha256" + debug: "cae034e62841efb5d7c7176b654e74a698a37741e96102432da171bb11316637",
  ],
  "executorch": [
    "sha256": "8efff9b0c7a4e3e3b51eef3811981b53f6f11726916608128c5ca3d2fae81e54",
    "sha256" + debug: "b1a0aea21107ec2edaa65b8ddb6cd0b6d85586bf76e3ccf5c4aefff920598a94",
  ],
  "kernels_custom": [
    "sha256": "90df55b982c9008efa7e8563f55027396b30e44d48d9ae01d2d14ac9e1497880",
    "sha256" + debug: "3f5888fc5c48e6bdf1ad88e52f2fcab897d9221badacc268d85a976c386e5439",
  ],
  "kernels_optimized": [
    "sha256": "96b0362f006b091e71f4e7ccf6a4214b7c8b3ce568a0d1f0ff12031ec3541658",
    "sha256" + debug: "856e45be1f706cb94d08bfca95c0feaf9f5d096a48cd5b65c51354c5d213b5a8",
  ],
  "kernels_portable": [
    "sha256": "5725219126e93aacd20c694a4a7763bb228afb6ff7ace0850b31185110c5b56b",
    "sha256" + debug: "c468eca4576c203965f012f781ee6e635ab6138c5d2f00cd06ca04fe4cb7a86a",
  ],
  "kernels_quantized": [
    "sha256": "14a7af463e64e66f8f5906f0f8ac5963dd1afd5d5080da312f0d0736771657d4",
    "sha256" + debug: "23cdec2de43fe2ba95bfcf82680f75b194b330730cbca90f6668cf6415cd20c2",
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
