// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250415"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "d09c98e893e09087c0802ec84b7299f846b10d037040404409c1e7afadd3e7fe",
    "sha256" + debug: "ac9795aba57012efc3c7ac620a235535efd6bcb0e1ee208bab1eb23135fd78a5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "063304271fed5216f9f78796477bef2986807de02a6216274cd02863659e5c4d",
    "sha256" + debug: "b5a05a0133f8bbb2f34825e367946392d77a25fcc52a4cbc20bf87ba8bf3cab5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b6c3d40c4367f596b0067af837445659f6169ae19974b39f295025e60fdb9ee2",
    "sha256" + debug: "00aad9049a33ce194aa6d1899c58992ea00cad649796e4c0b11383e516331b6f",
  ],
  "executorch": [
    "sha256": "fd377365970d436e3efe3220ff3b9265b1298044ee1277ce6e34c0f7d96b2368",
    "sha256" + debug: "1104cce27de3dc4c77b587e4c6576d298d51c1831af112a381983417c3c77c10",
  ],
  "kernels_custom": [
    "sha256": "a0371cb322e5d5243ba0135dc9b11e2502f99ac55c93366b932c786754edb4f0",
    "sha256" + debug: "00f076bce74e55b6413fb5cbec0087c62b14757aad4b42f547769daa0dcfaeec",
  ],
  "kernels_optimized": [
    "sha256": "1e47ba83bd005629a40eb5ce2973787d6371212b0a00cf4b051562acc4ca620e",
    "sha256" + debug: "c38a3ea7bfcafe96d6a38e4506c17474489602724a59c240b1809112ddbf7cce",
  ],
  "kernels_portable": [
    "sha256": "cc96e7183b1dc932e9e9861d2a8f77b7f5550ac85ca62882e2fa5b2ac7e4800b",
    "sha256" + debug: "9defe939ee0c3df9f9392a84e841976eb705156f6010d45d12487d11e1443a86",
  ],
  "kernels_quantized": [
    "sha256": "b230cd0fdddad1e0c88a6d7b96a9c529d0ed55b57221fbc5b551b36e566ddc62",
    "sha256" + debug: "4aeb0d6198594b5a0088931bcd79151915e7d4f5e524ebe54ab1d67c36cfaa1c",
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
