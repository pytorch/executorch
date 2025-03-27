// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250327"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f497c1b2c6a4647bd2b5f88362f7c0abe86c3e4bca2f71dc5dcaef2e34fcc325",
    "sha256" + debug: "77bc08125aa0f5eb8d8ac89a88d41cf7fbd5734835064d62f34de4f012f46152",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f911c2960fdda1d9b1249c389a2a8c05072fe3d330099a6431b38a95db993f8b",
    "sha256" + debug: "9f424950bf9f6bb15cf1b95a845fa20cd3982110cec939eadb8074a815eed373",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "040040e7f919bc4ba01a7f03303593ee81e7a726108e4ed562e3cdfab55b3829",
    "sha256" + debug: "8daa2e1a7dabd0af2db68e8e162fbfcc121f731cbc675409764b223c501342a3",
  ],
  "executorch": [
    "sha256": "d8bd4dfbdab74b040fce25ee6c508d8a702d01fe7f145c2ee07302001a0e616e",
    "sha256" + debug: "44aaf4129a17b75dce05b6020f0b6871e92a99ad013135b9b0072c2e39ec2e1e",
  ],
  "kernels_custom": [
    "sha256": "27198c7abe9cb379dfcfb4b8216c33d88c6df62189d8ea406e3922f533e44f23",
    "sha256" + debug: "34b6064c72be87f880e2e7ee81a7c9176c7543f1656f98b6453d649eb7d3383f",
  ],
  "kernels_optimized": [
    "sha256": "ff238ea897f0c49ef80f380fd9dfafcb18108359a0aea9aeb6c7a6b48a0f2323",
    "sha256" + debug: "918a64cca9685a87b9ce56f16f7ca5106a1f3576ff44ee354396bfc7eafa6dc2",
  ],
  "kernels_portable": [
    "sha256": "de28de5ea4355f5f9703d25bdb7ea58f1fb27445ac3c1f519afab20a116f8d50",
    "sha256" + debug: "c1d4cef11a1b2d4c7dfb585431fa84a73b4a51fb24ab86f3e2f0b114efe2fad8",
  ],
  "kernels_quantized": [
    "sha256": "68a595b42c08e0fb70d0374c9359fb5a766157be16ebda6c251fc79e8f7210b0",
    "sha256" + debug: "b07f27c0c0d78f2b0250d77ee0043d9ebb327c1507f1bbe1a6005515c79e9c20",
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
