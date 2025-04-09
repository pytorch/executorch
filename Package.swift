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
    "sha256": "8736fc32fede1ab866ef4621bd048030d66b34c75aad7b3cc0a119e96f8e4c1f",
    "sha256" + debug: "507c7f5c267b5a315900afcadfd6ef2f7de20a57ce7402242f35e0770f3e17e1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e200716cd3e73ba5df1b070730cbafcbe29181e41f9d983759358f439f3833fd",
    "sha256" + debug: "194400d25d41b107c005431ef0210a1647e557962c04c0990efe643645448450",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "831772a6b63ab267781ec433b40760d2b0644f5dfe6ce7d6cd64fe97d52000e8",
    "sha256" + debug: "1dbe77e0af90097831fb63f1954bed16301c03ff571b5249c90d45db45215e34",
  ],
  "executorch": [
    "sha256": "ebed3bbee3013edb156b2f8ce6bc2838157ac3f5564a643ff348d443d300a88f",
    "sha256" + debug: "9073def483963b6b255abcf3a086053f9f4a72aee04866797747f225810f0924",
  ],
  "kernels_custom": [
    "sha256": "4dffcf703e1b5958388c352ec34ea11c3bda3a4b0a55bdaaf9ebe0bd5d359a4d",
    "sha256" + debug: "98169e609298839cdf372d7c2b912ff41aac68be2f5199178c817595dbc01277",
  ],
  "kernels_optimized": [
    "sha256": "11367833b1407dbb27e1effb4ba94d3a111d569cfe1edf791b84983e383d70c0",
    "sha256" + debug: "b8d318980fbf1990f7da62039fb70d8cc901366d729197de99e9be353f12e14a",
  ],
  "kernels_portable": [
    "sha256": "2b9622a7d87a70947577349c96e292377a7443217fa7207b822da1761355586e",
    "sha256" + debug: "c9a6628af110ce8b528ba14c23613d02dbcfd068157a121979962315c4fa00e0",
  ],
  "kernels_quantized": [
    "sha256": "759da4df7f48f166b2efa58ab82046db019ee46960654e45aa356d175eaef7dd",
    "sha256" + debug: "a184d5a3f35d17c6cb9dfc6fc8e862b5c5ffd89654a34c060a58c8979cb506de",
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
