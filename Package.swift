// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250510"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "a533c10fc3b0149fd090ce52aaf6a4331c9b529789919114ac2c3a493bb50cbd",
    "sha256" + debug: "9a928c33bfd24f358d2bc54edcc76ac355c37294399cf5dc7630fb6f15451278",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9a6493a198462ac5843df44bf8bda7098646d6b36e5181911719d029aaa10c8d",
    "sha256" + debug: "3f7cf71a22c4f18b8bc932bbd5d626feecf6f13481c6e16c5a75326ba44351dd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6c073de77b7d0b8b73e281648f3fcaefc8fd8afddb7848a19fee899db75eba97",
    "sha256" + debug: "b6bb93adca869ddff78f2b185a54f151bedc340317825772dc32818e14cd4a50",
  ],
  "executorch": [
    "sha256": "23092c5fd3e6a7d22591db57d67dd86f8b5e43e29797f89923c73c719f53c93e",
    "sha256" + debug: "fd4043f9db25ae668b55673b97985ba2f3fd4031c57c19594c55593e96613ef2",
  ],
  "kernels_custom": [
    "sha256": "412dc13792f9073bdcfc60e448365ac3bc54de2a11d51175d11a68d717dac8c0",
    "sha256" + debug: "4bb710218c0cac81782ff4184706ea0be5779669a5ea2abfc7a004f2497a6616",
  ],
  "kernels_optimized": [
    "sha256": "ccd9e5145aa70983d25c4d329a6de933d59321aa5e2508b0bd8aeee28d8b60e3",
    "sha256" + debug: "6d1661025049c572c5fef94f42ad97df742ef3269f5b828686f1b49111310ac4",
  ],
  "kernels_portable": [
    "sha256": "eb7ec1035d8cd28626fc3ade42828f90f62845bfc4b7d9c25a8f605ba95490f5",
    "sha256" + debug: "1e387596082dd629a1996272106454f94eee5bee69bbc2a89da141c02caa48d1",
  ],
  "kernels_quantized": [
    "sha256": "5d134b56d6df9a08ee17314f0767d5108947c4eb73eb7bef211e2af2fd6aa72b",
    "sha256" + debug: "7538348a6e89ba41c01c76fe8bcc19ad806dfaee73a45a40f54ed79952f90009",
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
