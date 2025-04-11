// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250411"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "a93bb16655c5ae9994cf64927a9107d2f66531d61322968ac8b6c29a278a0341",
    "sha256" + debug: "d64dac8eb06fa740ffb201876d08a33479ec775921488211525a2eb6504f6429",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "69c092eb0d49eaa356f914d919dbec24d5d04045507943601a0d975aebf1caaf",
    "sha256" + debug: "c2177fb2b539ea296890c4e5e19e0cd78bd4a4926b4474c87597be764ea7c410",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1611019057c1c933723c725e840f8b78d93bc300bae0ec388ecf46b5b1b1fbf9",
    "sha256" + debug: "6bd1a209e4c52730d556b9ac30da2bb1f57523423daa1cab18813a05aa20ef72",
  ],
  "executorch": [
    "sha256": "b22653398cdbbca09b2941cc087dd37f1de6dc280624d3b6e42517d9c1fc5ff0",
    "sha256" + debug: "a9d14dd23e05c4b63292148d37e3fecf9ca67e3c40f4a8e51c3af0f846865b93",
  ],
  "kernels_custom": [
    "sha256": "2c0ce43902e0c74e67ace38eaef93536bb53d93924f6ccf20ca44045ce5dfb2b",
    "sha256" + debug: "0de3d926016e3196a83494660ddb1efc5d84403c8c4fa4da3f5c1b3fc4d6ca4c",
  ],
  "kernels_optimized": [
    "sha256": "9d564492166fbe296660d8c74bc18fedd795d3f9813cbe1828dac6d89844d415",
    "sha256" + debug: "3b961a68cd8ffb0a7f9c3ba7cc66ca375f53bed460e7bc2da0c835a3ca915b21",
  ],
  "kernels_portable": [
    "sha256": "440fec673bd84116b5324412146a37ebd9491d34328c38c8b9c85e5ff52ef3a4",
    "sha256" + debug: "c5cfb7e3b756abfc062d732ecaf84e7b4acdbb143bf61ada7855fd0fd0caa5b0",
  ],
  "kernels_quantized": [
    "sha256": "18a96e2edb490dc03321f5b9711d08ce026bf9349f00fdcf9db61d96b7c65c29",
    "sha256" + debug: "9ac02cfad16b7a451d271f284e921b03cc31b018515fa9ad95aa147efd3e0f88",
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
