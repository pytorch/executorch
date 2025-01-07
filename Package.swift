// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250107"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "75de6020433bcb405391d5a4ba5f608c221f21a0d5c752ecb9831c88d3fa28df",
    "sha256" + debug: "c7f96a4884a6928809e006e76aad3ba1d140d71c3489cff33c0230a995e29f95",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9735e7108395766047086ad7e5f95e1e4172891a7a94d6366e62ac062005a2c3",
    "sha256" + debug: "b4f665884ff5acd7cf4b2ee5ba95e8494abffc61789449e7feffd950f0bb9be1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4d75728984f938436b40c81f10d59a37710135a7466a37ad140da116d4e5e58c",
    "sha256" + debug: "31a74b22a273c79f01b89f807e02d44b3c3f4e688d26e57aaa58df0dc04fb40a",
  ],
  "executorch": [
    "sha256": "8f2af9eb4172bd092226fae44148f732069fec7dc2b49ab1fd23bacaa4722fe6",
    "sha256" + debug: "43a2e14035ca6395b518ef63caa116f22f672357ec1dfe542f6f465d7622d53f",
  ],
  "kernels_custom": [
    "sha256": "7e6711fba2327569a1003cb2786df6f1bea3bde1792cb265ca69431889085d99",
    "sha256" + debug: "59e4c7987b9e2870ce53a0d03d43d43a41491714eadb3ec9e08a691170d9234a",
  ],
  "kernels_optimized": [
    "sha256": "5ed9d9c1c1883d09bcbfd720cfa33c456677f89224f672f495b612e3fc86c048",
    "sha256" + debug: "b38bedeb45308cce2d0f6d83b635387a1a20a1b792570b46ca6307f3d9bb3318",
  ],
  "kernels_portable": [
    "sha256": "2bb92785cb2f297af18102c47ff28ee1fc066960ee78bd97183cd56860b47095",
    "sha256" + debug: "aeb7318a22b72721e0ef0781e461e2d8c03631cd5898d426bf7f1af6402cb20a",
  ],
  "kernels_quantized": [
    "sha256": "6b63e8877a593fdb38c2833afd1e41dfff927b95eddd9ca205ca27a6da2f45dc",
    "sha256" + debug: "a1c16b51f31d0915b2db664fa4d0417d1a3463c7ad9e7bdaf3a4aaec41cf538f",
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
        path: ".swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
