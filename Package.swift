// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241116"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "2f71424957db9e915530eee3087b0ecb300049da2d88bf72f7aff12dd7a8a9aa",
    "sha256" + debug: "bdf0132dc30941e364557a09de02350df9d9a09abb9fd670fc79bfc8190f3da6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7dc09150639ca97127400dd95beed0089efb8f5d42d47427b2f372e07256ac7b",
    "sha256" + debug: "99f01af7d3392a1910a93178ab63de5eaf7dd5e2345d7f4130736ed645636183",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c143e3b2a88c36de01e0090dc63551f437e1cb8914271a8bba3f9d14a8b7532d",
    "sha256" + debug: "5ce35fb112c8baab94bad36d6ff3ba67b748b02703d5d497ae2df8ec3d163e0d",
  ],
  "executorch": [
    "sha256": "90c7b0598d1251507515e333610b628fc763d34db766905e1ec1383ccd6fea40",
    "sha256" + debug: "f1f2f592644f18859ffb59c7b3f332c7ba326da4d5a3513824da0399dd81d467",
  ],
  "kernels_custom": [
    "sha256": "a883b5a3330b26ed8f200a9297fb5bf5fe4eff3edb51881bbc64db4f87167c0e",
    "sha256" + debug: "122fe5130357cf9a2338d8e10f71da86b663dd8d7bf358a5312d1d1373047bd8",
  ],
  "kernels_optimized": [
    "sha256": "38d26ef720f8c44aae414d3f1ed16dbf003fe236c79fd06897fb706fce86aba6",
    "sha256" + debug: "78fcbda4b7b32a6df4644d537059229ea9f7e2178ad5e1dcc80d91a45744e52a",
  ],
  "kernels_portable": [
    "sha256": "b99ed722d4837290e572f0d06b9c357b60b0af5caced194bd2e62d63fb8968e8",
    "sha256" + debug: "80feefd1721c32334b88097eada8fe487e5f7cd20a6789c709329c2a04b3f3e1",
  ],
  "kernels_quantized": [
    "sha256": "c116c42534447fce5abc94c7bc74c089d59ea19807a39c45ea2bb4d8cf83fce4",
    "sha256" + debug: "a188f2b0c1121a3fca9d1aec48bccc2f71633b3a3257c332e69de9acf91cf2d5",
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
