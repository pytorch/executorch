// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241202"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "49f8198b6e1f5a52da4c5bad39c0322318ef87221e56e5b04600d0b542ecab88",
    "sha256" + debug: "4cd77e459b26023cdd45e94ced52492eaeab7b3b139d30d8020774ee5b39b98f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8a8fed8223a7f8cd291d689c13f6fc1c63080785703b2b204ab48bbe7e2f78cb",
    "sha256" + debug: "41a5628e5f2fe491941820cf9b7ac3394d5c384ee48cf70eefecc3fb36f332fe",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a381ad7126dfe97108ce4ccba6287b8e2b2b6af94da2a0d359251513206e4753",
    "sha256" + debug: "a2799bb49bff665efa0b59ccefc93f74cece7a149b7d7db38250184cfc11e46a",
  ],
  "executorch": [
    "sha256": "1039a1264accf11c7c9d64d41f7c85be4f6a59fd81439c53e58579c90a11345a",
    "sha256" + debug: "ddca387b62a0673b57e3dcca09a3f8a087f4bf51dcd8ef8f10e811ff01d1554a",
  ],
  "kernels_custom": [
    "sha256": "a76f244756ab086bacf0d91876cf46b88112742119120cd92996a4214cd4947d",
    "sha256" + debug: "c8a238d796ce65ba6736400c84186d3fc4b5eb0256e5afc84d0bc8b1d20d5597",
  ],
  "kernels_optimized": [
    "sha256": "845bd8ee6825f88abc853b40ab8910e8849d2e6e0ff2943fa8be2b621b9b4ab1",
    "sha256" + debug: "88464e1143db3fd6146b8c61c7d63afa9a3c978678f2caf42a98344be6fd4a45",
  ],
  "kernels_portable": [
    "sha256": "742df6ce05c6c5046b5bf888599fb2b929e9cfea2102281afd2d93b7c3d53184",
    "sha256" + debug: "bac38ca1b11a263f0f7b3af0da41e15af7d228ecdf97ed1a9d38392fbd775805",
  ],
  "kernels_quantized": [
    "sha256": "99ac1196f7e77f1b68380df1b95ab58b8f823c38b95c7b673badc9620d224d3d",
    "sha256" + debug: "9e72e0ba768653f1b54e9bbb6af503034ced1a79d37a88fe3a136f653e43a4a4",
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
