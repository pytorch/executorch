// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241231"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "8017f5f13c49663493296a0735073fa3f3e8ccf69c0cd2b9cedbd45264d97a30",
    "sha256" + debug: "500814930b8584519f8e4c0c8e398032d337e7f4919a96cc0429feaf75870a8c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e85bc1a0e669daef3cbefc371e4cc21916d8dab055050e9c596ee097f2ad7145",
    "sha256" + debug: "772df0f464faaa94e9f1800ed872537656cd90a91df4757b64deadada55da95c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5d2322844c50b639eab4886adc0535d731c3d8b6b0d57dee5c54b8358193f803",
    "sha256" + debug: "cad1756ea3024beaf3c9574349eb4667bcbb0da81f65a89532f41cba054b0d55",
  ],
  "executorch": [
    "sha256": "59f0f50489c0ec2e19f2dde213d55517aa0b276a893703db943787969c7b4a17",
    "sha256" + debug: "c1249bd334e8196494b2b589774f3a8936b8013451eb59ee5a0710d3db4ce490",
  ],
  "kernels_custom": [
    "sha256": "5670761aa026bd14ed61d811043af8204433b15899bb6eddd4c6f6d29bc499c2",
    "sha256" + debug: "6d77125a691adba0877a3b08a0b3bfb76b3d2f39068209f03c54ed41b1c64eb4",
  ],
  "kernels_optimized": [
    "sha256": "4180bdf91fad1739465b45514e37c4328b617d5bc5a4759ee34a8baa2df53a8d",
    "sha256" + debug: "74d679e7312888589ef701714afc0e046f654d0789936e03844e36267fb0dd68",
  ],
  "kernels_portable": [
    "sha256": "d362b801db049b0f8de713059c816b9c2ee516bde52e88eddbf52bd8c21e2c06",
    "sha256" + debug: "78e35fb6388c30e14cf36975104eb6e0500f8589eef8a22af545593257cb205d",
  ],
  "kernels_quantized": [
    "sha256": "76e02e3da3757863de8bf984b53b34e18a1a1ae01190d506e4f02f661425709f",
    "sha256" + debug: "c25d80afc98240a24196ff41a4beb592447224ac9b63852a6258eede86244d29",
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
