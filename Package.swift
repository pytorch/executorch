// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250317"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "e27e5a891ce020d8438b7b9870a9b697e7d362719c17ba5f96d3179c1604a825",
    "sha256" + debug: "f2b7e7acef570821ade29423ec2753b55d0282a12f48fe8ddd7db9dc5c0e450c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bac4f4fc0dcd6c2274ffa2184f685177cb42a07576c02a10a86de9300640b4a4",
    "sha256" + debug: "651133f299f7cc71c7cd4991358f0b4eae2247e2342f07d7b24866267522cdd1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c46bb5c96d9e4d3081a9d48a9790900c112824063392a1b350f00b8e5ba7fed4",
    "sha256" + debug: "ff881eaad42ca177b55d7468f3b5e6f69ae55ef3c9025c2fc2971faa1a918b17",
  ],
  "executorch": [
    "sha256": "95cbfb8fca214d4892c6bd8322bd26355734197b93cd8711289f4e655dfe4e49",
    "sha256" + debug: "11d16cf79b26e431f5d61b1fcbf785be924ff0202c73e580efe75c044612e4e3",
  ],
  "kernels_custom": [
    "sha256": "3e13b41ca3069d206680006e884e97d81a4afa3edf5e507faea09a11a56d3a5f",
    "sha256" + debug: "5d0fe8cce4aae1ecffad3021fa5a6291d023489b567db5b1bee06a033106eeda",
  ],
  "kernels_optimized": [
    "sha256": "e90c360506c9984e9732e5507dbb20934d78a5279dcfbd1d572bd40c7f8eaaca",
    "sha256" + debug: "8668132cc6187a48fcedb6c07ff1e85ec49b7466f84a001ca183adde73f66607",
  ],
  "kernels_portable": [
    "sha256": "f831337642c8d18ad60f157409cff8b674f64198d9d96d7e582a01125ee55b06",
    "sha256" + debug: "3f3f0a64741615465f673fac1ddd1b67b314544e60574afd0f7f4464feb167b7",
  ],
  "kernels_quantized": [
    "sha256": "0d4fa50787ee5ca09314f385bf3101f55597f61bbedf27c9e8879db7343b8fdd",
    "sha256" + debug: "82c2467e27110ee45033224d9dd285fc9e5d0cc9ecd9c5aa5f45983d19a97a77",
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
