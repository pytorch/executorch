// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250426"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "ac669b8f8a95ffdd147d839eb24c1b35b68705f89254ad3d432ef02caae9b552",
    "sha256" + debug: "9010211bc8cac27e0f7008c4cf60300f45ce6ec26cb973db16b98dd9cad8ee26",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8690fffae36da89b92edfbc5a5286394e2df0dd1032e57bcd9cb9e0e15dcd7bc",
    "sha256" + debug: "2cbdaa61cb8e690a8ce10215479855e47b5a8bbfb66d900ba83d85395acd9499",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "39ef046057ab2e3fa3194f06835fd778622d1b3450359f02a9c4a2bc9673d460",
    "sha256" + debug: "6a03b5cfc3d47430e351042ec984c7742af8cbd40bca33a06a7786d24aab79ce",
  ],
  "executorch": [
    "sha256": "153a51c82c2b0ce72533760d274c7d03ec3dbdd70e28ce20be864ef6295aa8f9",
    "sha256" + debug: "f7f8c841d8e29498a01f6c37faee5babc2f1ad7a0e58c4f80105a3ecf7fd20d3",
  ],
  "kernels_custom": [
    "sha256": "d5442bc4a9aca0840dafd45101039dd2979af9ef9ef3eefc0448f281c4619dea",
    "sha256" + debug: "51405f20108e4df76707bc8722cd2ac8dea1ac7eb7c9d36109e34096af7cc51f",
  ],
  "kernels_optimized": [
    "sha256": "5e453b8b249cfacc8831b70c75428c252498ace5851bdd03adf63e35925b1bd6",
    "sha256" + debug: "b7668ba20d61a2c78cb3ae505b444195fef65a5cb89b65b31a48526706235f78",
  ],
  "kernels_portable": [
    "sha256": "2dd3ae47ea3329c49cac58f384eeb49c2478d13dd0f63876e1f478fe92b40313",
    "sha256" + debug: "cf6e1912252898e54fb19fa4ed53bce672efaa91ac04454c4f16ae3dd87eef15",
  ],
  "kernels_quantized": [
    "sha256": "90ec3d5883d3ed33f04d717a600b4f6555075611ec279da68d673e46fe780f15",
    "sha256" + debug: "bcd210407825a54a973fd2bad15178e10880dba22db2521e5f60c8989c5a81a0",
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
