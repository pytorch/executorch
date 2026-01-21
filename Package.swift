// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260121"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "sha256": "927d08eb22e9b3ca7fcd6c5005decd1e739a24590e0a8545c4fb4b4ec5ff7cd0",
    "sha256" + debug_suffix: "a84fcfd146364f63ae11126d70ec656dd18524218db79bd430a47490d59e06b4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "621174e0e4c29816c177b17768a6d726df5914f4de90d586fceb34f5d68bdfcf",
    "sha256" + debug_suffix: "9a32b7794ffae9e13e12721322031154afc88764e7cc96ddaa6a57234f17bc29",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "87c4e86338f6b77bf694a5487e423e11551f37bee74826db2e1849a1e967d9e8",
    "sha256" + debug_suffix: "7e2a108109850e598c4f59f1c9130f0d455cdfa94559c6164ab7295b4fa1ab4e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0e1a2e1fbb6558388926b35ccc076fcaa8cd6a5810779c01d247effe0f172588",
    "sha256" + debug_suffix: "56ef68034cf27d036407a091a75133b5fd84f0c50ee568b3ce341f8b6f052e2d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a087758fe8aeec16a9ffafe856460b21fbe7b232da949585015b320f800ad77c",
    "sha256" + debug_suffix: "9301a8a81ce9fa9a845348a3c07bade111f6a0d6fceaf0a5a5191ff751ac04e8",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f9ee223e91baf5c2c6895ff91f4056c20ed12fb4f51416f05a1e48e6329d85e6",
    "sha256" + debug_suffix: "9928ea349ae92ca845d55c903bcfcae15c68bed3beab4a087200dada2fb05f37",
  ],
  "kernels_optimized": [
    "sha256": "ed6aa9d6eba6b5e2aad0092e0ed03dd1bb9b7ff48b309fb2649ebb06872c360a",
    "sha256" + debug_suffix: "71cdf79223b251fcbc575aebed81a00404a987f895611f1e6167a9728462037a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ae60cb6d94c8460d2ef6d1b47e5f330371ae054be9de9f492ac970deebff710d",
    "sha256" + debug_suffix: "692d65de14f2c1c3feee7683a1b4821d4169475917e8e8e138266bec5a9c6a6d",
  ],
  "kernels_torchao": [
    "sha256": "971b5eb28c1962f552bad0ad1039a3b740c55104ce184ceef943b4c247348f8c",
    "sha256" + debug_suffix: "5e7390f587dd32898c40eb3528af081cd79625abc0a9be2a66d3508fe6d6bab3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4bb218f4a88b2a2b0d53f29edee33f46dc0349ef02db8c95fb66eac6388b8b63",
    "sha256" + debug_suffix: "103aa74b405e1e77a71385ae352da97f04ec1b01a517c540fcef6ec3dcbfdd93",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
