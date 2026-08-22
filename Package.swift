// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260822"
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
    "sha256": "9e1411ecf9785a9cb0681938ad35db1092b1f2c4b59f94b56400d0340b196bff",
    "sha256" + debug_suffix: "a66e642c02bce0218837e68048ed18f063aaf385f0d5a0da943c893d979c4005",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ec7fc88ba45320eeb1bad026c3c750fac46c65df31a709621d9785fb32a7d9dc",
    "sha256" + debug_suffix: "df05ad7967e9eaadbc3bb99de6e52fe89e16fc28a294b051a51264b8ffd57bb0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0472ddba52ccb344f3366ce758e47f37c4af46c072a61a19fb9b2c71d4e91582",
    "sha256" + debug_suffix: "d5b068171d9041c8d81f1db0835e145017f4424e7fc0e57e24bc807d0f88f1c9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8d6c637eb7a45ece0c1a9025566572e2c0bd6b4be3f27dd7b368e1d064e92ae3",
    "sha256" + debug_suffix: "636600bd8ca660ddd47e7ec078caf45eb580a381f3930d635b5fd33dd1240e98",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d578699e3aee5dea53698ba75bde38a2e46b3a49ff1703aa1f453044e2fee465",
    "sha256" + debug_suffix: "cc109cd102c170a979f0b19a181db5e7edeed60d02eede7f1f4ec074094fdcdd",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "85649ea4d840cc67df485776cfa41eaf8f6df3efccf15d4e01f7ee3993d01454",
    "sha256" + debug_suffix: "760fd928710d9ba7cb2c0c237a30a2aac0faec6f5d47fa0e7356915e861c22ad",
  ],
  "kernels_optimized": [
    "sha256": "541df335369192d3f7cf43cd3514414c29c3dc70f8ea7aa6afc28f3febe1a45b",
    "sha256" + debug_suffix: "d8d5fbce393ed77cd8fb1ba3844a64e59ee9d1407c77ce7993b2250c5c8516a2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a81848fc30534a32cdbc30e0310c475df3b1af731cf68ad79c5df164807a211e",
    "sha256" + debug_suffix: "a2d6cbb042c5ab0a1b0286825b9fc9bf5537e9575685ce3a7e3347b360a25000",
  ],
  "kernels_torchao": [
    "sha256": "d7e7bad079847f668267a1965765377569b1bd3b3919828131ff3db101e814a7",
    "sha256" + debug_suffix: "2d20f41747776b18d47b89fcedf4395e509127a70f26877673b7e955090be04d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "28c68a5c57619494cec89806f3b28b0cff584acb0a2df56a63d4e003fec16efd",
    "sha256" + debug_suffix: "d487b2b064ebbf1b4c05d2f9e409c13092358cf3857fafc472e9592cb78a1ba7",
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
