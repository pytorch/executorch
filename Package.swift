// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251124"
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
    "sha256": "15e6effa308f082113852e98fa1b706dfe226edddb82ba70a0761757010dcc91",
    "sha256" + debug_suffix: "bbb1df0bc733e60a9ba5f9fec714bb7052d5bc4bd7fbe3c2b12f15504181f1ef",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e888575a1a5325e8fb2fdfddc25ae5056e92a71f774ae75a41873bd6107bf143",
    "sha256" + debug_suffix: "516675ff72b2acc52b7cdbc7c90e02645265c2ea90691efa43e2561066eb235f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b96b9f5c57916fafad7003ef2bce2fe181ab2d6e4b0fc4f42fb4fd58c0a01aa5",
    "sha256" + debug_suffix: "2efa48cf6fd2902ff996a18c6625f061282fb7c40f8653864f768da78df45bd4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "bd9190434b48f1f3034c814b5f451d7364669b25905976a4fa358422f9aacdf8",
    "sha256" + debug_suffix: "8facfe380ccf62ebe06c1c5f6669200e2c620486c77b6f410ebfc98069bb136b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b0a37d3df07bd2964bc40b3f698acaef93b64b6489af3e129b85ef363f6fb7b6",
    "sha256" + debug_suffix: "1cdfc0676d39d87a359fe596eaf2e98ceb9d07fa559d280b15563893fa227c6b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f3423a159624f992faf86f928206653b1ce5fa794046e71b6c43337a94e6e035",
    "sha256" + debug_suffix: "192857b2fe862ac2dda9b0099047050e269f6a9c0454f1273db2ba5302a02f03",
  ],
  "kernels_optimized": [
    "sha256": "65bf1648695ff0d87ff785f41f44d339dff1df9edcefbb3c81b0c27ad71f903b",
    "sha256" + debug_suffix: "90bee714867b2a3ee8134ce6eefdc53b25e55c7bdfc287f32edcfaf6ea81b557",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "896e255779e0d532a14d8c339707320090796e97ab53926cebfb7d185a3cdf62",
    "sha256" + debug_suffix: "d903589ff13e15b04b866f7967abde4eecd6297c81486bf3ae61357143790285",
  ],
  "kernels_torchao": [
    "sha256": "61c65adf1f10466b886868bd901ab230a7f91cc3eab07aca0e394d32168b9eca",
    "sha256" + debug_suffix: "7de3ad6d2092b0ac0d2ced9216d43c86ca7d7affed5d3a2cf7ee61ca14537e2d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b144b5951ec8f556a645c27c7b539ed1bd2fa4a0a98297fed649b1fe800f77fa",
    "sha256" + debug_suffix: "4a37cd1762288d0dd5e5e1863cd7dc15aa12ed4042ac221234a6a7b132a86ed3",
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
