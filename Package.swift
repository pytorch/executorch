// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260721"
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
    "sha256": "39cc67fafb73c79776747763464188df900ddb976928f613c6f455bc83ca1156",
    "sha256" + debug_suffix: "2591e3efe000226bc4fa1cb994d2c900026e2a8dbdb4569f304b5d223a88d2ba",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ff42205e5d2ec2194af1bd2fef2ee9713858a07dcf7d927879f866896168da9f",
    "sha256" + debug_suffix: "b1ccf96bdb3f3ffbe8f7b99144154751d1e54d6d37f88bbc6ac4672c399cd674",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c9ab775242df07b0ebe7412df8ba5d45a7a8ff667c255ccd1533536e44fc0a45",
    "sha256" + debug_suffix: "7e6baec9d81016b07eef6eab098b670244d9399c52b9c2398c910ad410ca77ad",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d3497c99c2ba37b13b42d64e254aeabd46a05a5d96709f2f3d19923fdf9a1428",
    "sha256" + debug_suffix: "6783ebd451eda1e62230cbcd348e17ec6097f03d3bdf56c7cca82c3f8729ecf0",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "6c6857f9a147c12512d6bf88fd983d97782845fb2aa655fd125e36dd23696378",
    "sha256" + debug_suffix: "752dbb7acc682e567d66b8126fb61d0d22c02e53c84b86f16beccefd9500404b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "66ea60aa13bcad9d4d6a70a25fd7733a3c8ea9378e04547262df9491819c4a03",
    "sha256" + debug_suffix: "a273dc3899cfc76602b8af949ea6261dad269bbe80e5cb2da6c8f5b43a916163",
  ],
  "kernels_optimized": [
    "sha256": "67f3cec63a337b7633db195f2978b610664a8ca5cbdce812bdf540fcd11f9580",
    "sha256" + debug_suffix: "50d44beab5b918e8f5c70ee04f73d5f20ba2b0611b3234b685e4647967eead21",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "59b1919c39d049ed538a4ac393c692fd618e9b944eb06bad7f15bc6164e49872",
    "sha256" + debug_suffix: "5da9b28d454fb587a09ba92844bcf35d1b786d253616216374d76f5c8cf4e97d",
  ],
  "kernels_torchao": [
    "sha256": "2c78080a888d36f0b0785bf09393aa404b25333b0886d1e61202afc61b03e176",
    "sha256" + debug_suffix: "c4508a3eabd503cc934b693c66362766938ca52d3ad22aedd65b9928ab9859d7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "de6fd7557e8cdcbbfe036f892d638bcaf1a560706abc508a5f0d665947055057",
    "sha256" + debug_suffix: "a10dca9478fae2ef2f9be4a36f488fdf8ff892023a6ac6b44c5d03b97f6cce96",
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
