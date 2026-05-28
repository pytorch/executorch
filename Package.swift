// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260528"
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
    "sha256": "08dc34d4b808d1186307f44b2c62df7de2eaa5c4c44e61381dafbfa5a5ef21f6",
    "sha256" + debug_suffix: "1dac970c5b4c7060b7d987f10fd025e56028f5cc7a60ed4ce7782cbc04bb5707",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b80384aa847113dab014264d1b2dd4e572835e2fa1b972e0205c899b1edb0ba9",
    "sha256" + debug_suffix: "525ba095db71becae51d87560b11446b687b464b0b0cfcf9787330c8f5f1bcc4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "41bac81a7598f0a27ab506fa57d7cf8c9ce2836614ff0ee984dcad4a5055fcaa",
    "sha256" + debug_suffix: "212c6e2921cfd331b6622e4ad09912193c3beeaf7b3de813ef46e25ad04bef2e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e2baad3548e8748f73e55d5b363b4a6bf0280db5e91944bb86b6801ccf3f2b53",
    "sha256" + debug_suffix: "7e16c63bb5067623d1fd01e59630bc2497aeccfc94393974fac224df7105b2aa",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "cf6ff110f42c4f9ec54ce2fd3c0bda74ea04c1a90b01716f613c4135f0759921",
    "sha256" + debug_suffix: "0d8d3d0beb46ed7b5e51892558427de25ee55298bdef31436b230c30d8ebffdc",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "dfa91ef928ed582d7aaf2363b8a5b109419e99b134c354cbc94655f78cb6da57",
    "sha256" + debug_suffix: "7608d9b464032b6fdcb667cd6ec4aebfa86067e79d88c717640a69b6e30f10a4",
  ],
  "kernels_optimized": [
    "sha256": "8584c9f9c95e1dbe29ba37d9e4496a0d34f60bc07aa5ce2958c7d0da4d36e8b0",
    "sha256" + debug_suffix: "360509f3b1be105caab9261a4a51dcaff1b3ada1a26ab2e9885b04c45168186f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "003ee7f3fb0e5d8f59863567122112168567ee050e53223544215c1196466ee0",
    "sha256" + debug_suffix: "05d3b3e3c9921339c4d7bf6e7010d94c878b3b9351b3a0e89663515ac2c157cb",
  ],
  "kernels_torchao": [
    "sha256": "f7f3525bef38fd632618b0a49321a2c745f6300627485c10df5a76a753d52acf",
    "sha256" + debug_suffix: "97058e91b31821b0a6dc78f87087d1ea72a26eecf931689753b0f23083f5717b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9dee62705fe760d1ce997c59327ac98293d1bdbd9973a296493788e25349887a",
    "sha256" + debug_suffix: "c5b2924bd0bda83a054ce250e7ca02183c1f69b1531244d11173a736f346e498",
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
