// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260424"
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
    "sha256": "beadce811a35924ba0e1874382ffa54d385e701d870dcf5e49ca54549656e43c",
    "sha256" + debug_suffix: "a937d5de4a0bf8f2ae1f21f36cfc882ea23b0d62a9ac4af1955c8b47d5fa40a5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b1a484ac9ece08928d9e8c4c63f8f5a993b05b98c20302972da7a43dee23ab00",
    "sha256" + debug_suffix: "def87ed874ed8cda779a4c12f4826117d9458f427501a71f34e839ebcbcc9427",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "48311518b8055fee8a1a72ebd18dca1dd5a77b2b2c4a9f930eb84c68d7a359dd",
    "sha256" + debug_suffix: "a67c34dd3093ac12b32b525dd9041352bc28c03fed14d398c37c53b1c29b8826",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d340de32026ac5d58cf6de54e63a2daf1abae37c789508d483d361083dc1ffc9",
    "sha256" + debug_suffix: "33b6c9bce43cf3ee4dcc37f0467f0b9b6f87487dca8908f4a9f1f4f39906fb58",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5b2a16167f18aeaf76b36c7e6e1ebbe2ff0db8f8a5f104390d5425a5708385a2",
    "sha256" + debug_suffix: "abab3e475ebbf9a621b118f2ecce287dc92cd9350e148315a2194764e8bdce70",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f6776b87395b2382e3071f950ab8bb882b1251b9d4dcbcfc3718f4c98dee5e99",
    "sha256" + debug_suffix: "d644b3d1c66e25a2a90f72e37d9fa102f38432669330600f805e53f13ce17d9a",
  ],
  "kernels_optimized": [
    "sha256": "f76f21eb26df769736836b4e9441efb979813dbfb69de67ed3d99a9555846fcf",
    "sha256" + debug_suffix: "a297d0dd5a220861fe0e100b44cbde3b5166f205aaadb11de3789edaa7d35d3c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "34d4fefb00e708c35a1c5153527923d75a6b351286ac9082949e5f29c043c93b",
    "sha256" + debug_suffix: "8d14377e2923652b22339bfa9ed26530a62aeee12270476123664cd3740a407d",
  ],
  "kernels_torchao": [
    "sha256": "91b824ce298d0c52309d68b7d711c89d82fc3c326337d5b422ec56c08bfb4907",
    "sha256" + debug_suffix: "160f5543c3beaf47d4ca6c4d5f623c26619bc1d04e424bc14176cc252e750fc2",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4faa9c9a150b8e78d39243a65283cd2ce0c7b345e1381fcfccb67697b6662c9a",
    "sha256" + debug_suffix: "641ba1138f2ca792d4e0127b8c4cda6119f3d9f24ee01bdd9bae85912f9412f4",
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
