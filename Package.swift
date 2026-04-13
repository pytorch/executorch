// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260413"
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
    "sha256": "4c71289cbd34a1402decb97992ed7f2c910aef0077a1b18b34d8a49f38cb7d51",
    "sha256" + debug_suffix: "7c6e356ba5cdc3ed101e1e06a049a855ce4e8f025e0c469c18afa5fa4acd4e4e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9b96035a8fe73444795ca29bcd5a64ca0db433dc6964ce78037ea60c4c1f32ce",
    "sha256" + debug_suffix: "93eaaef6cf2cb26dbd70a0e6ead09ae6cedd5133cf9059346980c19a74cb3653",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "852c6fbaaf7359baa2edcaa626f0e376f6721882033f69215e9aaf6a5ad9ade2",
    "sha256" + debug_suffix: "6c9e184effe9b9808ca97a4487a4a090d407df38dd535dcbfb82d052726fc158",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9022f0929a38ae95096507676264a500b5b5c3ed34f0f0593390e5310301c092",
    "sha256" + debug_suffix: "bbc6b7e894a726dccb73d9107ef2b79a1de1459372ba74b5de73344312c2f877",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c83c7dcdb6953e82722ffb09173e8d37e964dab0bbe8fdbef321876966925644",
    "sha256" + debug_suffix: "b3536a169f0af6901c344d4de956d5c084505490bd89567079ff63f73693e774",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9af574d28e08fb8b483591f8e7ef434f8585b614d6cb9559e9e29ce18e1d9788",
    "sha256" + debug_suffix: "2c00ad6e7ca526a58fd6c29ec6816ca1f6013b17847c35d0415407c8190d5a7c",
  ],
  "kernels_optimized": [
    "sha256": "fb5b83864d47af532125a4aac901a88b8550d36dbbcf8f866d75797dbcc71dd1",
    "sha256" + debug_suffix: "3380fde0c6ff5439c5ee643438a234c84bdfb425b9a87b5e24eef0ab672897e7",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f09e5ce1f3acc4b3b3a96f8ffb8de2751e437c24d460247e5e2eb2f4a0c6c6aa",
    "sha256" + debug_suffix: "96e5a57459655f1a5d98ef640c789bf1011260a95ad780cc1a5e91057180e961",
  ],
  "kernels_torchao": [
    "sha256": "72fa271daacd497b9e295351e7ab9a36bceefd66e53693bdddcc2dab2bdc3b89",
    "sha256" + debug_suffix: "a218cdd00b972e7e8be9577a772815f4d60ffe5bb25b5e26548256da4363d0ca",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "69a1bf5c90d1ae78038f6fe3745ee7c15b663bb293c390b2d38b6e5e1f98e0bc",
    "sha256" + debug_suffix: "fa8d30e95278f8e12da9e5488fe2de27ae5cd6084f4281b994b97413f5df074d",
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
