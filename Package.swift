// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260521"
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
    "sha256": "0c759c360d62d6ce48c81f487173208715f8ec30e1621a28ca0f2290609119c7",
    "sha256" + debug_suffix: "1079e3069bbea9f71c408694361e6094031ac8d1cc5665ea3efe3b1fd4a7fe02",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "90c392924cc04ec77696f58c02814c04ad3f7c4b8ff24504892a90e3aeaf7bd1",
    "sha256" + debug_suffix: "b110723192ddee6f51afa01962ada971c13d569aeaa4ade6e68a06facbae8029",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2edd3f793c3787df774da99ee02329f724e9979cb8e94cf9defa2a070d08cce6",
    "sha256" + debug_suffix: "e9376eac3c7ff4a6f566de495f0e8d4cd565f6dbe6a0a74af3d2d7977bfdfde6",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "10f92e2e63627949e9023e5c6c7e01ac79435778ce868ef06bb893132595c463",
    "sha256" + debug_suffix: "856428c9ba4a5b95b362b48a96b16b088479f925ad69fa0cd6f76e372207f172",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ac754f59e77c31cb05e4c519fc8e29e0194908106c719bdff2862ff98a0b1bb6",
    "sha256" + debug_suffix: "a5be380fcff931e1e70a6ba3092f3167037e7369124a360a7cd579fb5e1b364b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "bf89d5f4f2e4c9ba445ef0daf62eebaba1b0a0d49aacfc07085473dcc47e05cf",
    "sha256" + debug_suffix: "101d18f64f681884713f2a622b041347fcd18b0edf5c7bf22fcaeec1194c980a",
  ],
  "kernels_optimized": [
    "sha256": "e3e551b0ffe2ef83475ee2ca7460d86a14a2067823f644de6706f551d07a61b4",
    "sha256" + debug_suffix: "983bd3cc4cbab657574d5f08a4cd8a6eb4cc36da5b32fa0d4164f880dbbd3fba",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "bd486eae30188d3ddd0a2f62ef6ac2546973c9eb20851e293a55f371b009b5ad",
    "sha256" + debug_suffix: "17a22aba07279da558b738cfdce65a21ec8b94bc9b193b432a07f028d984ba56",
  ],
  "kernels_torchao": [
    "sha256": "c7e650b9001df6b44df44d62d6157c76d02728866ea8a51f5f5fa638b14d6724",
    "sha256" + debug_suffix: "ac4a24216e3964f6801f7c6f7e837ec11595d5186b1c01a938c32f9d4a0f7c73",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cee760ba4dbb73e8153d52b216a30cabec4cc083b900030adecf46538aabe2b9",
    "sha256" + debug_suffix: "70fac1e069831663c8469bec826ed25c0170125944dfcdedc644508c0c4ba76e",
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
