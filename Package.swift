// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250709"
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
    "sha256": "a2f2c008e94d257fe8dda872dfb137e958edcf3f1448cbd34083f44b1dc7974f",
    "sha256" + debug_suffix: "c6795d29ea1903c9db96c16de9dcbec51db2255869a79954e7cc04981097ca08",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d769dd04ef13da6a385794af5c868a2294ad163380f17c5b30cd404d7aea8c2c",
    "sha256" + debug_suffix: "b75780e794186425314a81195a1d00c067069aaf6bdd45a72f9990dba7ef841a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ae00a4277001440d19a3aedd951fbb1632e041fa6a63fb933f5c5aa248b1c9b1",
    "sha256" + debug_suffix: "9644b6f71fdb38304b11e2ae3292aace8fee71f3228d4edbea60048dd9e2c420",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e7b6af29fbfce6475b1834efef3ae5384fc7e0282c6ccce11d379c091c1cd90e",
    "sha256" + debug_suffix: "67ed179343136d3fb90b48fb9fe970913a266c8bdb317966f9b0617b84a093cd",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "075790da721d0686bebd59e1e54dfd6af18932fa07cb833c67a41dff4264ffe1",
    "sha256" + debug_suffix: "de72eefcbf96e635b4bf8d16dc969cb846a28810eb2d53fe813a58cad98031ff",
  ],
  "kernels_optimized": [
    "sha256": "a5cf42025be99bcd1dca3739fe74fde164d2e2d86a4671f591de780a739e7273",
    "sha256" + debug_suffix: "734b07f9faf05ba0e11d495aafa4e1a5df4196fc95056dc429050fd88f280f6e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "537e49acb58f4e66c76d2a6689edc10f96360d3ad53ddbe43f845e8876e23b67",
    "sha256" + debug_suffix: "11fe0d18e9d02ddfa8b536f4ae72bb0695252014c1547f9e63e36edd55f1a565",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3aa153f491f8e8c56ea0a59c8a4d97e76d807ff9fca122f0f1d381c7ca877fa2",
    "sha256" + debug_suffix: "b9300415455209e42d07584c241f5e69123f7c86e98b3ad352fd251310022225",
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
