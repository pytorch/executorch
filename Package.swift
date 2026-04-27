// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260427"
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
    "sha256": "4c8a0a165ca4547c13d1208134d280c7b23e402e9b140d306d82137bc2c7b1b3",
    "sha256" + debug_suffix: "30cb826cb501d6dc40819ba5762c2f0e6ee795ae469d15e81896ecf2ec8055be",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5754e4974ff400311e2106621c956d00da7b01452549208250c1428ea92bda32",
    "sha256" + debug_suffix: "ee4e2da5aabfdb1c23830827df35c08209c0f9a25ff4a8207e02e6d40bbf0bf7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "49c6870d47e84ac029ffb337111e647f4383db43a4af60ccf48fedcc840c1527",
    "sha256" + debug_suffix: "6b22cd448ce59cc820b1ff3ba9439736b1d8f3c5a07a13507b1156114e91af7a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f063b52552ab3e26ce28f16684be4736a0a1003f15a41c25396d11b1871f8ffb",
    "sha256" + debug_suffix: "cafcdb9ebd99f1ad6fe9ddc45272f97bbfbd111a7892330cd95bc1ea12ea0c0c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "781e074996f97f4ff6f4091f72f0f0a75d75863fa859cb9d6e8d3c6784b01ea9",
    "sha256" + debug_suffix: "4b918f2d1697a713211ebed1faf164a3cd19eae6ebcae13138c34843d9e4bd36",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d11c4be85065bf05503034aff5414a7dd51515f0d55404ea766dcdf201ac6082",
    "sha256" + debug_suffix: "73f5d1770f64934d12da9eeb6e83d1ce0844fd9752a90d3bf25cdb9e33c10cab",
  ],
  "kernels_optimized": [
    "sha256": "bd5a107d4904657a123369191d02646313a5df593dc30dedd756923420d6c8c1",
    "sha256" + debug_suffix: "89bd59b533e5f4cd27a0922a8188210d67ea81a4d5554d79a2484dabc6d351e9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a1f8d8aab048cc46cbff34ddb3f25f5135e14abea9d24826fc1482b92f13f090",
    "sha256" + debug_suffix: "25e50a5abf74382928efb83ee4df4bf9af06dbda95225e525e68b262ecb65246",
  ],
  "kernels_torchao": [
    "sha256": "cfa34cf3949bfdbfd7005951606013e3fd3c0e6b650a8c4ac9a4a9e5ebab711c",
    "sha256" + debug_suffix: "f070aa80b36ed8b84cfbfe8eb1912c4bbebb5a9ebbefc210adabc7e8ff314c72",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "62da4a479e556e802bd3bfc3af8a6964ff400118b548adcc5962f69f6c67077b",
    "sha256" + debug_suffix: "0a1cfe7fc4dd2802714eeabc082408ffda22eb8c4467d38cf11585c421adaa07",
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
