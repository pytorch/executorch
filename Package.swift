// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260117"
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
    "sha256": "4d270b85ae635fdd32695c7d0946dda8811fe24b721a25fce5aa00e27a6737c3",
    "sha256" + debug_suffix: "523c118b7f7bc0bcd85538831de9449587d6bd23bab91d4d95dfd6c66120364f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d4ba21b7813dfefd281701c20fa1eff090f68835156f68aba37308c0d7780ebf",
    "sha256" + debug_suffix: "1b88ff3505956fb064bc4ef821a2dda98842dcdcdb9ea058c43b9d78d345f2b0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "887f78dc037e2a18ec535c5d09300f0182318116bfb6571bce28657a40dedbe3",
    "sha256" + debug_suffix: "542285ca12fa6a59e5a49357accc20ba95c9e3c397355ec3eceee30dd6551be8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0be34ac7e7684677d025a5f642e24be6145abc8b182587a52d2b139256570482",
    "sha256" + debug_suffix: "3ef28f8c7da5cc83586597a99806b283efdf012e7c174a7fc87ada7367d33ba0",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "57ecf92f4a7083935ee3423759d6f81461e58fca77693589858964775901c02b",
    "sha256" + debug_suffix: "37ce5d25bc8d8016ee16aecf09b01f93d8177d372012dfb5401d30deac3b0ab9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8801e2425f676feeab7fc488ee12d6caebcf9d3e926a3a045efc065976c5218c",
    "sha256" + debug_suffix: "f38d73a42d8dd92be978b3f277c6f2452dd898ed6fc379e8ed5dcabe4b6a598c",
  ],
  "kernels_optimized": [
    "sha256": "2f85edfdf96cd482fd5f5700793ed99e37c3880e43f44ea16be570db77e54d3e",
    "sha256" + debug_suffix: "b198b9bf3fbe24e4d903afa17412da3f24cc80b930be07d8176638689e58ccdb",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f234fe679bfd5ab2cbd32a7ad296780cc35c0dfab9e5a48304bb22f9dbb1ad9d",
    "sha256" + debug_suffix: "be2640475b50a553177811a3fb018eb087cad741ad1fcb9ced6c3ef4b350a864",
  ],
  "kernels_torchao": [
    "sha256": "0f0f8dba43f66069edb52290cd5bf831c3e3532d9e86f73a7c4784975a27cb4a",
    "sha256" + debug_suffix: "df93232355e8205ead175d7a2fc8188f19f5bd2b30a4b9c3f0fa41ab4bf5c0ba",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "53b728db2ff96ff5e770d80331cf9f7e655a6bd680ec3ec482268002ed263d6c",
    "sha256" + debug_suffix: "8c47bb96ff5dfb3b25b827205b72e666fba81b755998f8d6e45df2fe5125abd9",
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
