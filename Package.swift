// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260506"
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
    "sha256": "dcb1f95975a27f2f949a222c2ddf0b6e04be52a46a2e691e759cb41c46b08cf6",
    "sha256" + debug_suffix: "0bad0a9d71222261071fc90a79236f10ea0067c4640b37449d06f1df0fa91cd6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "96e75b08018a8154bbd9f2d59731f65b139fc7fe91b0fb667630f39226a05520",
    "sha256" + debug_suffix: "fa1e8cb8f030356e3338f25cd6b25950525484136a6f224f706d32fb335b0a66",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b1ed971bde1cd4b40ba4883fc50fc0346c05b68e63f89fa9f802f33308d4e8af",
    "sha256" + debug_suffix: "bf9c69fb209c783e45ea44b78b1ceee27fcf6afddd3299bce772a43201a8090f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d1adc724f3b24f32119f7dbfb32f46a609812dce1e09a806574317c0a2766f61",
    "sha256" + debug_suffix: "8acdc747bb07c29dd7437015c90d17081bc5497e1ca2dcfdf1ee6b6196e6e710",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "01eaca9a143b7fa31d031f02938898a5535d1fdef4499dff9e951e9e95759339",
    "sha256" + debug_suffix: "7a8aba93edb1de0fd6ff462436e49822c482d7f128d79b0ed932544495c395a9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "fdc912385cbea99598a52c83ddf757a0aee442c50ff1961ae26290001d58599a",
    "sha256" + debug_suffix: "275ce48db74bc2ef31969f0f7219d15587fb5a784ba62e7ee239c19c1f08cbcb",
  ],
  "kernels_optimized": [
    "sha256": "fe286a6b6b41ec0c5090c29da31c9bd903cc3bd5e278801b24e0420ee9efa89d",
    "sha256" + debug_suffix: "247f4beaf9946b3d2bb540758ca4bdf28ef55a7a1251548637bf52286f02dbbf",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ea92ad7b81e881ac9fc071ea7d2b7007b2e7ff6298db4599acaed3eeff790023",
    "sha256" + debug_suffix: "a4b8e629255bf84c410865b74ea547a3f72827281a26a5e62867135f9e9f9da5",
  ],
  "kernels_torchao": [
    "sha256": "47df0e7fcdb8f415c8a1006e62f8500b5af3bf80e276decd6255e0035d1a4fdf",
    "sha256" + debug_suffix: "95c22ddadc5e80cfe209cd97ddf5995f9c7cade196434fd48b9dc8da0ce04907",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "92a4d764e1c3cbbf71d4c7dbb8d78d19b0400375830433562702a516abae2121",
    "sha256" + debug_suffix: "8419c86c16621c0dc5614d04ad88837dc02d6318236ad492f4b59257352de99b",
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
