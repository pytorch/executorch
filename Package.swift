// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250817"
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
    "sha256": "b1205e7a2f1a9e22d1bf6ded524ed9f4aae41cbee931e699e026f856b8cc4e48",
    "sha256" + debug_suffix: "261e7d60d04163332912d9172e5b1f73345f07c527a72366b3bed467e75b1796",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b71466a969faf8b8bdbadd93a966439e705bc8571fd483479b036ccb9d70cd9c",
    "sha256" + debug_suffix: "4601febd18718681c3c048cba657a824357d2d273131f316d9c5c6c900083f69",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7a1fa70ef586e3d7a938505831515b75c5c58ac42747c297eeb6dc7abe9daa7f",
    "sha256" + debug_suffix: "6b7adae957f3262d04ca1805290db78f5dfb963668efd39149fda1c97a7e22b8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "bcd267969f3afdb8fcd401ae9c4e7c885deda69d28386696adbb4a353440a663",
    "sha256" + debug_suffix: "5f39884dadd1cd8b5831299a8974afcb51047c32cb39637f58af707f73aab971",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "6c36ef22aed9f25781a9320289339bfdb9791f1121ffbe1ca4308001c7fc2351",
    "sha256" + debug_suffix: "448d444b7b3aae00d08a793700e9c73b876e8fd0a98db0621362bce3700077ca",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b58efcba63ca44fd3645360d8f57707278588ee0f15e873d7ae2da531c0a5b04",
    "sha256" + debug_suffix: "593c3677107c1732f145693150a9bdeeef86272d1ff68d04c1120bcc40c311eb",
  ],
  "kernels_optimized": [
    "sha256": "dc8aa9617393c8bd00d95d9dc507da669433e66500d24a94f49f693162a5b375",
    "sha256" + debug_suffix: "52261e877baa0a8707873567da8f6747d6931f53e5856bcbd6e5be86dd97cfe5",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b0ecdbb7ce0c0b19cdd39053bee578a2ef4fbf3e7db2aa9e7a91ae0735601bb1",
    "sha256" + debug_suffix: "1d3fb383dd5de00c75f44f5688e9ee9897cb1e537e5579e0c21ca6fb1a1cc42d",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1f0389a0725b9015006c147cf07c92e504ce6ef143abf95183de51bd937e9fe1",
    "sha256" + debug_suffix: "b5d190a4e5fb78f71246b11d7fe270c5a4c2fcf0787a76b6fe5bb16a6749c96e",
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
