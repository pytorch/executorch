// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250606"
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
    "sha256": "513ae1a9aa833cfa90c76784e370976f5a4a0656766e896d3c4ffa9ec253dcdd",
    "sha256" + debug_suffix: "8a8de6a415922a394fbe5d2e61ce41e4be84be2d30e3a27333a1c6f20d535304",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "52affadf87c878669b6ab0f1f6887fdebba3898f5fe5f027dc2ddb00df48eee4",
    "sha256" + debug_suffix: "57c4aa4fe7914461bd7904b33fac92cf70881800e48294c0ce400f494a84331a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7283f094b639e867b5af13bc614caa5f43e5600c579538b1d1c86fcb9512c930",
    "sha256" + debug_suffix: "2588a717ef72f553d7515babe241ff712015e3bd8f5747e1a7cad6a9182c3a94",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "950b5c85785488c8504bc891bf4191a61fc8431b2711d27ad048180cf9b8d831",
    "sha256" + debug_suffix: "5905257b5eea6cb9de8cc013bf09698b9e2b32981d28850d68f182fdd9a7a195",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "b80d45d75451cc7dc997058c11c97cc670b1ef09dd6be6bf071b8233e5ad83f0",
    "sha256" + debug_suffix: "fafb7731101b4481f9fdf8c1f74196159c4ac9fec3cd84ae510b6652dc88fe34",
  ],
  "kernels_optimized": [
    "sha256": "f50e2e673bf121fd46cffb75df0de88c2a0405ac0fdf78270b9322e27d4ef9c7",
    "sha256" + debug_suffix: "8af58f85498c6a6b55bdfeb5da4f5297e50fe6c52fdaa6ff3febc5e360af49b9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f6652fa7603096c578e2233d55bbd1c7401d53cf0c04e8e3d4658856a04bec2c",
    "sha256" + debug_suffix: "0a03b91c9123e06e24dac5c229fe982a613fff6c20b07e4825c0478727221a69",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5f1ff323bb775d4d3b7dfe8a1fbba8e129edd500b15c112ee1287eb101a53e54",
    "sha256" + debug_suffix: "081b086ae8e4622ab4cef7207e7b2a165cd580eedcae6f0a6816ad83cf1c931f",
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
