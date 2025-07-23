// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250723"
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
    "sha256": "5394e62a9785406abd78df884834530a7a444aee4eaf03c3fb86621a327cdff1",
    "sha256" + debug_suffix: "1e54a0dc87f56314a11fe4c4ac8c4fa239a6bf8946a5bdd53e7ea3396ec00026",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3e55828a159d2dee7dc8bf48655eccd0d3f99da6a94787c44e9c4e1fab313c7c",
    "sha256" + debug_suffix: "90e73c528e8f4b4047925b97a982ecfc118161b9eb0cf88be330bc0f2446bf93",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "05835eebe7d3279919ac530e8ff309b3878681221052826e3329d2b3d6854d92",
    "sha256" + debug_suffix: "d88be68cdf17a3cd542d3ab8497ecd9f4f1de53e1e40913833aec9b69dfe2880",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "02528a08631844f29226e6b39b5311bb340caef2b324cd545d389260255e0bb4",
    "sha256" + debug_suffix: "d9fc87777b366d88851f1ed4a794149f567d0eb0dc2f1b166bff3fe32e06aa13",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e7f71e84e30897dc6a8d3fef0a3e332dc9b006d7f6a323f3ef931e90f33d20ec",
    "sha256" + debug_suffix: "14128d0232a25c12040d5d84e2d44552951254fff7435aea948b4e28569d432c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6e2904bb2df525fd19049eb78b2919ba75ae0f0094e8a867019614ae22164303",
    "sha256" + debug_suffix: "74eb2321c9a74469f9b30fb3264677ec276a7be739d354a9cbac145a59e4c7d4",
  ],
  "kernels_optimized": [
    "sha256": "b145a04c552c076217bf3f8d3d44157b7e178304c95cbb75b637c173e04d04b4",
    "sha256" + debug_suffix: "cff212c3c7577fc2ed50d7ca982193bcc04675d5ab01be78ecddbe8385eaa1d4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3adf9b4adaa90a910173cc7455513ddb3e25055af35e5ff67551de2e8546adcb",
    "sha256" + debug_suffix: "703e0d864c348c82f64bc9630c9d476975ca2b2116c2a067ab695049d997b7de",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4ef4878fe1b5fd7c7529a2e4c814d7bff01f8c0fa22471deee8ebc02359717b8",
    "sha256" + debug_suffix: "856b9b95eb3700789e8113e8e9d036023fcbeefb94091ab0235af8b0770c8ea2",
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
