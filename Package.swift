// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260806"
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
    "sha256": "166d8144647bb4822df2ac35e0365c5f2bedbc5134270de26da7a1f71e0516a1",
    "sha256" + debug_suffix: "e50f248ed5ebf1bf249fbab1c3bed209674e2e25ab0c74cb3edd7e07fb1ed0fa",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "11a05cb33a71bd77392841218f14d5d8e72df79f9ddfe5d3e2cd16dae51c38de",
    "sha256" + debug_suffix: "62ad485f33b08bf2a4217909d227ad25f37b19b8e1c769c66535179512557642",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4deca30f48d4bb78a3453ad8b59281df5ecbb671c261e338a9e46b081176744d",
    "sha256" + debug_suffix: "3b0510ad48e2c7c9e1e8df634dd32a770a55711175b5693298169fa409999afb",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "385ba3a011a9c49abf7424964cdf91b76cd915237b2d0d0fb549d8b668c3acd2",
    "sha256" + debug_suffix: "8e812f3f99190f4de0c75c0c5bc877811be052fd4012b97f240c5bd60e34d0cc",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f3b1321c75ddc42c13d70e9af7f56a1cff1bb891b88b08c6bdafe10ff0f654b8",
    "sha256" + debug_suffix: "261988a4ff038a23f8003c76c7a113345b0aec3c5c396c9ffa36c39eb4c185ba",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9ecd576945d77407bdb5bd13d512101b219e8623dee681ed3c97f31522adee50",
    "sha256" + debug_suffix: "c1b282f27c407a9ec677570fbdb277a571f0934621f5c9b93e256730a3d69272",
  ],
  "kernels_optimized": [
    "sha256": "5be01d34507b7b9d9642b1e2b57739fd83e1911588aac621af0c517c099e4861",
    "sha256" + debug_suffix: "e96356ab978d9db05698fc69c4b3f6705e63ecd8b725c32cd599b506b4b06a98",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5ffb34b42181704291bec88a74d2f07d47ef15dfae608787af0d6635e2403c2a",
    "sha256" + debug_suffix: "8a98ee2ff22d90ec99f0f4c860ab6a879f6bd055f2c1c4026a5540219e0e9da9",
  ],
  "kernels_torchao": [
    "sha256": "3697fd4df464826963b63fe5abb37dfab1876f47e2cc126548c801323f234739",
    "sha256" + debug_suffix: "9534707b6d4688c62713654712932929c1dcd731996a73a9b01e2774047f1227",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2455639d113e116c3a816185f709ae33798af914279410e13da15a7711f8744d",
    "sha256" + debug_suffix: "b7fd5287b1142eca3865c86fbc901fa10c7d386a386edb922ab4a705ef7b3619",
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
