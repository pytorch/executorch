// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260126"
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
    "sha256": "5a7b712b68a0df1b35d2593a466f991e91306d26b3c820b146c79e954e34f354",
    "sha256" + debug_suffix: "df7c30be0fbb1b3169e553cbebdfce0a4c710c82d9d75b0079c5b881dff476bd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9c75d6e71c2a475b65585e7121d563913e0b6a4b93f5f91a7e1c3650b46b35c0",
    "sha256" + debug_suffix: "85d6312ae77de2807fca79c436b579e3bf5d25d63051a4597907fe7362294899",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d8361e8596a9ba5e000ce75f2eb06099c92990738ba5f1cb13491ad5f4a52fb4",
    "sha256" + debug_suffix: "17ddbeb808d5bca2c73fd20b72dd8a27534803baa18c334de6b38b85f9cd4f87",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "83a266ca0d8fa638a60bbd71e169034dd6ade3309140213d5e7ac9ef7a37bba9",
    "sha256" + debug_suffix: "52a59e3f99e7bce9ce990d44e57bc6861ca68378a87068a18a218bc9e11700e4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "92a05cdeb5bdbb1bf7601d66c76919f2a1f9385535c529bab61fb5fec3f41064",
    "sha256" + debug_suffix: "1628147c69c92012c15cff13cdbaf3cfb0acdff4eb4179fb5f0b5965116fe87b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "15b21447e68bca799c9cdc6086ec84d5f80d29e6adaec5cf2c11e31c043e16cd",
    "sha256" + debug_suffix: "72ceb780d000c997bd27d34d464de29d778cf1f2bf13f992337b9ff1c489b83a",
  ],
  "kernels_optimized": [
    "sha256": "9a1518fac929f6f686f76a3bffab308a6aebeb06fc5f555fdf3fbef039a5bd32",
    "sha256" + debug_suffix: "f1d8b9e427e80f17ee2eba5442a8e1b0a42da46c5889b9e5765d3d7ec2bb6eb8",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d2de9d4eef3bdacccdd271029a252e58ec80f0d31e7085624ecaf237e9e4c7f8",
    "sha256" + debug_suffix: "fdf0e385a0882d16a798a1ca703496ee95fd1332e93bab6026d241c32ca85d2f",
  ],
  "kernels_torchao": [
    "sha256": "624cc2056e58f3dfac713eef85edbcbd5a26685eea9dd98fa3688fb3af834504",
    "sha256" + debug_suffix: "9868d195ee055bb7a498a098bfefe32936d18ab974d1407c31ae4c623b4be519",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "01a05f14d511846a3ddbea363310a2ee8924e109be1d3c17c89f368615f632fa",
    "sha256" + debug_suffix: "e48598ca49bf17498dfb242828677354dbb0dad47dd0246a3c872757d6be0c6f",
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
